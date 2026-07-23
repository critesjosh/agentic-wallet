"""FastAPI wallet capability and transaction proof of concept.

The default remains read-only.  An explicit configuration can additionally
enable the isolated, user-confirmed native-transfer path.
"""

from __future__ import annotations

import os
from ipaddress import ip_address
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..harness import HarnessError, MockReadOnlyHarness
from ..providers import (
    LlamaCppHTTPProvider,
    LocalTransformersProvider,
    OllamaProvider,
    OpenRouterProvider,
)
from ..registry import BASE_REGISTRY, RegistryError
from ..schemas.tool_call import ToolCall
from ..state_machine import TERMINAL, TRANSITIONS, WorkflowState
from .chat import DemoChatAgent
from .transcripts import debug_transcripts
from .transactions import (
    BrowserSessionStore,
    TransactionController,
    TransactionFlowError,
    configured_transaction_controller,
)

_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "portfolio_base_watch.json"
_STATIC = Path(__file__).resolve().parent / "static"
_ROOT = Path(__file__).resolve().parents[3]

# Local development convenience only. Values already present in the process
# environment win, and no secret is ever returned by an API endpoint.
load_dotenv(_ROOT / ".env", override=False)

app = FastAPI(title="Agentic Wallet")
_harness = MockReadOnlyHarness.from_fixture(_FIXTURE)
browser_sessions = BrowserSessionStore()


def _inference_endpoint(provider_name: str) -> str | None:
    if provider_name == "ollama":
        return os.getenv(
            "AGENTIC_WALLET_INFERENCE_BASE_URL", "http://127.0.0.1:11434"
        )
    if provider_name == "llama-cpp-http":
        return os.getenv(
            "AGENTIC_WALLET_INFERENCE_BASE_URL", "http://127.0.0.1:18080"
        )
    if provider_name == "openrouter":
        return "https://openrouter.ai/api/v1"
    return None


def _hostname_is_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _classify_inference_location(
    provider_name: str, endpoint: str | None
) -> str:
    if provider_name == "keyword":
        return "server-deterministic"
    if provider_name == "local-transformers":
        return "server-model"
    hostname = urlsplit(endpoint).hostname if endpoint else None
    return "server-model" if _hostname_is_loopback(hostname) else "remote-model"


def _build_chat_agent() -> DemoChatAgent:
    provider_name = os.getenv("AGENTIC_WALLET_INFERENCE_PROVIDER", "keyword")
    if provider_name == "keyword":
        return DemoChatAgent(_harness)
    if provider_name == "local-transformers":
        model_id = os.getenv("AGENTIC_WALLET_MODEL_ID", "google/gemma-4-E2B-it")
        if os.getenv(
            "AGENTIC_WALLET_ALLOW_UNCONSTRAINED_INFERENCE", "false"
        ).lower() not in {"1", "true", "yes"}:
            raise RuntimeError(
                "local-transformers lacks native constrained decoding and is "
                "development-only; set AGENTIC_WALLET_ALLOW_UNCONSTRAINED_INFERENCE=true "
                "only for explicit evaluation"
            )
        return DemoChatAgent(
            _harness, provider=LocalTransformersProvider(model_id=model_id)
        )
    if provider_name == "ollama":
        model_id = os.getenv("AGENTIC_WALLET_MODEL_ID", "gemma4:e2b")
        base_url = _inference_endpoint(provider_name)
        assert base_url is not None
        return DemoChatAgent(
            _harness, provider=OllamaProvider(model=model_id, base_url=base_url)
        )
    if provider_name == "llama-cpp-http":
        base_url = _inference_endpoint(provider_name)
        assert base_url is not None
        return DemoChatAgent(
            _harness, provider=LlamaCppHTTPProvider(base_url=base_url)
        )
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_id = os.getenv("AGENTIC_WALLET_MODEL_ID", "")
        if not model_id:
            raise RuntimeError(
                "AGENTIC_WALLET_MODEL_ID is required for the OpenRouter provider"
            )
        return DemoChatAgent(
            _harness,
            provider=OpenRouterProvider(
                api_key=api_key,
                model=model_id,
                data_collection=os.getenv(
                    "AGENTIC_WALLET_OPENROUTER_DATA_COLLECTION", "deny"
                ),
                zero_data_retention=os.getenv(
                    "AGENTIC_WALLET_OPENROUTER_ZDR", "false"
                ).lower()
                in {"1", "true", "yes"},
            ),
        )
    raise RuntimeError(
        "AGENTIC_WALLET_INFERENCE_PROVIDER must be 'keyword', "
        "'local-transformers', 'ollama', 'llama-cpp-http', or 'openrouter'"
    )


_chat = _build_chat_agent()
_provider_name = os.getenv("AGENTIC_WALLET_INFERENCE_PROVIDER", "keyword")
_model_id = os.getenv("AGENTIC_WALLET_MODEL_ID")
_inference_base_url = _inference_endpoint(_provider_name)
_inference_host = (
    urlsplit(_inference_base_url).hostname if _inference_base_url else None
)
_inference_location = _classify_inference_location(
    _provider_name, _inference_base_url
)

_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}

_SESSION_COOKIE = "agentic_wallet_session"


def _transactions_enabled() -> bool:
    return os.getenv("AGENTIC_WALLET_TRANSACTION_ENABLED", "false").lower() in {
        "1", "true", "yes"
    }


def _build_transaction_controller() -> TransactionController | None:
    """Only construct the signer boundary for an explicitly enabled deployment."""

    if not _transactions_enabled():
        return None
    rpc_url = os.getenv("AGENTIC_WALLET_SIGNER_RPC_URL", "")
    encoded_secret = os.getenv("AGENTIC_WALLET_APPROVAL_HMAC_KEY", "")
    if not rpc_url or not encoded_secret:
        # Do not run a partly configured transaction surface that could make a
        # user believe an action is live when its signer is unavailable.
        return None
    try:
        # The web process checks only backend availability. Key material is
        # loaded exclusively inside the isolated stdio signer process.
        from ..signer.capability import decode_approval_hmac_key
        from ..signer.key_store import require_secure_keyring_backend

        require_secure_keyring_backend()
        secret = decode_approval_hmac_key(encoded_secret)
    except Exception:
        return None
    return configured_transaction_controller(
        registry=BASE_REGISTRY, rpc_url=rpc_url, hmac_secret=secret
    )


_transactions = _build_transaction_controller()
# The chat path can request a *review* only when the deployment has a complete
# deterministic transaction controller.  It never receives an approval tool.
_chat.transfer_requests_enabled = _transactions is not None


def _require_transaction_controller() -> TransactionController:
    if _transactions is None:
        raise HTTPException(
            status_code=503,
            detail="transaction signing is not configured for this deployment",
        )
    return _transactions


async def _transaction_ready_for_request(request: Request) -> bool:
    return bool(
        _transactions is not None
        and _is_local_debug_request(request)
        and await _transactions.ready()
    )


def _require_consequential_session(request: Request) -> str:
    if not _is_local_debug_request(request):
        raise HTTPException(
            status_code=403,
            detail="live transaction endpoints are local-only until user authentication exists",
        )
    try:
        return browser_sessions.require(
            request.cookies.get(_SESSION_COOKIE), request.headers.get("X-CSRF-Token")
        ).session_id
    except TransactionFlowError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _transaction_error(exc: TransactionFlowError) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


@app.middleware("http")
async def _transaction_response_headers(request: Request, call_next):
    """Keep approval digests and review summaries out of browser/proxy caches."""

    response = await call_next(request)
    if request.url.path == "/sessions" or request.url.path.startswith("/transactions/"):
        response.headers.update(_NO_STORE_HEADERS)
    return response


def _debug_transcripts_enabled() -> bool:
    return os.getenv("AGENTIC_WALLET_DEBUG_TRANSCRIPTS", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _is_loopback(request: Request) -> bool:
    if request.client is None:
        return False
    try:
        return ip_address(request.client.host).is_loopback
    except ValueError:
        return False


def _host_header_is_loopback(request: Request) -> bool:
    host = request.headers.get("host", "")
    try:
        hostname = urlsplit(f"//{host}").hostname
    except ValueError:
        return False
    return _hostname_is_loopback(hostname)


def _is_local_debug_request(request: Request) -> bool:
    proxy_headers = {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
    }
    return (
        _is_loopback(request)
        and _host_header_is_loopback(request)
        and not proxy_headers.intersection(request.headers)
    )


def _require_local_debug(request: Request) -> None:
    if not _debug_transcripts_enabled() or not _is_local_debug_request(request):
        raise HTTPException(status_code=404, detail="Not found")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4_000)
    allow_remote_inference: bool = False


class TransferProposalRequest(BaseModel):
    chain_id: int = Field(gt=0)
    recipient: str = Field(min_length=42, max_length=42)
    amount_base_units: str = Field(pattern=r"^(0|[1-9]\d*)$")


class ExactApprovalRequest(BaseModel):
    workflow_id: str = Field(min_length=16, max_length=128)
    envelope_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> dict:
    """The wallet's only user interface. Read-only in this demo."""
    if _inference_location == "remote-model" and not req.allow_remote_inference:
        response = {
            "reply": (
                "Remote inference is configured, but this request was not sent "
                "because remote processing was not explicitly allowed."
            ),
            "state": WorkflowState.IDLE.value,
            "data": None,
        }
    else:
        response = _chat.respond(req.session_id, req.message)
    if not await _transaction_ready_for_request(request):
        # A remotely reachable read-only deployment never emits an actionable
        # review request. A local deployment also remains read-only until its
        # isolated signer key and Base RPC both pass readiness.
        response["transaction_request"] = None
        response["transaction_status_request"] = None
    if _debug_transcripts_enabled() and _is_local_debug_request(request):
        debug_transcripts.record(req.session_id, req.message, response)
    return response


@app.post("/sessions")
async def create_browser_session(request: Request, response: Response) -> dict:
    """Mint the cookie and anti-CSRF token required for live actions.

    The opaque session identifier is HttpOnly; only the independently random
    CSRF token is returned to page JavaScript.  The optional chat session ID is
    not an authorization credential and merely keeps the existing chat ledger
    aligned with the transaction review for this browser.
    """

    if not await _transaction_ready_for_request(request):
        return {
            "csrf_token": None,
            "chat_session_id": secrets.token_urlsafe(24),
            "expires_at": None,
        }
    session = browser_sessions.create()
    response.set_cookie(
        _SESSION_COOKIE,
        session.session_id,
        httponly=True,
        samesite="strict",
        secure=os.getenv("AGENTIC_WALLET_SESSION_SECURE", "true").lower()
        not in {"0", "false", "no"},
        path="/",
    )
    response.headers.update(_NO_STORE_HEADERS)
    return {
        "csrf_token": session.csrf_token,
        "chat_session_id": session.chat_session_id,
        "expires_at": session.expires_at,
    }


@app.post("/transactions/propose")
async def propose_transfer(req: TransferProposalRequest, request: Request) -> dict:
    """Create a review-only exact native-transfer proposal; never approve it."""

    session_id = _require_consequential_session(request)
    try:
        return await _require_transaction_controller().propose_native_transfer(
            session_id=session_id,
            chain_id=req.chain_id,
            recipient=req.recipient,
            amount_base_units=req.amount_base_units,
        )
    except TransactionFlowError as exc:
        raise _transaction_error(exc) from exc


@app.post("/transactions/approve")
async def approve_transfer(req: ExactApprovalRequest, request: Request) -> dict:
    """Record explicit approval for exactly the displayed digest only."""

    session_id = _require_consequential_session(request)
    try:
        return await _require_transaction_controller().approve(
            session_id=session_id,
            workflow_id=req.workflow_id,
            envelope_digest=req.envelope_digest,
        )
    except TransactionFlowError as exc:
        raise _transaction_error(exc) from exc


@app.post("/transactions/submit")
async def submit_transfer(req: ExactApprovalRequest, request: Request) -> dict:
    """Freshness-check and hand an approved digest to the isolated signer."""

    session_id = _require_consequential_session(request)
    try:
        return await _require_transaction_controller().submit(
            session_id=session_id,
            workflow_id=req.workflow_id,
            envelope_digest=req.envelope_digest,
        )
    except TransactionFlowError as exc:
        raise _transaction_error(exc) from exc


@app.get("/transactions/{transaction_hash}")
async def transaction_status(transaction_hash: str, request: Request) -> dict:
    """Session-scoped status lookup and receipt refresh with a trusted explorer URL."""

    session_id = _require_consequential_session(request)
    try:
        return await _require_transaction_controller().transaction_status(
            session_id=session_id, transaction_hash=transaction_hash
        )
    except TransactionFlowError as exc:
        raise _transaction_error(exc) from exc


@app.get("/capabilities")
async def capabilities(request: Request) -> dict:
    signing_ready = await _transaction_ready_for_request(request)
    return {
        "inference_provider": _provider_name,
        "inference_location": _inference_location,
        "inference_host": _inference_host,
        "model_id": _model_id,
        "remote_consent_required": _inference_location == "remote-model",
        "read_only": not signing_ready,
        "signing": signing_ready,
        "transaction_scope": "native_eip1559_transfer" if signing_ready else None,
        "debug_transcripts": _debug_transcripts_enabled(),
        "native_constrained_decoding": bool(
            _chat.provider and _chat.provider.native_constrained_decoding
        ),
        "two_stage_tool_proposals": bool(_chat.provider),
        "bounded_repair_attempts": 1 if _chat.provider else 0,
        "typed_conversation_ledger": True,
        "grounded_result_narration": True,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/portfolio")
async def portfolio() -> dict:
    return _harness.get_portfolio().model_dump()


@app.get("/registry")
async def registry() -> dict:
    return {
        "version_digest": BASE_REGISTRY.version_digest(),
        "entries": [e.__dict__ for e in BASE_REGISTRY.entries()],
    }


@app.get("/balance/{asset_id}")
async def balance(asset_id: str) -> dict:
    try:
        if asset_id == "base:native":
            amt = _harness.get_native_balance()
        else:
            amt = _harness.get_token_balance(asset_id)
    except (RegistryError, HarnessError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"asset_id": asset_id, "amount": amt.model_dump()}


@app.get("/states")
async def states() -> dict:
    return {
        "states": [s.value for s in WorkflowState],
        "terminal": sorted(s.value for s in TERMINAL),
        "transitions": {
            s.value: sorted(t.value for t in targets) for s, targets in TRANSITIONS.items()
        },
    }


@app.post("/validate/tool-call")
async def validate_tool_call(payload: dict) -> dict:
    """Demonstrates fail-closed schema validation of model output."""
    try:
        tc = ToolCall.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"valid": True, "tool_call": tc.model_dump()}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/project")
async def project_status() -> HTMLResponse:
    """Human-readable architecture, delivery status, and testing guide."""
    return HTMLResponse((_STATIC / "project.html").read_text())


@app.get("/debug/transcripts")
async def debug_transcript_page(request: Request) -> HTMLResponse:
    _require_local_debug(request)
    return HTMLResponse(
        (_STATIC / "debug-transcripts.html").read_text(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/debug/transcripts.json")
async def debug_transcript_data(request: Request) -> JSONResponse:
    _require_local_debug(request)
    return JSONResponse(debug_transcripts.snapshot(), headers=_NO_STORE_HEADERS)


@app.delete("/debug/transcripts")
async def clear_debug_transcripts(request: Request) -> JSONResponse:
    _require_local_debug(request)
    debug_transcripts.clear()
    return JSONResponse({"cleared": True}, headers=_NO_STORE_HEADERS)
