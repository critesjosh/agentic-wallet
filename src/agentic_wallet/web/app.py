"""FastAPI read-only demo.

Exposes the deterministic harness, registry, and workflow state machine as a
small typed API, and serves a single static page. No signing, no submission,
no key custody. Model inference is added later behind ``InferenceProvider``.
"""

from __future__ import annotations

import os
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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

_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "portfolio_base_watch.json"
_STATIC = Path(__file__).resolve().parent / "static"
_ROOT = Path(__file__).resolve().parents[3]

# Local development convenience only. Values already present in the process
# environment win, and no secret is ever returned by an API endpoint.
load_dotenv(_ROOT / ".env", override=False)

app = FastAPI(title="Agentic Wallet (read-only demo)")
_harness = MockReadOnlyHarness.from_fixture(_FIXTURE)


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
        model_id = os.getenv("AGENTIC_WALLET_MODEL_ID", "google/gemma-4-E2B")
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
    return _is_loopback(request) and _host_header_is_loopback(request)


def _require_local_debug(request: Request) -> None:
    if not _debug_transcripts_enabled() or not _is_local_debug_request(request):
        raise HTTPException(status_code=404, detail="Not found")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4_000)
    allow_remote_inference: bool = False


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
    if _debug_transcripts_enabled() and _is_local_debug_request(request):
        debug_transcripts.record(req.session_id, req.message, response)
    return response


@app.get("/capabilities")
async def capabilities() -> dict:
    return {
        "inference_provider": _provider_name,
        "inference_location": _inference_location,
        "inference_host": _inference_host,
        "model_id": _model_id,
        "remote_consent_required": _inference_location == "remote-model",
        "read_only": True,
        "signing": False,
        "debug_transcripts": _debug_transcripts_enabled(),
        "native_constrained_decoding": bool(
            _chat.provider and _chat.provider.native_constrained_decoding
        ),
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
