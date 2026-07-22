from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from agentic_wallet.web.app import app  # noqa: E402
import agentic_wallet.web.app as web_app  # noqa: E402
from agentic_wallet.harness import MockReadOnlyHarness  # noqa: E402
from agentic_wallet.inference import ScriptedProvider  # noqa: E402
from agentic_wallet.web.chat import DemoChatAgent  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1"
    ) as test_client:
        yield test_client


@pytest.mark.anyio
async def test_health(client):
    assert (await client.get("/health")).json() == {"status": "ok"}


@pytest.mark.anyio
async def test_project_status_page_explains_boundaries(client):
    response = await client.get("/project")
    assert response.status_code == 200
    assert "Where the project stands" in response.text
    assert "Model proposes" in response.text
    assert "No signing or key custody" in response.text


@pytest.mark.anyio
async def test_debug_transcript_is_opt_in(client, monkeypatch):
    monkeypatch.delenv("AGENTIC_WALLET_DEBUG_TRANSCRIPTS", raising=False)
    assert (await client.get("/debug/transcripts")).status_code == 404


@pytest.mark.anyio
async def test_local_debug_transcript_records_displays_and_clears(client, monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_DEBUG_TRANSCRIPTS", "1")
    web_app.debug_transcripts.clear()

    chat_response = await client.post(
        "/chat", json={"session_id": "debug-s1", "message": "show my portfolio"}
    )
    assert chat_response.status_code == 200

    page = await client.get("/debug/transcripts")
    assert page.status_code == 200
    assert "Local development only" in page.text
    assert page.headers["cache-control"].startswith("no-store")

    transcript = await client.get("/debug/transcripts.json")
    body = transcript.json()
    assert body["storage"] == "process-memory-only"
    assert body["sessions"][0]["session_id"] == "debug-s1"
    turn = body["sessions"][0]["turns"][0]
    assert turn["user_message"] == "show my portfolio"
    assert turn["assistant_reply"] == chat_response.json()["reply"]
    assert turn["data"]["type"] == "portfolio"

    cleared = await client.delete("/debug/transcripts")
    assert cleared.json() == {"cleared": True}
    assert (await client.get("/debug/transcripts.json")).json()["sessions"] == []


@pytest.mark.anyio
async def test_debug_transcript_rejects_non_loopback_clients(monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_DEBUG_TRANSCRIPTS", "1")
    web_app.debug_transcripts.clear()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 43120))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as remote:
        await remote.post(
            "/chat", json={"session_id": "remote", "message": "show my portfolio"}
        )
        assert (await remote.get("/debug/transcripts.json")).status_code == 404
    assert web_app.debug_transcripts.snapshot()["sessions"] == []


@pytest.mark.anyio
async def test_debug_transcript_rejects_non_loopback_host_header(client, monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_DEBUG_TRANSCRIPTS", "1")
    response = await client.get(
        "/debug/transcripts.json", headers={"Host": "attacker.example:8000"}
    )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_capabilities_disclose_inference_location(client):
    body = (await client.get("/capabilities")).json()
    assert body["inference_location"] == "server-deterministic"
    assert body["signing"] is False


@pytest.mark.anyio
async def test_portfolio_is_read_only_snapshot(client):
    body = (await client.get("/portfolio")).json()
    assert body["chain_id"] == 8453
    assert body["native_balance"]["base_units"] == "241000000000000000"


@pytest.mark.anyio
async def test_registry_exposes_version_digest(client):
    body = (await client.get("/registry")).json()
    assert body["version_digest"].startswith("sha256:")
    assert any(e["asset_id"] == "base:usdc" for e in body["entries"])


@pytest.mark.anyio
async def test_unknown_balance_is_404(client):
    assert (await client.get("/balance/base:unknown")).status_code == 404


@pytest.mark.anyio
async def test_tool_call_validation_fails_closed(client):
    ok = await client.post("/validate/tool-call", json={"action": "get_swap_quote"})
    assert ok.status_code == 200 and ok.json()["valid"] is True
    bad = await client.post("/validate/tool-call", json={"reason": "no action field"})
    assert bad.status_code == 422


@pytest.mark.anyio
async def test_chat_reads_portfolio(client):
    r = await client.post(
        "/chat", json={"session_id": "s1", "message": "show my portfolio"}
    )
    body = r.json()
    assert body["data"]["type"] == "portfolio"
    assert body["state"] == "IDLE"
    assert "suggested_actions" in body


@pytest.mark.anyio
async def test_chat_refuses_state_changing_actions(client):
    r = await client.post(
        "/chat", json={"session_id": "s1", "message": "swap 100 USDC for ETH"}
    )
    reply = r.json()["reply"].lower()
    assert "not enabled" in reply and r.json()["data"] is None


@pytest.mark.anyio
async def test_remote_mode_requires_per_request_consent(client, monkeypatch):
    monkeypatch.setattr(web_app, "_inference_location", "remote-model")
    r = await client.post(
        "/chat", json={"session_id": "s1", "message": "show my portfolio"}
    )
    assert "not sent" in r.json()["reply"]


def test_llama_cpp_http_provider_can_be_selected(monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_INFERENCE_PROVIDER", "llama-cpp-http")
    monkeypatch.setenv("AGENTIC_WALLET_INFERENCE_BASE_URL", "https://model.example")
    agent = web_app._build_chat_agent()
    assert agent.provider.name == "llama-cpp-http"
    assert agent.provider.base_url == "https://model.example"


def test_inference_location_comes_from_actual_endpoint():
    assert (
        web_app._classify_inference_location("ollama", "http://127.0.0.1:11434")
        == "server-model"
    )
    assert (
        web_app._classify_inference_location("ollama", "https://model.example")
        == "remote-model"
    )
    assert (
        web_app._classify_inference_location(
            "llama-cpp-http", "http://localhost:18080"
        )
        == "server-model"
    )


def test_local_ollama_provider_can_be_selected(monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("AGENTIC_WALLET_MODEL_ID", "gemma4:e2b")
    monkeypatch.setenv("AGENTIC_WALLET_INFERENCE_BASE_URL", "http://localhost:11434")
    agent = web_app._build_chat_agent()
    assert agent.provider.name == "ollama"
    assert agent.provider.model == "gemma4:e2b"
    assert agent.provider.base_url == "http://localhost:11434"


def test_unconstrained_transformers_is_blocked_from_normal_chat(monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_INFERENCE_PROVIDER", "local-transformers")
    monkeypatch.delenv("AGENTIC_WALLET_ALLOW_UNCONSTRAINED_INFERENCE", raising=False)
    with pytest.raises(RuntimeError, match="development-only"):
        web_app._build_chat_agent()

    monkeypatch.setenv("AGENTIC_WALLET_ALLOW_UNCONSTRAINED_INFERENCE", "true")
    agent = web_app._build_chat_agent()
    assert agent.provider.name == "local-transformers"


def test_openrouter_provider_can_be_selected_without_exposing_key(monkeypatch):
    monkeypatch.setenv("AGENTIC_WALLET_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("AGENTIC_WALLET_MODEL_ID", "google/gemma-4-E2B-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-key")
    agent = web_app._build_chat_agent()
    assert agent.provider.name == "openrouter"
    assert agent.provider.model == "google/gemma-4-E2B-test"
    assert "private-test-key" not in repr(agent.provider.last_response_metadata)


def _model_agent(raw_call):
    harness = MockReadOnlyHarness.from_fixture(FIXTURE)
    return DemoChatAgent(harness, provider=ScriptedProvider({None: raw_call}))


def test_model_backed_chat_executes_validated_read_tool():
    agent = _model_agent(
        {"action": "get_balance", "arguments": {"asset_id": "base:usdc"}}
    )
    body = agent.respond("model-session", "what is my usdc balance?")
    assert body["data"]["type"] == "balance"
    assert body["data"]["asset_id"] == "base:usdc"


def test_model_backed_chat_fails_closed_on_tool_specific_extra_arguments():
    agent = _model_agent(
        {"action": "get_portfolio", "arguments": {"recipient": "0xdeadbeef"}}
    )
    body = agent.respond("model-session", "show my portfolio")
    assert body["data"] is None
    assert "no wallet tool was run" in body["reply"].lower()


def test_model_backed_chat_fails_closed_on_unavailable_signing_proposal():
    agent = _model_agent({"action": "sign_transaction", "arguments": {}})
    body = agent.respond("model-session", "sign it")
    assert body["data"] is None
    assert "no wallet tool was run" in body["reply"].lower()


def test_production_chat_rejects_legacy_literal_recipient_action():
    agent = _model_agent(
        {
            "action": "create_transfer_plan",
            "arguments": {
                "chain_id": 8453,
                "asset_id": "base:usdc",
                "amount_base_units": "2500000",
                "recipient": "0x3333333333333333333333333333333333333333",
            },
        }
    )
    body = agent.respond("model-session", "draft a transfer")
    assert body["data"] is None
    assert "no wallet tool was run" in body["reply"].lower()


def test_read_only_prefilter_uses_words_not_substrings():
    agent = _model_agent({"action": "show_help", "arguments": {}})
    significant = agent.respond("model-session", "is this balance significant?")
    history = agent.respond("model-session", "show my transaction history")
    assert "not enabled" not in significant["reply"].lower()
    assert "not enabled" not in history["reply"].lower()


def test_suggestion_chips_reenter_the_normal_chat_pipeline():
    page = (Path(__file__).resolve().parents[1] / "src/agentic_wallet/web/static/index.html").read_text()
    assert 'button.addEventListener("click", () => send(item.prompt))' in page
    assert "item.action" not in page.split("function renderSuggestions", 1)[1].split("async function send", 1)[0]
