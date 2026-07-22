from __future__ import annotations

import pytest

from agentic_wallet.inference import InferenceError
from agentic_wallet.providers.ollama import OllamaProvider, ollama_tool_call_schema


def test_schema_limits_actions_and_argument_fields():
    schema = ollama_tool_call_schema(
        ["get_balance", "show_help"], ["base:native", "base:usdc"]
    )
    assert [item["properties"]["action"]["const"] for item in schema["oneOf"]] == [
        "get_balance",
        "show_help",
    ]
    assert schema["oneOf"][0]["properties"]["arguments"]["required"] == [
        "asset_id"
    ]
    assert schema["oneOf"][0]["properties"]["arguments"]["properties"][
        "asset_id"
    ]["enum"] == ["base:native", "base:usdc"]
    assert schema["oneOf"][1]["properties"]["arguments"]["properties"] == {}


def test_provider_uses_local_chat_structured_output():
    captured = {}

    def transport(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {
            "model": "gemma4:e2b",
            "message": {
                "content": (
                    '{"action":"get_balance","arguments":'
                    '{"asset_id":"base:usdc"},"reason":"read"}'
                )
            },
            "done": True,
            "done_reason": "stop",
            "eval_count": 12,
        }

    provider = OllamaProvider(model="gemma4:e2b", transport=transport)
    call = provider.propose_tool_call({}, ["get_balance", "show_help"])

    assert call.action == "get_balance"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {"temperature": 0, "seed": 0}
    assert captured["payload"]["format"]["oneOf"]
    assert provider.last_response_metadata["eval_count"] == 12


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"message": {"content": 12}, "done": True, "done_reason": "stop"},
        {
            "message": {"content": "not-json"},
            "done": True,
            "done_reason": "stop",
        },
        {
            "message": {
                "content": (
                    '{"action":"sign_transaction","arguments":{},'
                    '"reason":"bad"}'
                )
            },
            "done": True,
            "done_reason": "stop",
        },
    ],
)
def test_provider_fails_closed_on_bad_or_disallowed_response(response):
    provider = OllamaProvider(model="test", transport=lambda *_: response)
    with pytest.raises(InferenceError):
        provider.propose_tool_call({}, ["show_help"])


def test_model_is_required():
    with pytest.raises(InferenceError, match="required"):
        OllamaProvider(model=" ")


@pytest.mark.parametrize(
    "response",
    [
        {
            "message": {"content": "", "thinking": "partial"},
            "done": False,
        },
        {"message": {"content": ""}, "done": True, "done_reason": "stop"},
        {
            "message": {"content": '{"action":"show_help"}'},
            "done": True,
            "done_reason": "length",
        },
    ],
)
def test_provider_rejects_incomplete_ollama_envelopes(response):
    provider = OllamaProvider(model="test", transport=lambda *_: response)
    with pytest.raises(InferenceError, match="incomplete|normal stop|empty"):
        provider.propose_tool_call({}, ["show_help"])
    assert "thinking" not in str(provider.last_raw_output)
