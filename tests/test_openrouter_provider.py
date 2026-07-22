from __future__ import annotations

import pytest

from agentic_wallet.inference import InferenceError
from agentic_wallet.providers.openrouter import (
    OpenRouterProvider,
    openrouter_tool_call_schema,
)


def test_schema_limits_actions_and_argument_fields():
    schema = openrouter_tool_call_schema(["get_balance", "show_help"])
    assert [variant["properties"]["action"]["const"] for variant in schema["oneOf"]] == [
        "get_balance",
        "show_help",
    ]
    assert schema["oneOf"][0]["properties"]["arguments"]["required"] == ["asset_id"]
    assert schema["oneOf"][1]["properties"]["arguments"]["additionalProperties"] is False


def test_provider_requests_strict_structured_output_and_privacy_routing():
    captured = {}

    def transport(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {
            "id": "generation-1",
            "model": "google/test",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"action":"get_balance","arguments":'
                            '{"asset_id":"base:usdc"},"reason":"read"}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

    provider = OpenRouterProvider(
        api_key="secret-test-key",
        model="google/gemma-4-E2B-test",
        zero_data_retention=True,
        transport=transport,
    )
    call = provider.propose_tool_call({}, ["get_balance", "show_help"])

    assert call.action == "get_balance"
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
    assert captured["payload"]["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert captured["headers"]["Authorization"] == "Bearer secret-test-key"
    assert "secret-test-key" not in repr(provider.last_response_metadata)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "not-json"}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": '{"action":"sign_transaction","arguments":{},"reason":"bad"}'
                    }
                }
            ]
        },
    ],
)
def test_provider_fails_closed_on_bad_or_disallowed_response(response):
    provider = OpenRouterProvider(
        api_key="test", model="google/gemma-4-E2B-test", transport=lambda *_: response
    )
    with pytest.raises(InferenceError):
        provider.propose_tool_call({}, ["show_help"])


def test_api_key_is_required():
    with pytest.raises(InferenceError, match="required"):
        OpenRouterProvider(api_key=" ", model="google/gemma-4-E2B-test")


def test_non_target_model_emits_explicit_warning():
    with pytest.warns(RuntimeWarning, match="target Gemma"):
        OpenRouterProvider(api_key="test", model="anthropic/frontier")
