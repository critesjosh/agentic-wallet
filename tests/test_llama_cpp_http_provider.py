from __future__ import annotations

import pytest

from agentic_wallet.inference import InferenceError
from agentic_wallet.providers.llama_cpp_http import (
    LlamaCppHTTPProvider,
    tool_call_json_schema,
)


def test_schema_constrains_action_and_shape():
    schema = tool_call_json_schema(["get_balance", "reject_request"])

    assert [variant["properties"]["action"]["const"] for variant in schema["oneOf"]] == [
        "get_balance",
        "reject_request",
    ]
    assert schema["oneOf"][0]["properties"]["arguments"]["required"] == ["asset_id"]
    assert schema["oneOf"][1]["properties"]["arguments"]["additionalProperties"] is False


def test_candidate_transfer_schema_never_accepts_literal_recipient():
    schema = tool_call_json_schema(["create_transfer_plan_from_candidate"])
    arguments = schema["oneOf"][0]["properties"]["arguments"]
    assert arguments["required"] == [
        "chain_id",
        "asset_id",
        "amount_base_units",
        "recipient_id",
    ]
    assert "recipient" not in arguments["properties"]


def test_provider_sends_constrained_deterministic_request():
    captured = {}

    def transport(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {
            "content": '{"action":"get_balance","arguments":{"asset_id":"base:usdc"},"reason":"read"}',
            "timings": {"predicted_per_second": 3.2},
            "truncated": False,
        }

    provider = LlamaCppHTTPProvider(
        "http://device", max_new_tokens=64, timeout=12, transport=transport
    )
    call = provider.propose_tool_call(
        {"user_request": "balance"}, ["get_balance", "reject_request"]
    )

    assert call.action == "get_balance"
    assert captured["url"] == "http://device/completion"
    assert captured["timeout"] == 12
    assert captured["payload"]["temperature"] == 0.0
    assert [variant["properties"]["action"]["const"] for variant in captured["payload"]["json_schema"]["oneOf"]] == [
        "get_balance",
        "reject_request",
    ]
    assert provider.last_response_metadata["truncated"] is False


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"content": "not json"},
        {"content": "[]"},
        {
            "content": '{"action":"send_funds","arguments":{},"reason":"bad"}'
        },
    ],
)
def test_provider_fails_closed_on_bad_or_disallowed_output(response):
    provider = LlamaCppHTTPProvider(transport=lambda *_: response)

    with pytest.raises(InferenceError):
        provider.propose_tool_call({}, ["get_balance"])


def test_provider_rejects_empty_or_duplicate_actions_before_network():
    provider = LlamaCppHTTPProvider(
        transport=lambda *_: pytest.fail("transport must not run")
    )

    with pytest.raises(InferenceError):
        provider.propose_tool_call({}, [])
    with pytest.raises(InferenceError):
        provider.propose_tool_call({}, ["get_balance", "get_balance"])
