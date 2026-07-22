from __future__ import annotations

import pytest

from agentic_wallet.inference import InferenceError
from agentic_wallet.providers import (
    LlamaCppHTTPProvider,
    LocalTransformersProvider,
    OllamaProvider,
    OpenRouterProvider,
)
from agentic_wallet.tool_contract import (
    CONTRACT_VERSION,
    tool_call_json_schema,
)

CONTEXT = {
    "user_request": "what is my USDC balance?",
    "canonical_asset_ids": ["base:native", "base:usdc"],
}
ACTIONS = ["get_balance", "show_help"]
VALID = {
    "action": "get_balance",
    "arguments": {"asset_id": "base:usdc"},
    "reason": "read",
}
INVALID_ARGUMENTS = {
    "action": "get_balance",
    "arguments": {},
    "reason": "missing asset",
}


class GeneratedProvider(LocalTransformersProvider):
    def __init__(self, output: str):
        super().__init__()
        self.output = output

    def _build_prompt(self, context, available_actions):
        from agentic_wallet.tool_contract import tool_call_prompt

        return tool_call_prompt(context, available_actions)

    def _generate(self, prompt):
        return self.output


def _providers(raw: dict):
    import json

    content = json.dumps(raw)
    return [
        OllamaProvider(
            model="gemma4:e2b",
            transport=lambda *_: {
                "message": {"content": content},
                "done": True,
                "done_reason": "stop",
            },
        ),
        LlamaCppHTTPProvider(transport=lambda *_: {"content": content}),
        OpenRouterProvider(
            api_key="test",
            model="google/gemma-4-E2B-test",
            transport=lambda *_: {
                "choices": [{"message": {"content": content}}]
            },
        ),
        GeneratedProvider(content),
    ]


@pytest.mark.parametrize("provider", _providers(VALID), ids=lambda item: item.name)
def test_all_providers_accept_the_same_valid_contract(provider):
    call = provider.propose_tool_call(CONTEXT, ACTIONS)
    assert call.action == "get_balance"
    assert call.arguments == {"asset_id": "base:usdc"}


@pytest.mark.parametrize(
    "provider", _providers(INVALID_ARGUMENTS), ids=lambda item: item.name
)
def test_all_providers_reject_the_same_invalid_arguments(provider):
    with pytest.raises(InferenceError, match="invalid arguments"):
        provider.propose_tool_call(CONTEXT, ACTIONS)


def test_http_providers_receive_identical_schema_and_contract_prompt():
    expected_schema = tool_call_json_schema(
        ACTIONS, CONTEXT["canonical_asset_ids"]
    )
    captured: dict[str, dict] = {}

    def ollama_transport(_url, payload, _timeout):
        captured["ollama"] = payload
        return {
            "message": {"content": __import__("json").dumps(VALID)},
            "done": True,
            "done_reason": "stop",
        }

    def llama_transport(_url, payload, _timeout):
        captured["llama"] = payload
        return {"content": __import__("json").dumps(VALID)}

    def openrouter_transport(_url, payload, _headers, _timeout):
        captured["openrouter"] = payload
        return {"choices": [{"message": {"content": __import__("json").dumps(VALID)}}]}

    OllamaProvider(model="gemma4:e2b", transport=ollama_transport).propose_tool_call(
        CONTEXT, ACTIONS
    )
    LlamaCppHTTPProvider(transport=llama_transport).propose_tool_call(CONTEXT, ACTIONS)
    OpenRouterProvider(
        api_key="test",
        model="google/gemma-4-E2B-test",
        transport=openrouter_transport,
    ).propose_tool_call(CONTEXT, ACTIONS)

    assert captured["ollama"]["format"] == expected_schema
    assert captured["ollama"]["think"] is False
    assert captured["llama"]["json_schema"] == expected_schema
    assert captured["openrouter"]["response_format"]["json_schema"]["schema"] == expected_schema
    assert CONTRACT_VERSION in captured["ollama"]["messages"][0]["content"]
    assert CONTRACT_VERSION in captured["llama"]["prompt"]
    assert CONTRACT_VERSION in captured["openrouter"]["messages"][0]["content"]


def test_runtime_declares_native_constrained_decoding_capability():
    providers = _providers(VALID)
    assert {provider.name: provider.native_constrained_decoding for provider in providers} == {
        "ollama": True,
        "llama-cpp-http": True,
        "openrouter": True,
        "local-transformers": False,
    }
