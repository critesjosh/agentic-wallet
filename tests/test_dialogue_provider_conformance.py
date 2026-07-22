from __future__ import annotations

import json

import pytest

from agentic_wallet.inference import InferenceError
from agentic_wallet.providers import (
    LlamaCppHTTPProvider,
    LocalTransformersProvider,
    OllamaProvider,
    OpenRouterProvider,
)
from agentic_wallet.tool_contract import (
    dialogue_route_json_schema,
    dialogue_turn_json_schema,
)

CONTEXT = {"user_request": "hello", "canonical_asset_ids": ["base:usdc"]}
ACTIONS = ["get_balance", "show_help"]
SUGGESTIONS = ["get_portfolio", "get_balance"]
CONVERSATION = {
    "message": "Hi. I can check a balance.",
    "intent": "offer_action",
    "proposed_action": "none",
    "arguments": {},
    "reason": "",
    "suggested_actions": ["get_balance"],
}
ROUTE = {
    "message": "I will check the typed balance.",
    "intent": "propose_tool",
    "proposed_action": "get_balance",
    "reason": "explicit request",
    "suggested_actions": [],
}


class GeneratedDialogueProvider(LocalTransformersProvider):
    def __init__(self, raw: dict):
        super().__init__()
        self.output = json.dumps(raw)

    def _build_dialogue_prompt(self, context, available_actions, suggested_action_ids):
        return "dialogue prompt"

    def _render_prompt(self, request_text):
        return request_text

    def _generate(self, prompt):
        return self.output


def _providers(raw: dict):
    content = json.dumps(raw)
    return [
        OllamaProvider(
            model="gemma4:e2b",
            transport=lambda *_: {"message": {"content": content}},
        ),
        LlamaCppHTTPProvider(transport=lambda *_: {"content": content}),
        OpenRouterProvider(
            api_key="test",
            model="google/gemma-4-E2B-test",
            transport=lambda *_: {
                "choices": [{"message": {"content": content}}]
            },
        ),
        GeneratedDialogueProvider(raw),
    ]


@pytest.mark.parametrize("provider", _providers(CONVERSATION), ids=lambda item: item.name)
def test_all_providers_accept_display_only_conversation(provider):
    turn = provider.propose_dialogue_turn(CONTEXT, ACTIONS, SUGGESTIONS)
    assert turn.message == CONVERSATION["message"]
    assert turn.proposed_action is None
    assert turn.suggested_actions == ["get_balance"]


@pytest.mark.parametrize("provider", _providers(ROUTE), ids=lambda item: item.name)
def test_all_providers_accept_argument_free_route(provider):
    route = provider.propose_dialogue_route(CONTEXT, ACTIONS, SUGGESTIONS)
    assert route.proposed_action == "get_balance"
    assert "arguments" not in route.model_dump()


def test_native_providers_constrain_the_route_schema():
    expected = dialogue_route_json_schema(ACTIONS, SUGGESTIONS)
    captured = {}
    content = json.dumps(ROUTE)

    def ollama_transport(_url, payload, _timeout):
        captured["ollama"] = payload["format"]
        return {"message": {"content": content}}

    def llama_transport(_url, payload, _timeout):
        captured["llama"] = payload["json_schema"]
        return {"content": content}

    def openrouter_transport(_url, payload, _headers, _timeout):
        captured["openrouter"] = payload["response_format"]["json_schema"]["schema"]
        return {"choices": [{"message": {"content": content}}]}

    OllamaProvider(model="gemma4:e2b", transport=ollama_transport).propose_dialogue_route(
        CONTEXT, ACTIONS, SUGGESTIONS
    )
    LlamaCppHTTPProvider(transport=llama_transport).propose_dialogue_route(
        CONTEXT, ACTIONS, SUGGESTIONS
    )
    OpenRouterProvider(
        api_key="test",
        model="google/gemma-4-E2B-test",
        transport=openrouter_transport,
    ).propose_dialogue_route(CONTEXT, ACTIONS, SUGGESTIONS)
    assert captured == {"ollama": expected, "llama": expected, "openrouter": expected}


@pytest.mark.parametrize(
    "provider",
    _providers({**CONVERSATION, "suggested_actions": ["drain_wallet"]}),
    ids=lambda item: item.name,
)
def test_all_providers_reject_invented_suggestions(provider):
    with pytest.raises(InferenceError, match="unknown suggested action"):
        provider.propose_dialogue_turn(CONTEXT, ACTIONS, SUGGESTIONS)


@pytest.mark.parametrize(
    "provider",
    _providers(
        {
            "message": "Signing now",
            "intent": "propose_tool",
            "proposed_action": "proceed_to_signing",
            "arguments": {},
            "reason": "bad",
            "suggested_actions": [],
        }
    ),
    ids=lambda item: item.name,
)
def test_all_providers_reject_unavailable_consequential_action(provider):
    with pytest.raises(InferenceError, match="not available"):
        provider.propose_dialogue_turn(CONTEXT, ACTIONS, SUGGESTIONS)


def test_native_providers_receive_the_same_dialogue_schema():
    expected = dialogue_turn_json_schema(
        ACTIONS, SUGGESTIONS, CONTEXT["canonical_asset_ids"]
    )
    captured = {}
    content = json.dumps(CONVERSATION)

    def ollama_transport(_url, payload, _timeout):
        captured["ollama"] = payload["format"]
        return {"message": {"content": content}}

    def llama_transport(_url, payload, _timeout):
        captured["llama"] = payload["json_schema"]
        return {"content": content}

    def openrouter_transport(_url, payload, _headers, _timeout):
        captured["openrouter"] = payload["response_format"]["json_schema"]["schema"]
        return {"choices": [{"message": {"content": content}}]}

    OllamaProvider(model="gemma4:e2b", transport=ollama_transport).propose_dialogue_turn(
        CONTEXT, ACTIONS, SUGGESTIONS
    )
    LlamaCppHTTPProvider(transport=llama_transport).propose_dialogue_turn(
        CONTEXT, ACTIONS, SUGGESTIONS
    )
    OpenRouterProvider(
        api_key="test",
        model="google/gemma-4-E2B-test",
        transport=openrouter_transport,
    ).propose_dialogue_turn(CONTEXT, ACTIONS, SUGGESTIONS)
    assert captured == {"ollama": expected, "llama": expected, "openrouter": expected}


@pytest.mark.parametrize(
    "provider",
    _providers(
        {
            "message": "I'll check the watch-only balance.",
            "intent": "propose_tool",
            "proposed_action": "get_balance",
            "arguments": {"asset_id": "base:usdc"},
            "reason": "The user requested a USDC balance.",
            "suggested_actions": [],
        }
    ),
    ids=lambda item: item.name,
)
def test_all_providers_normalize_flat_typed_proposal(provider):
    turn = provider.propose_dialogue_turn(CONTEXT, ACTIONS, SUGGESTIONS)
    assert turn.proposed_action is not None
    assert turn.proposed_action.action == "get_balance"
    assert turn.proposed_action.arguments == {"asset_id": "base:usdc"}


@pytest.mark.parametrize(
    "provider",
    _providers(
        {
            "message": "I can offer help.",
            "intent": "offer_action",
            "proposed_action": "get_balance",
            "arguments": {"asset_id": "base:usdc"},
            "reason": "not executable",
            "suggested_actions": [],
        }
    ),
    ids=lambda item: item.name,
)
def test_non_tool_intent_cannot_carry_actionable_arguments(provider):
    with pytest.raises(InferenceError, match="inconsistent action fields"):
        provider.propose_dialogue_turn(CONTEXT, ACTIONS, SUGGESTIONS)


def test_flat_action_missing_required_arguments_fails_closed():
    raw = {
        "message": "I'll check it.",
        "intent": "propose_tool",
        "proposed_action": "get_balance",
        "arguments": {},
        "reason": "missing asset",
        "suggested_actions": [],
    }
    with pytest.raises(InferenceError, match="invalid arguments for get_balance"):
        GeneratedDialogueProvider(raw).propose_dialogue_turn(
            CONTEXT, ACTIONS, SUGGESTIONS
        )


def test_excess_valid_suggestions_are_safely_capped():
    raw = {
        "message": "Here are some options.",
        "intent": "offer_action",
        "proposed_action": "none",
        "arguments": {},
        "reason": "",
        "suggested_actions": [
            "get_portfolio",
            "get_balance",
            "get_allowances",
            "get_registry",
        ],
    }
    provider = GeneratedDialogueProvider(raw)
    turn = provider.propose_dialogue_turn(
        CONTEXT,
        ACTIONS,
        ["get_portfolio", "get_balance", "get_allowances", "get_registry"],
    )
    assert turn.suggested_actions == [
        "get_portfolio",
        "get_balance",
        "get_allowances",
    ]


def test_unknown_suggestion_after_cap_boundary_still_fails_closed():
    raw = {
        "message": "Here are some options.",
        "intent": "offer_action",
        "proposed_action": "none",
        "arguments": {},
        "reason": "",
        "suggested_actions": [
            "get_portfolio",
            "get_balance",
            "get_allowances",
            "drain_wallet",
        ],
    }
    with pytest.raises(InferenceError, match="unknown suggested action"):
        GeneratedDialogueProvider(raw).propose_dialogue_turn(
            CONTEXT,
            ACTIONS,
            ["get_portfolio", "get_balance", "get_allowances", "get_registry"],
        )
