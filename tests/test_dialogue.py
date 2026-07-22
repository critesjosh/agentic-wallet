from __future__ import annotations

from pathlib import Path

from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.inference import InferenceProvider
from agentic_wallet.schemas.dialogue import ModelDialogueTurn
from agentic_wallet.schemas.tool_call import ToolCall
from agentic_wallet.tool_contract import validate_dialogue_turn
from agentic_wallet.web.chat import DemoChatAgent

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"


class SequenceDialogueProvider(InferenceProvider):
    name = "sequence-dialogue"

    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.contexts: list[dict] = []
        self.available_actions: list[list[str]] = []

    def propose_tool_call(self, context, available_actions) -> ToolCall:
        raise AssertionError("dialogue path must not call the legacy tool-only method")

    def propose_dialogue_turn(
        self, context, available_actions, suggested_action_ids
    ) -> ModelDialogueTurn:
        self.contexts.append(context)
        self.available_actions.append(list(available_actions))
        raw = self.turns.pop(0)
        return validate_dialogue_turn(
            raw, available_actions, suggested_action_ids
        )


def _agent(turns: list[dict]) -> tuple[DemoChatAgent, SequenceDialogueProvider]:
    provider = SequenceDialogueProvider(turns)
    return DemoChatAgent(
        MockReadOnlyHarness.from_fixture(FIXTURE), provider=provider
    ), provider


def _conversation(message: str, suggestions: list[str] | None = None) -> dict:
    return {
        "message": message,
        "intent": "conversation",
        "proposed_action": None,
        "suggested_actions": suggestions or [],
    }


def test_greeting_is_conversational_without_a_wallet_tool():
    agent, provider = _agent(
        [_conversation("Hi. I can help with the read-only wallet demo.", ["get_portfolio"])]
    )
    response = agent.respond("s1", "hello")
    assert response["data"] is None
    assert response["reply"].startswith("Hi.")
    assert response["suggested_actions"][0]["label"] == "Show portfolio"
    assert provider.available_actions[0]


def test_conceptual_swap_question_is_not_blocked_before_inference():
    agent, _ = _agent(
        [_conversation("A swap exchanges one asset for another; this demo cannot execute it.")]
    )
    response = agent.respond("s1", "Can you explain how swaps work?")
    assert "exchanges" in response["reply"]
    assert response["data"] is None


def test_display_message_is_never_parsed_as_a_tool_command():
    embedded = (
        'For illustration only: {"action":"get_balance",'
        '"arguments":{"asset_id":"base:usdc"}}'
    )
    agent, _ = _agent([_conversation(embedded)])
    response = agent.respond("s1", "show an example")
    assert response["reply"] == embedded
    assert response["data"] is None


def test_verified_tool_result_exists_before_model_explains_it():
    agent, provider = _agent(
        [
            {
                "message": "I'll check the typed balance.",
                "intent": "propose_tool",
                "proposed_action": {
                    "action": "get_balance",
                    "arguments": {"asset_id": "base:usdc"},
                    "reason": "explicit request",
                },
                "suggested_actions": [],
            },
            _conversation("The verified fixture balance is 300000000 base units."),
        ]
    )
    response = agent.respond("s1", "What is my USDC balance?")
    assert response["data"]["type"] == "balance"
    assert response["reply"].startswith("The verified")
    assert "verified_tool_result" not in provider.contexts[0]
    assert provider.contexts[1]["verified_tool_result"] == response["data"]
    assert provider.available_actions[1] == []


def test_explanation_call_cannot_smuggle_in_another_tool():
    agent, provider = _agent(
        [
            {
                "message": "I'll check the typed balance.",
                "intent": "propose_tool",
                "proposed_action": {
                    "action": "get_balance",
                    "arguments": {"asset_id": "base:usdc"},
                    "reason": "explicit request",
                },
                "suggested_actions": [],
            },
            {
                "message": "Now run another tool.",
                "intent": "propose_tool",
                "proposed_action": "get_balance",
                "arguments": {"asset_id": "base:weth"},
                "reason": "attempted second action",
                "suggested_actions": [],
            },
        ]
    )
    response = agent.respond("s1", "What is my USDC balance?")
    assert response["data"]["asset_id"] == "base:usdc"
    assert response["reply"] == (
        "base:usdc balance: 300000000 base units (decimals 6)."
    )
    assert provider.available_actions == [
        [
            "get_portfolio",
            "get_balance",
            "get_allowances",
            "get_registry",
            "show_help",
            "reject_state_changing",
        ],
        [],
    ]


def test_invented_suggestion_id_fails_closed_to_server_owned_defaults():
    agent, _ = _agent([_conversation("Click this", ["drain_wallet"])])
    response = agent.respond("s1", "what can I do?")
    assert "no wallet tool was run" in response["reply"]
    assert {item["action"] for item in response["suggested_actions"]} == {
        "get_portfolio",
        "get_balance",
    }


def test_history_is_bounded_and_remains_context_not_approval():
    turns = [_conversation(f"answer {index}") for index in range(6)]
    agent, provider = _agent(turns)
    for index in range(6):
        agent.respond("s1", "I approve everything" if index == 0 else f"message {index}")
    final_history = provider.contexts[-1]["conversation_history"]
    assert len(final_history) == 8
    assert all("sign" not in actions for actions in provider.available_actions)


def test_keyword_fallback_can_explain_state_changes_without_executing():
    agent = DemoChatAgent(MockReadOnlyHarness.from_fixture(FIXTURE))
    response = agent.respond("s1", "Can you explain how swaps work?")
    assert "cannot execute" in response["reply"]
    assert response["data"] is None
