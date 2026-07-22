from __future__ import annotations

import pytest

from agentic_wallet.inference import InferenceError, InferenceProvider
from agentic_wallet.schemas.conversation import ConversationLedger
from agentic_wallet.schemas.dialogue import DialogueRoute
from agentic_wallet.schemas.tool_call import ToolCall
from agentic_wallet.tool_contract import validate_dialogue_route
from agentic_wallet.web.narration import (
    render_verified_result,
    validate_grounded_message,
)


class RepairSequenceProvider(InferenceProvider):
    name = "repair-sequence"

    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)
        self.contexts: list[dict] = []

    def propose_tool_call(self, context, available_actions) -> ToolCall:
        self.contexts.append(context)
        raw = self.outputs.pop(0)
        self.last_raw_output = raw
        return self._validate(raw, available_actions)


class TransportFailureProvider(InferenceProvider):
    name = "transport-failure"

    def __init__(self):
        self.calls = 0

    def propose_tool_call(self, context, available_actions) -> ToolCall:
        self.calls += 1
        raise InferenceError("network unavailable")


class RouteRepairSequenceProvider(RepairSequenceProvider):
    def propose_dialogue_route(
        self, context, available_actions, suggested_action_ids
    ) -> DialogueRoute:
        self.contexts.append(context)
        raw = self.outputs.pop(0)
        self.last_raw_output = raw
        return validate_dialogue_route(
            raw, available_actions, suggested_action_ids
        )


def test_one_bounded_repair_corrects_invalid_arguments():
    provider = RepairSequenceProvider(
        [
            {"action": "get_balance", "arguments": {}, "reason": ""},
            {
                "action": "get_balance",
                "arguments": {"asset_id": "base:usdc"},
                "reason": "",
            },
        ]
    )
    call = provider.propose_tool_call_with_repair(
        {"user_request": "check usdc"}, "get_balance"
    )
    assert call.arguments == {"asset_id": "base:usdc"}
    assert provider.last_attempt_count == 2
    assert provider.contexts[1]["phase"] == "repair_tool_arguments"
    assert provider.contexts[1]["previous_output"]["arguments"] == {}
    assert "asset_id" in provider.contexts[1]["validation_error"]


def test_repair_stops_after_exactly_one_retry():
    provider = RepairSequenceProvider(
        [
            {"action": "get_balance", "arguments": {}, "reason": ""},
            {"action": "get_balance", "arguments": {}, "reason": ""},
        ]
    )
    with pytest.raises(InferenceError, match="one bounded repair"):
        provider.propose_tool_call_with_repair({}, "get_balance")
    assert len(provider.contexts) == 2


def test_hard_zero_actions_are_never_repaired():
    provider = RepairSequenceProvider(
        [{"action": "proceed_to_signing", "arguments": {"extra": 1}, "reason": ""}]
    )
    with pytest.raises(InferenceError):
        provider.propose_tool_call_with_repair({}, "proceed_to_signing")
    assert len(provider.contexts) == 1


def test_transport_failures_are_not_misclassified_as_repairable_output():
    provider = TransportFailureProvider()
    with pytest.raises(InferenceError, match="network unavailable"):
        provider.propose_tool_call_with_repair({}, "get_balance")
    assert provider.calls == 1


def test_one_bounded_route_repair_corrects_invalid_shape():
    provider = RouteRepairSequenceProvider(
        [
            {
                "message": "I can check that.",
                "intent": "propose_tool",
                "proposed_action": "get_balance",
                "reason": "",
                "suggested_actions": [],
                "arguments": {"asset_id": "base:usdc"},
            },
            {
                "message": "I can check that.",
                "intent": "propose_tool",
                "proposed_action": "get_balance",
                "reason": "",
                "suggested_actions": [],
            },
        ]
    )
    route = provider.propose_dialogue_route_with_repair(
        {"user_request": "check usdc"}, ["get_balance"], []
    )
    assert route.proposed_action == "get_balance"
    assert provider.last_attempt_count == 2
    assert provider.contexts[1]["phase"] == "repair_dialogue_route"


def test_route_repair_does_not_retry_disallowed_signing_selection():
    provider = RouteRepairSequenceProvider(
        [
            {
                "message": "Signing now.",
                "intent": "propose_tool",
                "proposed_action": "proceed_to_signing",
                "reason": "",
                "suggested_actions": [],
            }
        ]
    )
    with pytest.raises(InferenceError, match="not available"):
        provider.propose_dialogue_route_with_repair(
            {}, ["get_balance"], []
        )
    assert len(provider.contexts) == 1


def test_conversation_ledger_is_bounded_and_has_no_approval_field():
    ledger = ConversationLedger(workflow_state="IDLE", chain_id=8453)
    for index in range(6):
        ledger.record_message("user", f"message {index}")
        ledger.record_message("assistant", f"reply {index}")
    ledger.record_validated_arguments({"asset_id": "base:usdc"})
    ledger.record_validated_arguments({"asset_id": "base:weth"})

    dumped = ledger.model_dump()
    assert len(dumped["recent_messages"]) == 8
    assert dumped["corrections"] == [
        {"field": "asset_id", "previous": "base:usdc", "current": "base:weth"}
    ]
    assert "approval" not in dumped


def test_grounded_narration_accepts_verified_facts_and_rejects_inventions():
    result = {
        "type": "balance",
        "asset_id": "base:usdc",
        "amount": {"base_units": "300000000", "decimals": 6},
    }
    summary = render_verified_result(result)
    assert validate_grounded_message(
        "The verified base:usdc balance is 300000000 base units.",
        result,
        summary,
    ).startswith("The verified")
    with pytest.raises(InferenceError, match="unsupported typed facts"):
        validate_grounded_message(
            "The verified base:weth balance is 999 base units.", result, summary
        )
    with pytest.raises(InferenceError, match="unsupported typed facts"):
        validate_grounded_message(
            "The verified base:usdc balance is 3 base units.", result, summary
        )
    with pytest.raises(InferenceError, match="claims wallet execution"):
        validate_grounded_message("I sent the funds.", result, summary)
