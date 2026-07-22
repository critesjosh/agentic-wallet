"""Build typed training examples from fixed, naturally authored source records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.common import UntrustedData
from .data import CoverageDimensions, TrainingExample

NATURAL_CURRICULUM_VERSION = "wallet-natural-curriculum-v3-1"
_CHAIN_ID = 8453
_SPENDER = "base:fixture-swap-router"
_READ_ACTIONS = [
    "get_portfolio",
    "get_balance",
    "get_allowances",
    "get_registry",
]
_SOURCE_KEYS = {
    "id",
    "split",
    "family",
    "user_request",
    "params",
}


def _context(record: dict[str, Any], state: str, **extra: Any) -> dict[str, Any]:
    return {
        "user_request": record["user_request"],
        "workflow_state": state,
        "chain_id": _CHAIN_ID,
        "canonical_asset_ids": ["base:native", "base:usdc", "base:weth"],
        **extra,
    }


def _tool_example(record: dict[str, Any]) -> TrainingExample:
    family = record["family"]
    params = record["params"]
    exposure = "production"
    trajectory_id = params.get("trajectory_id")
    turn_index = params.get("turn_index")
    ambiguity = "none"
    correction = "none"
    adversarial = "none"

    if family == "canonical_swap":
        state = "PLANNING"
        actions = ["get_swap_quote", "request_missing_information", "reject_request"]
        action = "get_swap_quote"
        arguments = {
            "chain_id": _CHAIN_ID,
            "input_asset_id": params["input_asset_id"],
            "output_asset_id": params["output_asset_id"],
            "amount": params["amount"],
            "max_slippage_bps": params["max_slippage_bps"],
        }
        context = _context(record, state)
        risk = "none"
    elif family == "recipient_correction":
        if turn_index == 0:
            state = "UNDERSTANDING_INTENT"
            actions = ["request_missing_information", "create_transfer_plan", "reject_request"]
            action = "request_missing_information"
            arguments = {"missing_fields": ["recipient"]}
            ambiguity = "missing-recipient"
            history: list[dict[str, Any]] = []
        else:
            state = "PLANNING"
            actions = ["create_transfer_plan", "request_missing_information", "cancel_request"]
            action = "create_transfer_plan"
            arguments = {
                "chain_id": _CHAIN_ID,
                "asset_id": params["asset_id"],
                "amount_base_units": params["amount_base_units"],
                "recipient": params["recipient"],
            }
            correction = "recipient"
            history = [
                {
                    "role": "assistant",
                    "typed_action": "request_missing_information",
                    "arguments": {"missing_fields": ["recipient"]},
                }
            ]
        context = _context(record, state, conversation_history=history)
        risk = "wrong-recipient"
    elif family == "expired_quote":
        state = "QUOTE_EXPIRED"
        actions = ["refresh_swap_quote", "cancel_request", "reject_request"]
        action = "refresh_swap_quote"
        arguments = {"quote_id": params["quote_id"]}
        context = _context(record, state, quote_id=params["quote_id"])
        risk = "stale-quote"
    elif family == "exact_approval":
        state = "PLANNING"
        actions = ["create_exact_approval_plan", "reject_request"]
        if params["adversarial"]:
            actions.insert(1, "create_unlimited_approval_plan")
            exposure = "adversarial"
            adversarial = "unlimited-approval-distractor"
        action = "create_exact_approval_plan"
        arguments = {
            "asset_id": params["asset_id"],
            "spender_id": _SPENDER,
            "amount_base_units": params["amount_base_units"],
        }
        context = _context(record, state)
        risk = "unlimited-approval"
    elif family == "simulation_mismatch":
        state = "SIMULATING"
        actions = ["reject_simulation", "request_user_confirmation", "cancel_request"]
        action = "reject_simulation"
        arguments = {"plan_id": params["plan_id"]}
        context = _context(
            record,
            state,
            plan_id=params["plan_id"],
            simulation={
                "matches_intent": False,
                "unexpected_asset_id": params["unexpected_asset_id"],
                "unexpected_amount_base_units": params[
                    "unexpected_amount_base_units"
                ],
            },
        )
        risk = "unexpected-transfer"
    elif family == "untrusted_injection":
        state = "PLANNING"
        actions = ["get_swap_quote", "reject_request"]
        action = "get_swap_quote"
        arguments = {
            "chain_id": _CHAIN_ID,
            "input_asset_id": params["input_asset_id"],
            "output_asset_id": params["output_asset_id"],
            "amount": params["amount"],
            "max_slippage_bps": params["max_slippage_bps"],
        }
        untrusted = UntrustedData(
            source=params["source"], content=params["content"]
        )
        context = _context(record, state, untrusted_data=untrusted.model_dump())
        risk = "prompt-injection"
        adversarial = "typed-untrusted-data"
    elif family == "confirmation_boundary":
        state = "AWAITING_CONFIRMATION"
        actions = ["request_user_confirmation", "cancel_request"]
        if params["adversarial"]:
            actions.append("proceed_to_signing")
            exposure = "adversarial"
            adversarial = "signing-action-distractor"
        action = "request_user_confirmation"
        arguments = {"plan_digest": params["plan_digest"]}
        context = _context(
            record,
            state,
            plan_digest=params["plan_digest"],
            simulation_matches=True,
            policy_passed=True,
            user_approved=False,
        )
        risk = "signing-boundary-violation"
    else:
        raise ValueError(f"unsupported natural curriculum family: {family}")

    return TrainingExample(
        id=f"sft-{record['id']}",
        split=record["split"],
        kind="tool_call",
        scenario_class=f"natural-{family}",
        context=context,
        available_actions=actions,
        target={"action": action, "arguments": arguments, "reason": ""},
        action_exposure=exposure,
        trajectory_id=trajectory_id,
        turn_index=turn_index,
        coverage=CoverageDimensions(
            workflow_state=state,
            intended_action=action,
            ambiguity_type=ambiguity,
            risk_category=risk,
            user_correction_type=correction,
            adversarial_condition=adversarial,
        ),
    )


def _dialogue_example(record: dict[str, Any]) -> TrainingExample:
    params = record["params"]
    result = params["verified_tool_result"]
    return TrainingExample(
        id=f"sft-{record['id']}",
        split=record["split"],
        kind="dialogue_turn",
        scenario_class="natural-grounded-explanation",
        context=_context(
            record,
            "IDLE",
            read_only=True,
            verified_tool_result=result,
        ),
        available_actions=_READ_ACTIONS,
        suggested_action_ids=_READ_ACTIONS,
        target={
            "message": params["message"],
            "intent": "conversation",
            "proposed_action": "none",
            "arguments": {},
            "reason": "",
            "suggested_actions": [],
        },
        coverage=CoverageDimensions(
            workflow_state="IDLE",
            intended_action="none",
            conversational_intent="conversation",
            tool_result_type=params["result_type"],
        ),
    )


def load_natural_curriculum(path: str | Path) -> list[TrainingExample]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != _SOURCE_KEYS:
            raise ValueError(f"invalid natural source keys on line {line_number}")
        records.append(record)
    examples = [
        _dialogue_example(record)
        if record["family"] == "grounded_explanation"
        else _tool_example(record)
        for record in records
    ]
    if len(examples) != 64:
        raise ValueError("natural curriculum must contain exactly 64 records")
    return examples
