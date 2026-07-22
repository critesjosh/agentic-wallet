"""Adapt the fixed natural curriculum to the runtime's multi-stage pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..candidate_binding import prepare_inference_context
from .data import TrainingExample
from .natural_curriculum import load_natural_curriculum

PIPELINE_CURRICULUM_VERSION = "wallet-pipeline-curriculum-v4-2"
CANDIDATE_PIPELINE_CURRICULUM_VERSION = "wallet-pipeline-curriculum-v5-2"
_LEGACY_TRANSFER_ACTION = "create_transfer_plan"
_CANDIDATE_TRANSFER_ACTION = "create_transfer_plan_from_candidate"


def _ledger_context(context: dict[str, Any]) -> dict[str, Any]:
    copied = dict(context)
    state = str(copied.pop("workflow_state", "IDLE"))
    chain_id = int(copied.pop("chain_id", 8453))
    history = copied.pop("conversation_history", [])
    recent_messages = []
    prior_proposals = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", item.get("message"))
        if role in {"user", "assistant"} and isinstance(content, str):
            recent_messages.append({"role": role, "content": content})
        if isinstance(item.get("typed_action"), str):
            prior_proposals.append(
                {
                    "action": item["typed_action"],
                    "arguments": item.get("arguments", {}),
                    "status": "validated",
                }
            )
    copied["conversation_ledger"] = {
        "workflow_state": state,
        "chain_id": chain_id,
        "resolved_intent": {
            "chain_id": None,
            "asset_id": None,
            "amount": None,
            "amount_base_units": None,
            "recipient": None,
        },
        "missing_fields": [],
        "active_plan_id": copied.get("plan_id"),
        "active_quote_id": copied.get("quote_id"),
        "corrections": [],
        "verified_facts": [],
        "prior_proposals": prior_proposals[-4:],
        "recent_messages": recent_messages[-8:],
    }
    return copied


def _invalid_arguments(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if action == "get_swap_quote":
        return {
            "asset_id": arguments.get("input_asset_id"),
            "output_asset_id": arguments.get("output_asset_id"),
            "amount": arguments.get("amount"),
            "max_slippage_percent": "0.5",
        }
    if action == "request_missing_information":
        return {}
    if action == "create_transfer_plan":
        return {
            "asset_id": arguments.get("asset_id"),
            "amount": arguments.get("amount_base_units"),
            "to": arguments.get("recipient"),
        }
    if action == "refresh_swap_quote":
        return {"scenario_id": arguments.get("quote_id")}
    if action in {"reject_simulation", "show_existing_plan"}:
        return {"scenario_id": arguments.get("plan_id")}
    if action == "create_exact_approval_plan":
        return {
            "asset_id": arguments.get("asset_id"),
            "spender": arguments.get("spender_id"),
            "amount": arguments.get("amount_base_units"),
        }
    if action == "request_user_confirmation":
        return {"digest": arguments.get("plan_digest")}
    if action == "get_balance":
        return {"asset": arguments.get("asset_id")}
    return {"reason": "top-level reason incorrectly nested in arguments"}


def _deterministic_summary(kind: str, result: dict[str, Any]) -> str:
    if kind == "balance":
        return (
            f"Verified {result['asset_id']} balance: "
            f"{result['balance_base_units']} base units."
        )
    if kind == "allowance":
        return (
            f"Verified allowance for {result['asset_id']} to "
            f"{result['spender_id']}: {result['allowance_base_units']} base units."
        )
    if kind == "portfolio":
        portfolio = result["portfolio"]
        return (
            f"Watch-only portfolio for {portfolio['address']} on chain "
            f"{portfolio['chain_id']} (block {portfolio['as_of_block']})."
        )
    if kind == "registry":
        return "Canonical registry (the trusted id to address mapping):"
    if kind == "quote":
        return f"Verified quote {result['quote_id']}."
    if kind == "simulation":
        return f"Verified simulation for {result['plan_id']}."
    raise ValueError(f"unsupported grounded result type: {kind}")


def _route_from_tool(example: TrainingExample) -> TrainingExample:
    action = str(example.target["action"])
    trajectory = (
        f"{example.trajectory_id}-route" if example.trajectory_id else None
    )
    return TrainingExample(
        id=example.id.replace("sft-", "sft-v4-route-", 1),
        split=example.split,
        kind="dialogue_route",
        scenario_class=f"route-{example.scenario_class}",
        context={**_ledger_context(example.context), "phase": "route_dialogue"},
        available_actions=example.available_actions,
        suggested_action_ids=[],
        target={
            "message": "I will prepare the next validated workflow step.",
            "intent": "propose_tool",
            "proposed_action": action,
            "reason": "",
            "suggested_actions": [],
        },
        action_exposure=example.action_exposure,
        trajectory_id=trajectory,
        turn_index=example.turn_index,
        coverage=example.coverage,
    )


def _arguments_from_tool(example: TrainingExample) -> TrainingExample:
    action = str(example.target["action"])
    trajectory = (
        f"{example.trajectory_id}-arguments" if example.trajectory_id else None
    )
    return example.model_copy(
        update={
            "id": example.id.replace("sft-", "sft-v4-arguments-", 1),
            "context": {
                **_ledger_context(example.context),
                "phase": "fill_tool_arguments",
                "selected_action": action,
            },
            "available_actions": [action],
            "action_exposure": "production",
            "trajectory_id": trajectory,
        }
    )


def _repair_from_tool(example: TrainingExample) -> TrainingExample:
    action = str(example.target["action"])
    arguments = example.target["arguments"]
    return example.model_copy(
        update={
            "id": example.id.replace("sft-", "sft-v4-repair-", 1),
            "scenario_class": f"repair-{example.scenario_class}",
            "context": {
                **_ledger_context(example.context),
                "phase": "repair_tool_arguments",
                "selected_action": action,
                "previous_output": {
                    "action": action,
                    "arguments": _invalid_arguments(action, arguments),
                    "reason": "",
                },
                "validation_error": (
                    f"invalid arguments for {action}; use only its canonical fields"
                ),
                "repair_attempt": 1,
            },
            "available_actions": [action],
            "action_exposure": "production",
            "trajectory_id": None,
            "turn_index": None,
        }
    )


def _route_from_dialogue(example: TrainingExample) -> TrainingExample:
    context = _ledger_context(example.context)
    result = context["verified_tool_result"]
    context.update(
        {
            "phase": "explain_verified_tool_result",
            "deterministic_summary": _deterministic_summary(
                example.coverage.tool_result_type, result
            ),
        }
    )
    return example.model_copy(
        update={
            "id": example.id.replace("sft-", "sft-v4-narration-", 1),
            "kind": "dialogue_route",
            "context": context,
            "available_actions": [],
            "target": {
                "message": example.target["message"],
                "intent": "conversation",
                "proposed_action": "none",
                "reason": "",
                "suggested_actions": [],
            },
        }
    )


def _repair_from_route(example: TrainingExample) -> TrainingExample:
    previous_output = dict(example.target)
    previous_output["arguments"] = (
        {"asset_id": "base:usdc"}
        if example.target["proposed_action"] != "none"
        else {}
    )
    return example.model_copy(
        update={
            "id": example.id.replace("sft-v4-", "sft-v4-route-repair-", 1),
            "scenario_class": f"repair-{example.scenario_class}",
            "context": {
                **example.context,
                "phase": "repair_dialogue_route",
                "previous_output": previous_output,
                "validation_error": (
                    "invalid dialogue route; arguments belong in the separate "
                    "selected-action stage"
                ),
                "repair_attempt": 1,
            },
            "trajectory_id": None,
            "turn_index": None,
        }
    )


def load_pipeline_curriculum(path: str | Path) -> list[TrainingExample]:
    base = load_natural_curriculum(path)
    output: list[TrainingExample] = []
    for example in base:
        if example.kind == "dialogue_turn":
            route = _route_from_dialogue(example)
            output.extend([route, _repair_from_route(route)])
        else:
            route = _route_from_tool(example)
            output.extend(
                [
                    route,
                    _repair_from_route(route),
                    _arguments_from_tool(example),
                    _repair_from_tool(example),
                ]
            )
    if len(output) != 240:
        raise ValueError("pipeline curriculum must contain exactly 240 records")
    return output


def load_candidate_pipeline_curriculum(path: str | Path) -> list[TrainingExample]:
    """Adapt v4 to the candidate-bound runtime without rewriting its history.

    Candidate transfer arguments are constructed deterministically, so their
    old free-generation and repair records are intentionally removed. The model
    is trained only to route to the candidate-bound action or clarification.
    """

    output: list[TrainingExample] = []
    for example in load_pipeline_curriculum(path):
        target_action = example.target.get("action")
        if example.kind == "tool_call" and target_action == _LEGACY_TRANSFER_ACTION:
            continue

        available_actions = [
            _CANDIDATE_TRANSFER_ACTION
            if action == _LEGACY_TRANSFER_ACTION
            else action
            for action in example.available_actions
        ]
        target = dict(example.target)
        if target.get("proposed_action") == _LEGACY_TRANSFER_ACTION:
            target["proposed_action"] = _CANDIDATE_TRANSFER_ACTION
        if (
            example.kind == "dialogue_route"
            and example.coverage.tool_result_type == "none"
        ):
            target = {"proposed_action": target["proposed_action"]}
        context = prepare_inference_context(example.context)
        previous = context.get("previous_output")
        if isinstance(previous, dict):
            previous = dict(previous)
            if previous.get("proposed_action") == _LEGACY_TRANSFER_ACTION:
                previous["proposed_action"] = _CANDIDATE_TRANSFER_ACTION
            context["previous_output"] = previous
        coverage = example.coverage
        if coverage.intended_action == _LEGACY_TRANSFER_ACTION:
            coverage = coverage.model_copy(
                update={"intended_action": _CANDIDATE_TRANSFER_ACTION}
            )
        output.append(
            example.model_copy(
                update={
                    "id": example.id.replace("sft-v4-", "sft-v5-", 1),
                    "available_actions": available_actions,
                    "context": context,
                    "target": target,
                    "coverage": coverage,
                }
            )
        )
    if len(output) != 232:
        raise ValueError(
            "candidate pipeline curriculum must contain exactly 232 records"
        )
    return output
