from __future__ import annotations

from agentic_wallet.inference import ScriptedProvider
from agentic_wallet.training import (
    CoverageDimensions,
    TrainingExample,
    evaluate_development_examples,
)


def _example(
    *,
    id_: str,
    action: str,
    arguments: dict,
    trajectory_id: str | None = None,
    turn_index: int | None = None,
    risk: str = "none",
) -> TrainingExample:
    return TrainingExample(
        id=id_,
        kind="tool_call",
        scenario_class=id_,
        context={"scenario_id": id_, "user_request": f"Unique request {id_}."},
        available_actions=[action, "reject_request"],
        target={"action": action, "arguments": arguments, "reason": ""},
        split="validation",
        trajectory_id=trajectory_id,
        turn_index=turn_index,
        coverage=CoverageDimensions(
            workflow_state="PLANNING",
            intended_action=action,
            risk_category=risk,
        ),
    )


def test_development_metrics_separate_argument_counts_and_sequences():
    examples = [
        _example(
            id_="sft-eval-zero",
            action="reject_request",
            arguments={},
        ),
        _example(
            id_="sft-eval-single",
            action="get_balance",
            arguments={"asset_id": "base:usdc"},
            trajectory_id="trajectory-eval",
            turn_index=0,
        ),
        _example(
            id_="sft-eval-multi",
            action="create_transfer_plan",
            arguments={
                "chain_id": 8453,
                "asset_id": "base:usdc",
                "amount_base_units": "25",
                "recipient": "0x1111111111111111111111111111111111111111",
            },
            trajectory_id="trajectory-eval",
            turn_index=1,
            risk="wrong-recipient",
        ),
    ]
    script = {
        example.id: {
            "action": example.target["action"],
            "arguments": example.target["arguments"],
            "reason": "",
        }
        for example in examples
    }
    report = evaluate_development_examples(ScriptedProvider(script), examples)
    payload = report.to_dict()

    assert payload["schema_valid_rate"] == 1.0
    assert payload["action_accuracy"] == 1.0
    assert payload["argument_accuracy"] == 1.0
    assert payload["sequence_accuracy"] == 1.0
    assert set(payload["by_argument_count"]) == {"zero", "single", "multi"}
    assert payload["hard_zero"]["wrong-recipient"]["failures"] == 0


def test_development_metrics_count_risky_wrong_arguments_as_safety_failure():
    example = _example(
        id_="sft-eval-risky",
        action="create_transfer_plan",
        arguments={
            "chain_id": 8453,
            "asset_id": "base:usdc",
            "amount_base_units": "25",
            "recipient": "0x1111111111111111111111111111111111111111",
        },
        risk="wrong-recipient",
    )
    script = {
        example.id: {
            "action": "create_transfer_plan",
            "arguments": {
                **example.target["arguments"],
                "recipient": "0x2222222222222222222222222222222222222222",
            },
            "reason": "",
        }
    }
    payload = evaluate_development_examples(
        ScriptedProvider(script), [example]
    ).to_dict()

    assert payload["argument_accuracy"] == 0.0
    assert payload["safety_failures"] == 1
    assert payload["hard_zero"]["wrong-recipient"]["failures"] == 1
