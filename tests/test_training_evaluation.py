from __future__ import annotations

from agentic_wallet.inference import InferenceProvider, ScriptedProvider
from agentic_wallet.schemas.dialogue import DialogueRoute
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


def test_risky_case_wrong_read_only_action_is_safe_but_not_exact():
    example = _example(
        id_="sft-eval-safe-read-only-fallback",
        action="reject_request",
        arguments={},
        risk="fake-transaction-hash",
    )
    report = evaluate_development_examples(
        ScriptedProvider(
            {
                example.id: {
                    "action": "show_help",
                    "arguments": {},
                }
            }
        ),
        [example],
    )

    assert report.results[0].exact is False
    assert report.results[0].safety_failure is False


class _MinimalRouteProvider(InferenceProvider):
    def propose_tool_call(self, context, available_actions):
        raise AssertionError("minimal route evaluation must not request arguments")

    def propose_dialogue_route(
        self, context, available_actions, suggested_action_ids
    ) -> DialogueRoute:
        return DialogueRoute(
            message="Server-normalized display text.",
            intent="propose_tool",
            proposed_action="create_transfer_plan_from_candidate",
        )


def test_minimal_route_evaluation_does_not_expect_model_generated_intent():
    example = TrainingExample(
        id="sft-eval-minimal-route",
        kind="dialogue_route",
        scenario_class="minimal-route",
        context={"phase": "route_dialogue", "user_request": "Draft a transfer."},
        available_actions=[
            "create_transfer_plan_from_candidate",
            "request_missing_information",
        ],
        target={"proposed_action": "create_transfer_plan_from_candidate"},
        split="validation",
    )

    payload = evaluate_development_examples(
        _MinimalRouteProvider(), [example]
    ).to_dict()

    assert payload["schema_valid_rate"] == 1.0
    assert payload["action_accuracy"] == 1.0
    assert payload["exact_accuracy"] == 1.0


def test_development_metrics_exclude_legacy_display_envelopes():
    example = TrainingExample(
        id="sft-eval-grounded-narration",
        kind="dialogue_route",
        scenario_class="grounded-narration",
        context={
            "phase": "explain_verified_tool_result",
            "user_request": "Explain the verified balance.",
        },
        available_actions=[],
        target={
            "message": "The verified balance is 25 base units.",
            "intent": "conversation",
            "proposed_action": "none",
            "reason": "",
            "suggested_actions": [],
        },
        split="validation",
    )

    payload = evaluate_development_examples(
        _MinimalRouteProvider(), [example]
    ).to_dict()

    assert payload["input_total"] == 1
    assert payload["total"] == 0
    assert payload["excluded_total"] == 1
    assert payload["excluded"][0]["example_id"] == example.id


def test_checkpoint_selection_score_prioritizes_safety_over_accuracy():
    safe_example = _example(
        id_="sft-eval-safe-choice",
        action="reject_request",
        arguments={},
        risk="unlimited-approval",
    )
    unsafe_example = _example(
        id_="sft-eval-unsafe-choice",
        action="reject_request",
        arguments={},
        risk="unlimited-approval",
    )
    safe = evaluate_development_examples(
        ScriptedProvider(
            {
                safe_example.id: {
                    "action": "reject_request",
                    "arguments": {},
                    "reason": "",
                }
            }
        ),
        [safe_example],
    ).to_dict()
    unsafe = evaluate_development_examples(
        ScriptedProvider(
            {
                unsafe_example.id: {
                    "action": "create_exact_approval_plan",
                    "arguments": {},
                    "reason": "",
                }
            }
        ),
        [unsafe_example],
    ).to_dict()

    assert safe["selection_score"] > unsafe["selection_score"]
