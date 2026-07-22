from __future__ import annotations

from pathlib import Path

import pytest

from agentic_wallet.benchmark import load_cases
from agentic_wallet.training import (
    TrainingExample,
    CoverageDimensions,
    balanced_semantic_subset,
    generate_error_driven_training_examples,
    generate_training_examples,
    load_candidate_pipeline_curriculum,
    load_natural_curriculum,
    load_pipeline_curriculum,
    validate_training_dataset,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def _benchmark():
    return load_cases(DATA / "benchmark" / "train_family.jsonl") + load_cases(
        DATA / "benchmark" / "eval_family.jsonl"
    )


def test_generated_dataset_is_balanced_valid_and_benchmark_isolated():
    examples = generate_training_examples(tool_count=96, dialogue_count=48)
    report = validate_training_dataset(examples, _benchmark())
    assert report.total == 144
    assert report.tool_calls == 96
    assert report.dialogue_turns == 48
    assert report.max_benchmark_similarity < 0.94
    assert max(report.label_counts.values()) / report.total <= 0.35


def test_generation_is_reproducible():
    first = generate_training_examples(tool_count=16, dialogue_count=12, seed=3)
    second = generate_training_examples(tool_count=16, dialogue_count=12, seed=3)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]


def test_error_driven_dataset_is_valid_and_covers_observed_failures():
    examples = generate_error_driven_training_examples()
    report = validate_training_dataset(examples, _benchmark())
    scenarios = {example.scenario_class for example in examples}
    targets = {
        example.target.get("action")
        for example in examples
        if example.kind == "tool_call"
    }

    assert report.total == 576
    assert report.tool_calls == 504
    assert report.dialogue_turns == 72
    assert report.max_benchmark_similarity < 0.94
    assert "v2-confirmation-before-signing" in scenarios
    assert "v2-transfer-canonical-arguments" in scenarios
    assert "v2-simulation-plan-id-required" in scenarios
    assert "proceed_to_signing" not in targets
    assert "create_unlimited_approval_plan" not in targets
    assert any(
        "proceed_to_signing" in example.available_actions
        for example in examples
        if example.kind == "tool_call"
    )


def test_unsafe_training_target_is_rejected():
    example = generate_error_driven_training_examples(tool_count=12, dialogue_count=6)[11]
    bad = example.model_copy(
        update={
            "target": {
                "action": "proceed_to_signing",
                "arguments": {},
                "reason": "unsafe",
            }
        }
    )
    with pytest.raises(ValueError, match="unsafe training target"):
        validate_training_dataset([bad], _benchmark(), max_label_share=1.0)


def test_natural_workflow_dataset_has_split_trajectories_and_grounding():
    examples = load_natural_curriculum(DATA / "training" / "natural_v3_source.jsonl")
    report = validate_training_dataset(examples, _benchmark())
    trajectories = {
        example.trajectory_id for example in examples if example.trajectory_id
    }

    assert report.total == 64
    assert report.tool_calls == 56
    assert report.dialogue_turns == 8
    assert len(trajectories) == 4
    assert report.split_counts == {"train": 48, "validation": 16}
    assert report.coverage_counts["tool_result_type"]["simulation"] == 2
    assert report.coverage_counts["user_correction_type"]["recipient"] == 4
    assert report.coverage_counts["adversarial_condition"][
        "signing-action-distractor"
    ] == 4
    assert all(
        example.target.get("reason") == ""
        for example in examples
        if example.kind == "tool_call"
    )


def test_pipeline_curriculum_matches_runtime_phases():
    examples = load_pipeline_curriculum(
        DATA / "training" / "natural_v3_source.jsonl"
    )
    report = validate_training_dataset(examples, _benchmark())
    routes = [item for item in examples if item.kind == "dialogue_route"]
    argument_repairs = [
        item
        for item in examples
        if item.context.get("phase") == "repair_tool_arguments"
    ]
    route_repairs = [
        item
        for item in examples
        if item.context.get("phase") == "repair_dialogue_route"
    ]
    arguments = [
        item
        for item in examples
        if item.context.get("phase") == "fill_tool_arguments"
    ]

    assert report.total == 240
    assert report.tool_calls == 112
    assert report.dialogue_routes == 128
    assert report.dialogue_turns == 0
    assert report.split_counts == {"train": 180, "validation": 60}
    assert len(argument_repairs) == 56
    assert len(route_repairs) == 64
    assert len(arguments) == 56
    assert all("arguments" not in item.target for item in routes)
    assert all(
        len(item.available_actions) == 1 for item in argument_repairs + arguments
    )
    assert all("arguments" in item.context["previous_output"] for item in route_repairs)
    assert all("conversation_ledger" in item.context for item in examples)
    train_requests = {
        item.context["user_request"] for item in examples if item.split == "train"
    }
    validation_requests = {
        item.context["user_request"]
        for item in examples
        if item.split == "validation"
    }
    assert train_requests.isdisjoint(validation_requests)
    assert all(
        item.target["action"]
        not in {"proceed_to_signing", "create_unlimited_approval_plan"}
        for item in argument_repairs
    )
    forbidden_markers = {
        "drill",
        "sample",
        "generated",
        "trajectory wording",
    }
    assert all(
        not any(
            marker in str(example.context["user_request"]).casefold()
            for marker in forbidden_markers
        )
        for example in examples
    )


def test_candidate_pipeline_removes_free_generated_transfer_arguments():
    examples = load_candidate_pipeline_curriculum(
        DATA / "training" / "natural_v3_source.jsonl"
    )
    report = validate_training_dataset(examples, _benchmark())
    candidate_routes = [
        item
        for item in examples
        if item.target.get("proposed_action")
        == "create_transfer_plan_from_candidate"
    ]

    assert report.total == 232
    assert report.tool_calls == 104
    assert report.dialogue_routes == 128
    assert candidate_routes
    assert all(
        item.target.get("action") != "create_transfer_plan_from_candidate"
        for item in examples
        if item.kind == "tool_call"
    )
    assert all(
        item.context["trusted_recipient_candidates"][0]["recipient_id"].startswith(
            "recipient:user-"
        )
        for item in candidate_routes
    )
    assert all(
        "recipient" not in item.target.get("arguments", {})
        for item in examples
        if item.kind == "tool_call"
    )


def test_pipeline_checkpoint_subset_round_robins_runtime_phases():
    validation = [
        item
        for item in load_pipeline_curriculum(
            DATA / "training" / "natural_v3_source.jsonl"
        )
        if item.split == "validation"
    ]
    selected = balanced_semantic_subset(validation, 20)
    phase_counts: dict[str, int] = {}
    for item in selected:
        phase = item.context["phase"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    assert phase_counts == {
        "explain_verified_tool_result": 2,
        "fill_tool_arguments": 5,
        "repair_dialogue_route": 5,
        "repair_tool_arguments": 4,
        "route_dialogue": 4,
    }


def test_production_examples_cannot_expose_unsafe_actions():
    example = TrainingExample(
        id="sft-production-unsafe-exposure",
        kind="tool_call",
        scenario_class="bad-production-contract",
        context={"user_request": "Review a unique pending plan safely."},
        available_actions=["request_user_confirmation", "proceed_to_signing"],
        target={
            "action": "request_user_confirmation",
            "arguments": {"plan_digest": "sha256:" + "c" * 64},
            "reason": "review",
        },
        action_exposure="production",
    )
    with pytest.raises(ValueError, match="exposes unsafe actions"):
        validate_training_dataset([example], _benchmark(), max_label_share=1.0)


def test_grounded_dialogue_rejects_unsupported_result_claim():
    example = next(
        item
        for item in load_natural_curriculum(
            DATA / "training" / "natural_v3_source.jsonl"
        )
        if item.coverage.tool_result_type == "balance"
    )
    assert example.kind == "dialogue_turn"
    assert example.coverage.tool_result_type == "balance"
    bad = example.model_copy(
        update={
            "target": {
                **example.target,
                "message": "The verified result says 999999999 base units for base:usdc.",
            },
            "coverage": CoverageDimensions(
                workflow_state="IDLE",
                conversational_intent="conversation",
                tool_result_type="balance",
            ),
        }
    )
    with pytest.raises(ValueError, match="(cites no|invents) typed result fact"):
        validate_training_dataset([bad], _benchmark(), max_label_share=1.0)


def test_eval_universe_marker_is_rejected_even_past_normal_fields():
    example = generate_training_examples(tool_count=1, dialogue_count=1)[0]
    bad = example.model_copy(
        update={"context": {**example.context, "note": "base:dai"}}
    )
    with pytest.raises(ValueError, match="held-out registry"):
        validate_training_dataset([bad], _benchmark(), max_label_share=1.0)


def test_exact_benchmark_prompt_is_rejected():
    benchmark = _benchmark()
    example = generate_training_examples(tool_count=1, dialogue_count=1)[0]
    bad = example.model_copy(
        update={
            "context": {
                **example.context,
                "user_request": benchmark[0].user_request,
            }
        }
    )
    with pytest.raises(ValueError, match="too similar"):
        validate_training_dataset([bad], benchmark, max_label_share=1.0)


def test_invalid_target_arguments_fail_closed():
    example = TrainingExample(
        id="sft-invalid-target",
        kind="tool_call",
        scenario_class="bad",
        context={"user_request": "A unique request that is not benchmark text."},
        available_actions=["get_balance"],
        target={"action": "get_balance", "arguments": {}, "reason": "bad"},
    )
    with pytest.raises(ValueError, match="invalid arguments"):
        validate_training_dataset([example], _benchmark(), max_label_share=1.0)
