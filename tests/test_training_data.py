from __future__ import annotations

from pathlib import Path

import pytest

from agentic_wallet.benchmark import load_cases
from agentic_wallet.training import (
    TrainingExample,
    generate_error_driven_training_examples,
    generate_training_examples,
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
