"""Training-example schema and fail-closed leakage validation."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ..benchmark.cases import BenchmarkCase
from ..benchmark.registries import EVAL_REGISTRY, TRAIN_REGISTRY
from ..inference import InferenceError
from ..schemas.common import StrictModel
from ..schemas.tool_call import ToolCall
from ..tool_contract import (
    validate_dialogue_route,
    validate_dialogue_turn,
    validate_tool_arguments,
)

TrainingKind = Literal["tool_call", "dialogue_turn", "dialogue_route"]
ActionExposure = Literal["production", "adversarial"]
DatasetSplit = Literal["train", "validation"]
FORBIDDEN_TRAINING_TARGETS = frozenset(
    {"proceed_to_signing", "create_unlimited_approval_plan"}
)


class CoverageDimensions(StrictModel):
    """Auditable dimensions that label counts alone cannot expose."""

    workflow_state: str = "unspecified"
    intended_action: str = "unspecified"
    ambiguity_type: str = "none"
    risk_category: str = "none"
    conversational_intent: str = "none"
    tool_result_type: str = "none"
    user_correction_type: str = "none"
    adversarial_condition: str = "none"


class TrainingExample(StrictModel):
    """One completion-only SFT record; frozen benchmark text is never eligible."""

    id: str = Field(pattern=r"^sft-[a-z0-9-]+$")
    kind: TrainingKind
    scenario_class: str = Field(min_length=1)
    context: dict[str, Any]
    available_actions: list[str]
    suggested_action_ids: list[str] = Field(default_factory=list)
    target: dict[str, Any]
    split: DatasetSplit = "train"
    action_exposure: ActionExposure | None = None
    trajectory_id: str | None = Field(default=None, pattern=r"^trajectory-[a-z0-9-]+$")
    turn_index: int | None = Field(default=None, ge=0)
    coverage: CoverageDimensions = Field(default_factory=CoverageDimensions)

    @model_validator(mode="after")
    def _trajectory_fields_are_paired(self) -> "TrainingExample":
        if self.action_exposure is None:
            unsafe_available = FORBIDDEN_TRAINING_TARGETS.intersection(
                self.available_actions
            )
            self.action_exposure = (
                "adversarial" if unsafe_available else "production"
            )
        if (self.trajectory_id is None) != (self.turn_index is None):
            raise ValueError("trajectory_id and turn_index must be provided together")
        return self
@dataclass(frozen=True)
class DatasetValidationReport:
    total: int
    tool_calls: int
    dialogue_turns: int
    dialogue_routes: int
    label_counts: dict[str, int]
    max_benchmark_similarity: float
    coverage_counts: dict[str, dict[str, int]]
    split_counts: dict[str, int]


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(TrainingExample.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid training example on line {line_number}: {exc}") from exc
    return examples


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _serialized(example: TrainingExample) -> str:
    return json.dumps(example.model_dump(), sort_keys=True, separators=(",", ":"))


def _exclusive_eval_markers() -> tuple[set[str], set[str]]:
    literal: set[str] = set()
    symbols: set[str] = set()
    for entry in EVAL_REGISTRY.entries():
        if entry.asset_id == "base:native":
            continue
        literal.update({entry.asset_id.casefold(), entry.address.casefold()})
        symbols.add(entry.symbol.casefold())
    return literal, symbols


def _assert_no_eval_universe(example: TrainingExample) -> None:
    text = _serialized(example).casefold()
    literal, symbols = _exclusive_eval_markers()
    leaked = sorted(marker for marker in literal if marker in text)
    leaked.extend(
        symbol
        for symbol in sorted(symbols)
        if re.search(rf"(?<![a-z0-9]){re.escape(symbol)}(?![a-z0-9])", text)
    )
    if leaked:
        raise ValueError(
            f"training example {example.id} leaks held-out registry markers: {leaked}"
        )


def _assert_train_registry_arguments(action: str, arguments: dict[str, Any]) -> None:
    for key, value in arguments.items():
        if key == "asset_id" or key == "spender_id" or key.endswith("_asset_id"):
            try:
                TRAIN_REGISTRY.resolve(value)
            except Exception as exc:
                raise ValueError(
                    f"training target {action} references non-training canonical id {value!r}"
                ) from exc


def _validate_target(example: TrainingExample) -> str:
    unsafe_available = FORBIDDEN_TRAINING_TARGETS.intersection(
        example.available_actions
    )
    if example.action_exposure == "production" and unsafe_available:
        raise ValueError(
            f"production example {example.id} exposes unsafe actions: "
            f"{sorted(unsafe_available)}"
        )
    if example.kind == "tool_call":
        try:
            call = ToolCall.model_validate(example.target)
        except Exception as exc:
            raise ValueError(f"invalid tool target for {example.id}: {exc}") from exc
        if call.action not in example.available_actions:
            raise ValueError(f"target action is unavailable for {example.id}")
        if call.action in FORBIDDEN_TRAINING_TARGETS:
            raise ValueError(f"unsafe training target for {example.id}: {call.action}")
        try:
            validate_tool_arguments(call.action, call.arguments)
        except InferenceError as exc:
            raise ValueError(f"invalid arguments in training target {example.id}: {exc}") from exc
        _assert_train_registry_arguments(call.action, call.arguments)
        return f"tool:{call.action}"

    if example.kind == "dialogue_route":
        try:
            route = validate_dialogue_route(
                example.target,
                example.available_actions,
                example.suggested_action_ids,
            )
        except InferenceError as exc:
            raise ValueError(f"invalid dialogue route for {example.id}: {exc}") from exc
        _assert_grounded_dialogue(example, route.message)
        route_label = route.proposed_action or route.intent
        return f"route:{route_label}"

    try:
        turn = validate_dialogue_turn(
            example.target,
            example.available_actions,
            example.suggested_action_ids,
        )
    except InferenceError as exc:
        raise ValueError(f"invalid dialogue target for {example.id}: {exc}") from exc
    if turn.proposed_action is not None:
        _assert_train_registry_arguments(
            turn.proposed_action.action, turn.proposed_action.arguments
        )
    _assert_grounded_dialogue(example, turn.message)
    return f"dialogue:{turn.intent}"


def _scalar_facts(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {fact for child in value.values() for fact in _scalar_facts(child)}
    if isinstance(value, list):
        return {fact for child in value for fact in _scalar_facts(child)}
    if isinstance(value, (str, int)):
        return {str(value).casefold()}
    return set()


def _assert_grounded_dialogue(example: TrainingExample, message: str) -> None:
    lowered = message.casefold()
    forbidden_execution_claims = (
        "i signed",
        "i submitted",
        "i sent",
        "transaction completed",
    )
    if any(claim in lowered for claim in forbidden_execution_claims):
        raise ValueError(f"dialogue target claims wallet execution: {example.id}")

    if example.coverage.tool_result_type == "none":
        return
    if "verified_tool_result" not in example.context:
        raise ValueError(f"grounded dialogue lacks typed tool result: {example.id}")
    facts = _scalar_facts(example.context["verified_tool_result"])
    referenced = {fact for fact in facts if fact in lowered}
    if not referenced:
        raise ValueError(f"grounded dialogue cites no typed result fact: {example.id}")
    mentioned_values = set(
        re.findall(r"(?<![a-z0-9-])\d+(?![a-z0-9-])", lowered)
    )
    mentioned_values.update(
        re.findall(r"(?<![a-z0-9])[a-z0-9]+:[a-z0-9-]+", lowered)
    )
    unsupported = sorted(value for value in mentioned_values if value not in facts)
    if unsupported:
        raise ValueError(
            f"grounded dialogue invents typed result facts for {example.id}: "
            f"{unsupported}"
        )


def validate_training_dataset(
    examples: list[TrainingExample],
    frozen_benchmark: list[BenchmarkCase],
    *,
    near_duplicate_threshold: float = 0.94,
    max_label_share: float = 0.35,
) -> DatasetValidationReport:
    """Validate schema, ground truth, balance, deduplication, and C3 isolation."""

    if not examples:
        raise ValueError("training dataset is empty")
    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("training example ids must be unique")

    fingerprints: set[str] = set()
    labels: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    coverage: dict[str, Counter[str]] = {
        field: Counter() for field in CoverageDimensions.model_fields
    }
    benchmark_prompts = [_normalized(case.user_request) for case in frozen_benchmark]
    benchmark_ids = {case.id for case in frozen_benchmark}
    max_similarity = 0.0
    trajectory_turns: dict[str, list[int]] = {}

    for example in examples:
        if example.id in benchmark_ids:
            raise ValueError(f"training id collides with frozen benchmark: {example.id}")
        _assert_no_eval_universe(example)
        label = _validate_target(example)
        labels[label] += 1
        splits[example.split] += 1
        for field in coverage:
            coverage[field][getattr(example.coverage, field)] += 1
        if example.trajectory_id is not None:
            trajectory_turns.setdefault(example.trajectory_id, []).append(
                int(example.turn_index)
            )
            has_legacy_history = isinstance(
                example.context.get("conversation_history"), list
            )
            has_typed_ledger = isinstance(
                example.context.get("conversation_ledger"), dict
            )
            if not (has_legacy_history or has_typed_ledger):
                raise ValueError(
                    f"trajectory example lacks typed context: {example.id}"
                )

        fingerprint = json.dumps(
            {
                "kind": example.kind,
                "context": example.context,
                "available_actions": example.available_actions,
                "target": example.target,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate training content: {example.id}")
        fingerprints.add(fingerprint)

        request = _normalized(str(example.context.get("user_request", "")))
        if not request:
            raise ValueError(f"training example {example.id} has no user_request")
        for benchmark_prompt in benchmark_prompts:
            similarity = SequenceMatcher(None, request, benchmark_prompt).ratio()
            max_similarity = max(max_similarity, similarity)
            if request == benchmark_prompt or similarity >= near_duplicate_threshold:
                raise ValueError(
                    f"training example {example.id} is too similar to frozen benchmark "
                    f"({similarity:.3f})"
                )

    for trajectory_id, indexes in trajectory_turns.items():
        if sorted(indexes) != list(range(len(indexes))):
            raise ValueError(f"trajectory turns are not contiguous: {trajectory_id}")

    largest_share = max(labels.values()) / len(examples)
    if largest_share > max_label_share:
        raise ValueError(
            f"training labels are imbalanced: largest share {largest_share:.3f}"
        )
    return DatasetValidationReport(
        total=len(examples),
        tool_calls=sum(example.kind == "tool_call" for example in examples),
        dialogue_turns=sum(example.kind == "dialogue_turn" for example in examples),
        dialogue_routes=sum(example.kind == "dialogue_route" for example in examples),
        label_counts=dict(sorted(labels.items())),
        max_benchmark_similarity=max_similarity,
        coverage_counts={
            field: dict(sorted(counts.items())) for field, counts in coverage.items()
        },
        split_counts=dict(sorted(splits.items())),
    )
