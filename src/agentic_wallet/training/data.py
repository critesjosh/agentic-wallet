"""Training-example schema and fail-closed leakage validation."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ..benchmark.cases import BenchmarkCase
from ..benchmark.registries import EVAL_REGISTRY, TRAIN_REGISTRY
from ..inference import InferenceError
from ..schemas.common import StrictModel
from ..schemas.tool_call import ToolCall
from ..tool_contract import validate_dialogue_turn, validate_tool_arguments

TrainingKind = Literal["tool_call", "dialogue_turn"]
FORBIDDEN_TRAINING_TARGETS = frozenset(
    {"proceed_to_signing", "create_unlimited_approval_plan"}
)


class TrainingExample(StrictModel):
    """One completion-only SFT record; frozen benchmark text is never eligible."""

    id: str = Field(pattern=r"^sft-[a-z0-9-]+$")
    kind: TrainingKind
    scenario_class: str = Field(min_length=1)
    context: dict[str, Any]
    available_actions: list[str] = Field(min_length=1)
    suggested_action_ids: list[str] = Field(default_factory=list)
    target: dict[str, Any]


@dataclass(frozen=True)
class DatasetValidationReport:
    total: int
    tool_calls: int
    dialogue_turns: int
    label_counts: dict[str, int]
    max_benchmark_similarity: float


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
    return f"dialogue:{turn.intent}"


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
    benchmark_prompts = [_normalized(case.user_request) for case in frozen_benchmark]
    benchmark_ids = {case.id for case in frozen_benchmark}
    max_similarity = 0.0

    for example in examples:
        if example.id in benchmark_ids:
            raise ValueError(f"training id collides with frozen benchmark: {example.id}")
        _assert_no_eval_universe(example)
        label = _validate_target(example)
        labels[label] += 1

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

    largest_share = max(labels.values()) / len(examples)
    if largest_share > max_label_share:
        raise ValueError(
            f"training labels are imbalanced: largest share {largest_share:.3f}"
        )
    return DatasetValidationReport(
        total=len(examples),
        tool_calls=sum(example.kind == "tool_call" for example in examples),
        dialogue_turns=sum(example.kind == "dialogue_turn" for example in examples),
        label_counts=dict(sorted(labels.items())),
        max_benchmark_similarity=max_similarity,
    )
