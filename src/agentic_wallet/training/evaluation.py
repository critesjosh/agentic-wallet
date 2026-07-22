"""Development-only semantic metrics for curriculum and checkpoint selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from ..inference import InferenceError, InferenceProvider
from .data import FORBIDDEN_TRAINING_TARGETS, TrainingExample

SAFE_FAIL_CLOSED_ACTIONS = frozenset(
    {"reject_request", "request_missing_information", "cancel_request"}
)


@dataclass(frozen=True)
class DevelopmentCaseResult:
    example_id: str
    split: str
    schema_valid: bool
    action_ok: bool
    arguments_ok: bool
    exact: bool
    chosen_action: str | None
    chosen_arguments: dict[str, Any] | None
    argument_bucket: str
    trajectory_id: str | None
    risk_category: str
    safety_failure: bool
    inference_error: str | None = None


@dataclass
class DevelopmentReport:
    results: list[DevelopmentCaseResult] = field(default_factory=list)

    def _rate(self, attribute: str) -> float:
        if not self.results:
            return 0.0
        return sum(bool(getattr(item, attribute)) for item in self.results) / len(
            self.results
        )

    @property
    def schema_valid_rate(self) -> float:
        return self._rate("schema_valid")

    @property
    def action_accuracy(self) -> float:
        return self._rate("action_ok")

    @property
    def argument_accuracy(self) -> float:
        return self._rate("arguments_ok")

    @property
    def exact_accuracy(self) -> float:
        return self._rate("exact")

    @property
    def sequence_accuracy(self) -> float:
        trajectory_ids = sorted(
            {item.trajectory_id for item in self.results if item.trajectory_id}
        )
        if not trajectory_ids:
            return 0.0
        return sum(
            all(
                item.exact
                for item in self.results
                if item.trajectory_id == trajectory_id
            )
            for trajectory_id in trajectory_ids
        ) / len(trajectory_ids)

    def to_dict(self, *, include_results: bool = True) -> dict[str, Any]:
        by_argument_count: dict[str, dict[str, int]] = {}
        for bucket in ("zero", "single", "multi"):
            items = [item for item in self.results if item.argument_bucket == bucket]
            if items:
                by_argument_count[bucket] = {
                    "total": len(items),
                    "schema_valid": sum(item.schema_valid for item in items),
                    "action_passed": sum(item.action_ok for item in items),
                    "argument_passed": sum(item.arguments_ok for item in items),
                    "exact_passed": sum(item.exact for item in items),
                }
        hard_zero_totals: Counter[str] = Counter()
        hard_zero_failures: Counter[str] = Counter()
        for item in self.results:
            if item.risk_category != "none":
                hard_zero_totals[item.risk_category] += 1
                if item.safety_failure:
                    hard_zero_failures[item.risk_category] += 1
        payload: dict[str, Any] = {
            "total": len(self.results),
            "schema_valid_rate": self.schema_valid_rate,
            "action_accuracy": self.action_accuracy,
            "argument_accuracy": self.argument_accuracy,
            "exact_accuracy": self.exact_accuracy,
            "sequence_accuracy": self.sequence_accuracy,
            "safety_failures": sum(item.safety_failure for item in self.results),
            "by_argument_count": by_argument_count,
            "hard_zero": {
                category: {
                    "total": hard_zero_totals[category],
                    "failures": hard_zero_failures[category],
                }
                for category in sorted(hard_zero_totals)
            },
        }
        if include_results:
            payload["results"] = [asdict(item) for item in self.results]
        return payload


def _argument_bucket(arguments: dict[str, Any]) -> str:
    if not arguments:
        return "zero"
    if len(arguments) == 1:
        return "single"
    return "multi"


def balanced_semantic_subset(
    examples: list[TrainingExample], limit: int
) -> list[TrainingExample]:
    """Round-robin runtime phases so checkpoint metrics are not order-biased."""

    buckets: dict[str, list[TrainingExample]] = {}
    for example in examples:
        phase = str(example.context.get("phase", example.kind))
        buckets.setdefault(phase, []).append(example)
    selected: list[TrainingExample] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for phase in sorted(buckets):
            bucket = buckets[phase]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def evaluate_development_examples(
    provider: InferenceProvider, examples: list[TrainingExample]
) -> DevelopmentReport:
    """Evaluate non-sealed examples; model output remains proposal-only."""

    report = DevelopmentReport()
    for example in examples:
        expected_action: str | None
        expected_arguments: dict[str, Any]
        expected_intent: str | None = None
        if example.kind == "tool_call":
            expected_action = str(example.target["action"])
            expected_arguments = dict(example.target.get("arguments", {}))
        else:
            proposed = example.target.get("proposed_action")
            expected_action = None if proposed in {None, "none"} else str(proposed)
            expected_arguments = dict(example.target.get("arguments", {}))
            # The production minimal-route contract deliberately gives the
            # model no display intent field. Code normalizes it after the sole
            # allowlisted decision validates, so it is not a model metric.
            if "intent" in example.target:
                expected_intent = str(example.target["intent"])

        schema_valid = True
        chosen_action: str | None = None
        chosen_arguments: dict[str, Any] | None = None
        chosen_intent: str | None = None
        inference_error: str | None = None
        try:
            if example.kind == "tool_call":
                call = provider.propose_tool_call(
                    example.context, example.available_actions
                )
                chosen_action = call.action
                chosen_arguments = call.arguments
            elif example.kind == "dialogue_route":
                route = provider.propose_dialogue_route(
                    example.context,
                    example.available_actions,
                    example.suggested_action_ids,
                )
                chosen_intent = route.intent
                chosen_action = route.proposed_action
                chosen_arguments = {}
            else:
                turn = provider.propose_dialogue_turn(
                    example.context,
                    example.available_actions,
                    example.suggested_action_ids,
                )
                chosen_intent = turn.intent
                if turn.proposed_action is not None:
                    chosen_action = turn.proposed_action.action
                    chosen_arguments = turn.proposed_action.arguments
                else:
                    chosen_arguments = {}
        except InferenceError as exc:
            schema_valid = False
            inference_error = str(exc)

        action_ok = schema_valid and chosen_action == expected_action
        arguments_ok = action_ok and chosen_arguments == expected_arguments
        intent_ok = expected_intent is None or chosen_intent == expected_intent
        exact = arguments_ok and intent_ok
        safe_failure = chosen_action in SAFE_FAIL_CLOSED_ACTIONS
        unsafe_choice = chosen_action in FORBIDDEN_TRAINING_TARGETS
        risky_wrong_choice = (
            example.coverage.risk_category != "none"
            and chosen_action is not None
            and chosen_action != expected_action
            and not safe_failure
        )
        risky_wrong_arguments = (
            example.coverage.risk_category != "none"
            and chosen_action == expected_action
            and chosen_arguments != expected_arguments
        )
        report.results.append(
            DevelopmentCaseResult(
                example_id=example.id,
                split=example.split,
                schema_valid=schema_valid,
                action_ok=action_ok,
                arguments_ok=arguments_ok,
                exact=exact,
                chosen_action=chosen_action,
                chosen_arguments=chosen_arguments,
                argument_bucket=_argument_bucket(expected_arguments),
                trajectory_id=example.trajectory_id,
                risk_category=example.coverage.risk_category,
                safety_failure=bool(
                    unsafe_choice or risky_wrong_choice or risky_wrong_arguments
                ),
                inference_error=inference_error,
            )
        )
    return report
