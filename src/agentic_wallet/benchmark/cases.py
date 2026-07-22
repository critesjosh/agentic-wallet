"""A single benchmark case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

HARD_ZERO_CATEGORIES = frozenset(
    {
        "arbitrary-address-invention",
        "insufficient-funds",
        "policy-bypass",
        "signing-boundary-violation",
        "unexpected-transfer",
        "unlimited-approval",
        "wrong-asset",
        "wrong-chain",
        "wrong-recipient",
    }
)

# Historical naming: both families are frozen evaluation data. ``train`` means
# the familiar synthetic registry universe; it is never eligible SFT text.
BENCHMARK_DATASET_ROLE = "frozen-evaluation-only"


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    family: str  # familiar vs held-out registry universe; both are eval-only
    scenario_id: str
    user_request: str
    workflow_state: str
    available_actions: list[str]
    expected_action: str
    expected_arguments: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    forbidden_actions: list[str] = field(default_factory=list)
    # If choosing a forbidden or wrong action here is a critical safety failure,
    # set the category (plan.md P6 hard-zero blockers). None means non-critical.
    hard_zero_category: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in {"train", "eval"}:
            raise ValueError(f"unknown benchmark family: {self.family}")
        if (
            self.hard_zero_category is not None
            and self.hard_zero_category not in HARD_ZERO_CATEGORIES
        ):
            raise ValueError(
                f"unknown hard-zero category: {self.hard_zero_category}"
            )
