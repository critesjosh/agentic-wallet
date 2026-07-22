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

# Historical naming: both families are immutable regression data. ``train``
# means the familiar synthetic registry universe, never SFT eligibility. After
# v2 was designed from observed failures, this suite became a development set;
# release claims require a separately authored sealed evaluation suite.
BENCHMARK_DATASET_ROLE = "development-regression-only"


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
    trajectory_id: Optional[str] = None
    turn_index: Optional[int] = None
    # If choosing a forbidden or wrong action here is a critical safety failure,
    # set the category (plan.md P6 hard-zero blockers). None means non-critical.
    hard_zero_category: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in {"train", "eval", "sealed"}:
            raise ValueError(f"unknown benchmark family: {self.family}")
        if (self.trajectory_id is None) != (self.turn_index is None):
            raise ValueError("trajectory_id and turn_index must be provided together")
        if self.turn_index is not None and self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        if self.family == "sealed" and not self.context.get("canonical_asset_ids"):
            raise ValueError("sealed cases require canonical_asset_ids in context")
        if (
            self.hard_zero_category is not None
            and self.hard_zero_category not in HARD_ZERO_CATEGORIES
        ):
            raise ValueError(
                f"unknown hard-zero category: {self.hard_zero_category}"
            )
