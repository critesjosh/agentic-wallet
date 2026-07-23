"""Result of simulating a transaction plan: the before/after state diff."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from .common import AssetId, StrictModel


class BalanceChange(StrictModel):
    asset_id: AssetId
    delta_base_units: str  # signed integer string, may start with '-'

    @field_validator("delta_base_units")
    @classmethod
    def _signed_integer(cls, v: str) -> str:
        if not re.fullmatch(r"-?\d+", v):
            raise ValueError("delta_base_units must be a signed integer string")
        return v


class SimulationResult(StrictModel):
    plan_id: str
    success: bool
    block: int
    gas_used: int
    balance_changes: list[BalanceChange] = Field(default_factory=list)
    unexpected_transfers: list[BalanceChange] = Field(default_factory=list)
    mismatch: bool = False
    logs_summary: str = ""
    # Present when a Phase 8 simulation was performed against an exact EIP-1559
    # preimage. Older unsigned-only simulations deliberately omit it.
    transaction_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
