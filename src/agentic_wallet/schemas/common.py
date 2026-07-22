"""Shared field types. Amounts are integer base units or decimal strings,
never floats (plan.md P-notes, sec 6). Untrusted content is isolated in a
typed wrapper that can never populate an actionable field (plan.md P5).
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

# Canonical IDs like "base:usdc", "base:native", "base:aave-v3". Resolved by the
# trusted registry, never invented by the model.
_ID_PATTERN = r"^[a-z0-9]+:[a-z0-9\-]+$"
AssetId = Annotated[str, StringConstraints(pattern=_ID_PATTERN)]
SpenderId = Annotated[str, StringConstraints(pattern=_ID_PATTERN)]
HexData = Annotated[str, StringConstraints(pattern=r"^0x[0-9a-fA-F]*$")]
EvmAddress = Annotated[str, StringConstraints(pattern=r"^0x[0-9a-fA-F]{40}$")]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields (no silent data smuggling)."""

    model_config = ConfigDict(extra="forbid")


class Amount(StrictModel):
    """A token amount as integer base units. Never a float.

    ``base_units`` is the on-chain integer (wei-like) as a decimal string;
    ``decimals`` records the token scale for display only.
    """

    base_units: str
    decimals: int

    @field_validator("base_units")
    @classmethod
    def _integer_string(cls, v: str) -> str:
        if not re.fullmatch(r"\d+", v):
            raise ValueError("base_units must be a non-negative integer string")
        return v

    @field_validator("decimals")
    @classmethod
    def _decimals_range(cls, v: int) -> int:
        if not 0 <= v <= 36:
            raise ValueError("decimals out of range")
        return v


class UsdValue(StrictModel):
    """A USD-denominated value (e.g. "swap $300"). Decimal string, never float."""

    usd: str

    @field_validator("usd")
    @classmethod
    def _decimal_string(cls, v: str) -> str:
        if not re.fullmatch(r"\d+(\.\d+)?", v):
            raise ValueError("usd must be a decimal string")
        return v


class UntrustedData(StrictModel):
    """Retrieved content that is NOT trusted (token metadata, protocol docs).

    Untrusted content lives only inside this typed wrapper. It must never be
    merged into instruction context and must never populate an actionable field
    such as a recipient, address, amount, or chain. See plan.md P5.
    """

    source: str
    content: str
