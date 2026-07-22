"""Normalized intent extracted from a user request."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import Field

from .common import Amount, AssetId, StrictModel, UntrustedData, UsdValue


class IntentConstraints(StrictModel):
    max_slippage_bps: Optional[int] = None
    preserve_gas_reserve: bool = False
    gas_reserve: Optional[Amount] = None
    deadline_unix: Optional[int] = None


class Intent(StrictModel):
    """Actionable fields are typed and set only from trusted parsing, never
    from ``untrusted_context``.
    """

    user_request: str
    normalized_action: Optional[str] = None  # "swap" | "transfer" | "read_portfolio" | ...
    chain_id: Optional[int] = None
    input_asset: Optional[AssetId] = None
    output_asset: Optional[AssetId] = None
    amount: Optional[Union[Amount, UsdValue]] = None
    constraints: IntentConstraints = Field(default_factory=IntentConstraints)
    missing_fields: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    untrusted_context: list[UntrustedData] = Field(default_factory=list)
