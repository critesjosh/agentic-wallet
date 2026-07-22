"""An unsigned transaction plan, built entirely by deterministic code."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .common import Amount, AssetId, EvmAddress, HexData, StrictModel


class AssetDelta(StrictModel):
    asset_id: AssetId
    amount: Amount


class TransactionPlan(StrictModel):
    plan_id: str
    chain_id: int
    kind: Literal["transfer", "swap", "approval"]
    from_address: EvmAddress
    to_address: EvmAddress
    recipient_address: Optional[EvmAddress] = None
    asset_id: AssetId
    value: Amount  # native value attached to the call
    calldata: HexData = "0x"
    expected_incoming: list[AssetDelta] = Field(default_factory=list)
    expected_outgoing: list[AssetDelta] = Field(default_factory=list)
    gas_reserve: Optional[Amount] = None
    max_slippage_bps: Optional[int] = None
    quote_id: Optional[str] = None
    quote_expires_at: Optional[int] = None
