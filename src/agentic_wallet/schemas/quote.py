"""A pinned quote returned by a narrow, typed swap provider."""

from __future__ import annotations

from pydantic import field_validator

from .common import Amount, AssetId, SpenderId, StrictModel


class SwapQuote(StrictModel):
    quote_id: str
    chain_id: int
    input_asset_id: AssetId
    output_asset_id: AssetId
    amount_in: Amount
    amount_out: Amount
    router_id: SpenderId
    max_slippage_bps: int
    issued_at_block: int
    expires_at: int

    @field_validator("chain_id", "issued_at_block", "expires_at")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("max_slippage_bps")
    @classmethod
    def _slippage_range(cls, value: int) -> int:
        if not 0 <= value <= 10_000:
            raise ValueError("max_slippage_bps must be between 0 and 10000")
        return value
