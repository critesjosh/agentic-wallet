"""Normalized, read-only portfolio snapshot for a watch-only address."""

from __future__ import annotations

from pydantic import Field

from .common import Amount, AssetId, SpenderId, StrictModel


class TokenBalance(StrictModel):
    asset_id: AssetId
    amount: Amount


class Allowance(StrictModel):
    asset_id: AssetId
    spender_id: SpenderId
    amount: Amount


class Portfolio(StrictModel):
    chain_id: int
    address: str
    native_balance: Amount
    token_balances: list[TokenBalance] = Field(default_factory=list)
    allowances: list[Allowance] = Field(default_factory=list)
    as_of_block: int
    stale: bool = False
