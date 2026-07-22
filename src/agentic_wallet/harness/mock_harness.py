"""Fixture-backed, network-free, watch-only harness.

No key custody, no signing, no submission. Exposes only read tools. Any
state-changing method is intentionally absent, so the first product ships with
zero key custody (plan.md: read-only needs no wallet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from ..registry import BASE_REGISTRY, Registry
from ..schemas.common import Amount
from ..schemas.portfolio import Portfolio


class HarnessError(RuntimeError):
    pass


class MockReadOnlyHarness:
    def __init__(self, portfolio: Portfolio, registry: Registry = BASE_REGISTRY) -> None:
        self.portfolio = portfolio
        self.registry = registry

    @classmethod
    def from_fixture(
        cls, path: Union[str, Path], registry: Registry = BASE_REGISTRY
    ) -> "MockReadOnlyHarness":
        data = json.loads(Path(path).read_text())
        return cls(Portfolio.model_validate(data), registry)

    # --- read-only tools (available in COLLECTING_STATE) ---

    def get_native_balance(self) -> Amount:
        return self.portfolio.native_balance

    def get_token_balance(self, asset_id: str) -> Amount:
        self.registry.resolve(asset_id)  # fail-closed on unknown canonical id
        for tb in self.portfolio.token_balances:
            if tb.asset_id == asset_id:
                return tb.amount
        raise HarnessError(f"no balance for {asset_id}")

    def get_allowance(self, asset_id: str, spender_id: str) -> Amount:
        entry = self.registry.resolve(asset_id)
        for al in self.portfolio.allowances:
            if al.asset_id == asset_id and al.spender_id == spender_id:
                return al.amount
        return Amount(base_units="0", decimals=entry.decimals)

    def get_portfolio(self) -> Portfolio:
        return self.portfolio
