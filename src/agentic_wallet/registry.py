"""Trusted registry: canonical IDs -> addresses.

This is the PRIMARY root of trust for address resolution (plan.md P4). Entries
are pinned in code here. A production registry must ship signed, versioned,
fail-closed updates with provenance; ``version_digest`` pins the current set so
a caller can detect drift. Simulation and policy checks are defense-in-depth
and do not reduce this integrity requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .digest import canonical_digest


@dataclass(frozen=True)
class RegistryEntry:
    asset_id: str
    chain_id: int
    address: str
    symbol: str
    decimals: int


class RegistryError(KeyError):
    """Raised when a canonical ID cannot be resolved (fail-closed)."""


class Registry:
    def __init__(self, entries: list[RegistryEntry]) -> None:
        if len({entry.asset_id for entry in entries}) != len(entries):
            raise ValueError("registry contains duplicate canonical ids")
        self._by_id: dict[str, RegistryEntry] = {e.asset_id: e for e in entries}

    def resolve(self, asset_id: str) -> RegistryEntry:
        try:
            return self._by_id[asset_id]
        except KeyError as exc:
            raise RegistryError(f"unknown canonical id: {asset_id}") from exc

    def entries(self) -> list[RegistryEntry]:
        return list(self._by_id.values())

    def version_digest(self) -> str:
        payload = sorted(
            f"{e.asset_id}|{e.chain_id}|{e.address.lower()}|{e.symbol}|{e.decimals}"
            for e in self._by_id.values()
        )
        return canonical_digest(payload)


# Pinned Base mainnet assets for the fixture-backed read-only demo. Native is a
# sentinel, not an address. The router is intentionally a non-production mock
# and must never be used for a live transaction.
BASE_REGISTRY = Registry(
    [
        RegistryEntry("base:native", 8453, "native", "ETH", 18),
        RegistryEntry("base:usdc", 8453, "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", "USDC", 6),
        RegistryEntry("base:weth", 8453, "0x4200000000000000000000000000000000000006", "WETH", 18),
        RegistryEntry(
            "base:fixture-swap-router",
            8453,
            "0x2222222222222222222222222222222222222222",
            "FIXTURE_SWAP_ROUTER_DO_NOT_USE",
            0,
        ),
    ]
)
