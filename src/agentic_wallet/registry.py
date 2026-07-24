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
    is_native: bool = False


class RegistryError(KeyError):
    """Raised when a canonical ID cannot be resolved (fail-closed)."""


class Registry:
    def __init__(self, entries: list[RegistryEntry]) -> None:
        if len({entry.asset_id for entry in entries}) != len(entries):
            raise ValueError("registry contains duplicate canonical ids")
        self._by_id: dict[str, RegistryEntry] = {e.asset_id: e for e in entries}
        native_by_chain: dict[int, RegistryEntry] = {}
        for entry in entries:
            if not entry.is_native:
                continue
            if entry.chain_id in native_by_chain:
                raise ValueError("registry contains multiple native assets for one chain")
            native_by_chain[entry.chain_id] = entry
        self._native_by_chain = native_by_chain

    def resolve(self, asset_id: str) -> RegistryEntry:
        try:
            return self._by_id[asset_id]
        except KeyError as exc:
            raise RegistryError(f"unknown canonical id: {asset_id}") from exc

    def entries(self) -> list[RegistryEntry]:
        return list(self._by_id.values())

    def native_asset(self, chain_id: int) -> RegistryEntry:
        """Return the code-pinned native asset for ``chain_id`` or fail closed."""

        try:
            return self._native_by_chain[chain_id]
        except KeyError as exc:
            raise RegistryError(f"no native asset configured for chain: {chain_id}") from exc

    def is_native(self, asset_id: str) -> bool:
        """Whether a canonical asset is the pinned native asset for its chain."""

        return self.resolve(asset_id).is_native

    def version_digest(self) -> str:
        payload = sorted(
            f"{e.asset_id}|{e.chain_id}|{e.address.lower()}|{e.symbol}|{e.decimals}|{e.is_native}"
            for e in self._by_id.values()
        )
        return canonical_digest(payload)


# Pinned Base assets for the fixture-backed read-only demo and the Phase 8
# native-transfer POC. Native is a sentinel, not an address. The router is
# intentionally a non-production mock and must never be used for a live
# transaction. Base Sepolia carries only its native asset because the POC signs
# native transfers and nothing else; its ID must match the code-owned
# ``chain_metadata`` entry for that chain.
BASE_REGISTRY = Registry(
    [
        RegistryEntry("base:native", 8453, "native", "ETH", 18, is_native=True),
        RegistryEntry(
            "base:sepolia-native", 84532, "native", "ETH", 18, is_native=True
        ),
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
