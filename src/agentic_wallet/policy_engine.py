"""Deterministic policy checks over plans and normalized simulations."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import Registry, RegistryError
from .schemas.policy import PolicyResult
from .schemas.simulation_result import SimulationResult
from .schemas.transaction_plan import TransactionPlan

_ERC20_TRANSFER_SELECTOR = "a9059cbb"


def _expected_erc20_transfer_calldata(recipient: str, amount: int) -> str:
    return (
        "0x"
        + _ERC20_TRANSFER_SELECTOR
        + recipient[2:].lower().rjust(64, "0")
        + f"{amount:064x}"
    )


@dataclass(frozen=True)
class WalletPolicy:
    chain_id: int
    wallet_address: str
    max_slippage_bps: int = 100
    approved_router_ids: tuple[str, ...] = ("base:fixture-swap-router",)


def evaluate_policy(
    plan: TransactionPlan,
    simulation: SimulationResult,
    *,
    policy: WalletPolicy,
    registry: Registry,
    now: int,
) -> PolicyResult:
    """Fail closed on every violation; the model cannot override this result."""

    violations: list[str] = []
    if plan.chain_id != policy.chain_id:
        violations.append("wrong-chain")
    if plan.from_address.lower() != policy.wallet_address.lower():
        violations.append("wrong-sender")
    if simulation.plan_id != plan.plan_id:
        violations.append("simulation-plan-mismatch")
    if not simulation.success:
        violations.append("simulation-failed")
    if simulation.mismatch:
        violations.append("simulation-state-mismatch")
    if simulation.unexpected_transfers:
        violations.append("unexpected-transfer")
    if plan.gas_reserve is None or int(plan.gas_reserve.base_units) <= 0:
        violations.append("missing-gas-reserve")

    resolved_assets = {}
    referenced_asset_ids = {
        plan.asset_id,
        *(delta.asset_id for delta in plan.expected_incoming),
        *(delta.asset_id for delta in plan.expected_outgoing),
    }
    for asset_id in referenced_asset_ids:
        try:
            resolved_assets[asset_id] = registry.resolve(asset_id)
        except RegistryError:
            violations.append(f"unknown-canonical-id:{asset_id}")

    if plan.kind == "transfer":
        asset = resolved_assets.get(plan.asset_id)
        if asset is not None and asset.chain_id != plan.chain_id:
            violations.append("asset-chain-mismatch")
        expected_target = (
            plan.recipient_address
            if plan.asset_id == "base:native"
            else asset.address if asset is not None else None
        )
        if expected_target is None or plan.to_address.lower() != expected_target.lower():
            violations.append("wrong-recipient-or-token-contract")
        if plan.recipient_address is None:
            violations.append("missing-recipient")
        if len(plan.expected_outgoing) != 1 or plan.expected_incoming:
            violations.append("invalid-transfer-deltas")
        else:
            outgoing = plan.expected_outgoing[0]
            if outgoing.asset_id != plan.asset_id:
                violations.append("transfer-asset-mismatch")
            elif plan.asset_id == "base:native":
                if plan.calldata != "0x" or plan.value != outgoing.amount:
                    violations.append("invalid-native-transfer-encoding")
            elif plan.recipient_address is not None:
                expected_calldata = _expected_erc20_transfer_calldata(
                    plan.recipient_address, int(outgoing.amount.base_units)
                )
                if plan.calldata.lower() != expected_calldata:
                    violations.append("calldata-recipient-or-amount-mismatch")
                if int(plan.value.base_units) != 0:
                    violations.append("unexpected-native-value")
    elif plan.kind == "swap":
        matching_routers = []
        for router_id in policy.approved_router_ids:
            try:
                router = registry.resolve(router_id)
            except RegistryError:
                violations.append(f"unknown-canonical-id:{router_id}")
                continue
            if router.chain_id == policy.chain_id:
                matching_routers.append(router)
        if plan.to_address.lower() not in {
            entry.address.lower() for entry in matching_routers
        }:
            violations.append("unapproved-router")
        if plan.recipient_address is None or (
            plan.recipient_address.lower() != policy.wallet_address.lower()
        ):
            violations.append("wrong-swap-recipient")
        if plan.max_slippage_bps is None:
            violations.append("missing-slippage-limit")
        elif plan.max_slippage_bps > policy.max_slippage_bps:
            violations.append("slippage-limit-exceeded")
        if plan.quote_expires_at is None or now >= plan.quote_expires_at:
            violations.append("quote-expired")
        if not plan.quote_id:
            violations.append("missing-quote")
    elif plan.kind == "approval":
        violations.append("approvals-not-enabled")

    return PolicyResult(allowed=not violations, violations=violations)
