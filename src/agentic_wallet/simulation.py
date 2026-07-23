"""Deterministic fixture-backed transaction simulation comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .schemas.common import Amount
from .schemas.signing import Eip1559Transaction
from .schemas.simulation_result import BalanceChange, SimulationResult
from .schemas.transaction_plan import TransactionPlan

if TYPE_CHECKING:
    from .registry import Registry


class SimulationError(RuntimeError):
    pass


def expected_balance_changes(
    plan: TransactionPlan,
    *,
    gas_fee: Amount,
    native_asset_id: str = "base:native",
    native_asset_decimals: int = 18,
) -> list[BalanceChange]:
    """Calculate the exact wallet delta expected from plan plus gas."""

    if gas_fee.decimals != native_asset_decimals:
        raise SimulationError("gas fee must use native-token decimals")
    deltas: dict[str, int] = defaultdict(int)
    for outgoing in plan.expected_outgoing:
        deltas[outgoing.asset_id] -= int(outgoing.amount.base_units)
    for incoming in plan.expected_incoming:
        deltas[incoming.asset_id] += int(incoming.amount.base_units)
    deltas[native_asset_id] -= int(gas_fee.base_units)
    return [
        BalanceChange(asset_id=asset_id, delta_base_units=str(delta))
        for asset_id, delta in sorted(deltas.items())
        if delta != 0
    ]


def simulate_plan(
    plan: TransactionPlan,
    *,
    block: int,
    gas_used: int,
    gas_fee: Amount,
    observed_changes: list[BalanceChange] | None = None,
    success: bool = True,
    logs_summary: str = "fixture simulation",
    signing_transaction: Eip1559Transaction | None = None,
    native_asset_id: str = "base:native",
    native_asset_decimals: int = 18,
    registry: "Registry | None" = None,
) -> SimulationResult:
    """Compare a provider's normalized diff with the deterministic expectation."""

    if block <= 0 or gas_used < 0:
        raise SimulationError("invalid simulation block or gas usage")
    if registry is not None:
        native_asset = registry.native_asset(plan.chain_id)
        native_asset_id = native_asset.asset_id
        native_asset_decimals = native_asset.decimals
    expected = expected_balance_changes(
        plan,
        gas_fee=gas_fee,
        native_asset_id=native_asset_id,
        native_asset_decimals=native_asset_decimals,
    )
    observed = observed_changes if observed_changes is not None else expected
    expected_map = {item.asset_id: int(item.delta_base_units) for item in expected}
    observed_map = {item.asset_id: int(item.delta_base_units) for item in observed}
    if len(observed_map) != len(observed):
        raise SimulationError("simulation returned duplicate asset deltas")

    unexpected = [
        item for item in observed if item.asset_id not in expected_map
    ]
    mismatch = (not success) or observed_map != expected_map or bool(unexpected)
    return SimulationResult(
        plan_id=plan.plan_id,
        success=success,
        block=block,
        gas_used=gas_used,
        balance_changes=observed,
        unexpected_transfers=unexpected,
        mismatch=mismatch,
        logs_summary=logs_summary,
        transaction_digest=(
            signing_transaction.digest() if signing_transaction is not None else None
        ),
    )
