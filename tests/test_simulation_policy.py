from pathlib import Path

from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.planning import build_swap_plan, build_transfer_plan, make_mock_quote
from agentic_wallet.policy_engine import WalletPolicy, evaluate_policy
from agentic_wallet.registry import BASE_REGISTRY
from agentic_wallet.schemas.common import Amount
from agentic_wallet.schemas.simulation_result import BalanceChange
from agentic_wallet.simulation import expected_balance_changes, simulate_plan


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
WALLET = "0x1111111111111111111111111111111111111111"
RECIPIENT = "0x3333333333333333333333333333333333333333"
GAS_RESERVE = Amount(base_units="50000000000000", decimals=18)
GAS_FEE = Amount(base_units="21000000000000", decimals=18)


def _harness():
    return MockReadOnlyHarness.from_fixture(FIXTURE)


def _transfer():
    return build_transfer_plan(
        _harness(),
        asset_id="base:usdc",
        amount=Amount(base_units="25000000", decimals=6),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )


def _swap():
    quote = make_mock_quote(
        chain_id=8453,
        input_asset_id="base:usdc",
        output_asset_id="base:weth",
        amount_in=Amount(base_units="100000000", decimals=6),
        amount_out=Amount(base_units="30000000000000000", decimals=18),
        max_slippage_bps=50,
        issued_at_block=21000000,
        expires_at=2000,
    )
    return build_swap_plan(
        _harness(), quote=quote, now=1900, gas_reserve=GAS_RESERVE
    )


def _policy(max_slippage_bps=100):
    return WalletPolicy(
        chain_id=8453,
        wallet_address=WALLET,
        max_slippage_bps=max_slippage_bps,
    )


def test_expected_diff_uses_signed_integer_strings_only():
    changes = expected_balance_changes(_transfer(), gas_fee=GAS_FEE)
    assert {(c.asset_id, c.delta_base_units) for c in changes} == {
        ("base:native", "-21000000000000"),
        ("base:usdc", "-25000000"),
    }


def test_matching_simulation_and_policy_allow_unsigned_plan():
    plan = _swap()
    simulation = simulate_plan(
        plan, block=21000001, gas_used=180000, gas_fee=GAS_FEE
    )
    result = evaluate_policy(
        plan,
        simulation,
        policy=_policy(),
        registry=BASE_REGISTRY,
        now=1900,
    )

    assert simulation.mismatch is False
    assert result.allowed is True
    assert result.violations == []


def test_unexpected_transfer_is_a_policy_blocker():
    plan = _swap()
    observed = expected_balance_changes(plan, gas_fee=GAS_FEE) + [
        BalanceChange(asset_id="base:usdc", delta_base_units="-1")
    ]
    # Duplicate asset deltas are malformed provider output, so use an asset that
    # is valid but absent from the expected diff.
    observed[-1] = BalanceChange(asset_id="base:native", delta_base_units="-1")
    # Native is also expected; replace the full normalized result with one
    # unexpected canonical asset to model a malicious extra transfer.
    observed = expected_balance_changes(plan, gas_fee=GAS_FEE) + [
        BalanceChange(asset_id="base:usdc-extra", delta_base_units="-1")
    ]
    simulation = simulate_plan(
        plan,
        block=21000001,
        gas_used=180000,
        gas_fee=GAS_FEE,
        observed_changes=observed,
    )
    result = evaluate_policy(
        plan,
        simulation,
        policy=_policy(),
        registry=BASE_REGISTRY,
        now=1900,
    )

    assert simulation.mismatch is True
    assert result.allowed is False
    assert "unexpected-transfer" in result.violations


def test_wrong_recipient_and_excess_slippage_are_blocked():
    plan = _swap().model_copy(update={"recipient_address": RECIPIENT})
    simulation = simulate_plan(
        plan, block=21000001, gas_used=180000, gas_fee=GAS_FEE
    )
    result = evaluate_policy(
        plan,
        simulation,
        policy=_policy(max_slippage_bps=25),
        registry=BASE_REGISTRY,
        now=1900,
    )

    assert result.allowed is False
    assert "wrong-swap-recipient" in result.violations
    assert "slippage-limit-exceeded" in result.violations


def test_erc20_calldata_cannot_disagree_with_displayed_recipient():
    original = _transfer()
    attacker = "0x4444444444444444444444444444444444444444"
    tampered_calldata = original.calldata.replace(RECIPIENT[2:], attacker[2:])
    plan = original.model_copy(update={"calldata": tampered_calldata})
    simulation = simulate_plan(
        plan, block=21000001, gas_used=65000, gas_fee=GAS_FEE
    )
    result = evaluate_policy(
        plan,
        simulation,
        policy=_policy(),
        registry=BASE_REGISTRY,
        now=1900,
    )

    assert result.allowed is False
    assert "calldata-recipient-or-amount-mismatch" in result.violations


def test_unknown_canonical_id_is_a_policy_violation_not_an_exception():
    plan = _transfer().model_copy(update={"asset_id": "base:unknown"})
    simulation = simulate_plan(
        plan, block=21000001, gas_used=65000, gas_fee=GAS_FEE
    )
    result = evaluate_policy(
        plan,
        simulation,
        policy=_policy(),
        registry=BASE_REGISTRY,
        now=1900,
    )
    assert not result.allowed
    assert "unknown-canonical-id:base:unknown" in result.violations
