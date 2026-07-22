from pathlib import Path

import pytest

from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.planning import (
    PlanningError,
    build_swap_plan,
    build_transfer_plan,
    make_mock_quote,
)
from agentic_wallet.schemas.common import Amount


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
WALLET = "0x1111111111111111111111111111111111111111"
RECIPIENT = "0x3333333333333333333333333333333333333333"
GAS_RESERVE = Amount(base_units="50000000000000", decimals=18)


@pytest.fixture
def harness():
    return MockReadOnlyHarness.from_fixture(FIXTURE)


def test_native_transfer_reserves_gas_and_has_no_calldata(harness):
    plan = build_transfer_plan(
        harness,
        asset_id="base:native",
        amount=Amount(base_units="100000000000000000", decimals=18),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )

    assert plan.to_address == RECIPIENT
    assert plan.recipient_address == RECIPIENT
    assert plan.value.base_units == "100000000000000000"
    assert plan.calldata == "0x"
    assert plan.plan_id.startswith("sha256:")


def test_erc20_transfer_resolves_contract_and_encodes_fixed_selector(harness):
    plan = build_transfer_plan(
        harness,
        asset_id="base:usdc",
        amount=Amount(base_units="25000000", decimals=6),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )

    assert plan.to_address.lower() == "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
    assert plan.recipient_address == RECIPIENT
    assert plan.value.base_units == "0"
    assert plan.calldata.startswith("0xa9059cbb")
    assert RECIPIENT[2:] in plan.calldata


@pytest.mark.parametrize(
    ("amount", "recipient", "message"),
    [
        (Amount(base_units="400000000", decimals=6), RECIPIENT, "insufficient"),
        (Amount(base_units="1", decimals=18), RECIPIENT, "decimals"),
        (Amount(base_units="1", decimals=6), "0xdeadbeef", "20-byte"),
    ],
)
def test_transfer_fails_closed_on_bad_amount_or_recipient(
    harness, amount, recipient, message
):
    with pytest.raises(PlanningError, match=message):
        build_transfer_plan(
            harness,
            asset_id="base:usdc",
            amount=amount,
            recipient=recipient,
            gas_reserve=GAS_RESERVE,
        )


def test_transfer_rejects_bad_mixed_case_eip55_checksum(harness):
    with pytest.raises(PlanningError, match="EIP-55"):
        build_transfer_plan(
            harness,
            asset_id="base:usdc",
            amount=Amount(base_units="1", decimals=6),
            recipient="0x5Aeda56215b167893e80B4fE645BA6d5Bab767De",
            gas_reserve=GAS_RESERVE,
        )


def _quote():
    return make_mock_quote(
        chain_id=8453,
        input_asset_id="base:usdc",
        output_asset_id="base:weth",
        amount_in=Amount(base_units="100000000", decimals=6),
        amount_out=Amount(base_units="30000000000000000", decimals=18),
        max_slippage_bps=50,
        issued_at_block=21000000,
        expires_at=2000,
    )


def test_swap_plan_is_bound_to_quote_and_pinned_router(harness):
    quote = _quote()
    plan = build_swap_plan(
        harness, quote=quote, now=1900, gas_reserve=GAS_RESERVE
    )

    assert plan.kind == "swap"
    assert plan.quote_id == quote.quote_id
    assert plan.quote_expires_at == 2000
    assert plan.recipient_address == WALLET
    assert plan.to_address == "0x2222222222222222222222222222222222222222"
    assert plan.calldata.startswith("0x5a19b9c3")


def test_swap_rejects_expired_or_tampered_quote(harness):
    quote = _quote()
    with pytest.raises(PlanningError, match="expired"):
        build_swap_plan(harness, quote=quote, now=2000, gas_reserve=GAS_RESERVE)

    tampered = quote.model_copy(
        update={"amount_out": Amount(base_units="1", decimals=18)}
    )
    with pytest.raises(PlanningError, match="integrity"):
        build_swap_plan(harness, quote=tampered, now=1900, gas_reserve=GAS_RESERVE)
