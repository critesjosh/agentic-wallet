from pathlib import Path

import pytest

from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.registry import RegistryError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"


def _harness() -> MockReadOnlyHarness:
    return MockReadOnlyHarness.from_fixture(FIXTURE)


def test_reads_native_balance():
    assert _harness().get_native_balance().base_units == "241000000000000000"


def test_reads_token_balance():
    assert _harness().get_token_balance("base:usdc").base_units == "300000000"


def test_unknown_asset_fails_closed():
    with pytest.raises(RegistryError):
        _harness().get_token_balance("base:unknown")


def test_absent_allowance_is_zero():
    amt = _harness().get_allowance("base:usdc", "base:some-other-spender")
    assert amt.base_units == "0"


def test_no_state_changing_surface():
    h = _harness()
    for attr in ("sign", "submit", "send", "transfer", "approve", "write_contract"):
        assert not hasattr(h, attr)
