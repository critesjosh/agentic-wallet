"""Account identity reads and the deterministic chat transfer command.

The transfer command is the only chat path that can reach the signing
boundary, so its parsing is covered for every representation that can produce,
or fail to produce, an exact candidate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_wallet.chain_metadata import (
    InvalidAddressError,
    UnknownChainError,
    explorer_address_url,
    get_chain_metadata,
)
from agentic_wallet.harness import HarnessError, MockReadOnlyHarness
from agentic_wallet.registry import BASE_REGISTRY, RegistryError
from agentic_wallet.web.chat import (
    DemoChatAgent,
    parse_native_transfer_command,
)
from agentic_wallet.web.transactions import (
    SIGNABLE_CHAIN_IDS,
    require_signable_chain_id,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"

BASE = 8453
SEPOLIA = 84532
RECIPIENT = "0x2222222222222222222222222222222222222222"
SIGNER = "0x3333333333333333333333333333333333333333"


def _agent(**kwargs) -> DemoChatAgent:
    return DemoChatAgent(MockReadOnlyHarness.from_fixture(FIXTURE), **kwargs)


def _parse(message: str, *, live_chain_id: int = BASE):
    return parse_native_transfer_command(
        message, live_chain_id=live_chain_id, native_decimals=18
    )


# --- account identity ----------------------------------------------------


def test_account_query_returns_an_account_not_the_registry():
    response = _agent().respond("s1", "what is my address?")

    assert response["data"]["type"] == "account"


def test_fixture_account_is_never_presented_as_a_real_account():
    """A placeholder address must not look fundable."""

    response = _agent().respond("s1", "who am i")
    account = response["data"]["account"]

    assert account["source"] == "fixture"
    assert account["watch_only"] is True
    # No explorer link: a working link beside a placeholder invites funding it.
    assert account["explorer_url"] is None
    assert "No real account is loaded" in response["reply"]
    assert "do not send funds to it" in response["reply"]


def test_provisioned_signer_account_is_reported_as_the_real_account():
    agent = _agent(live_chain_id=SEPOLIA)
    agent.signer_address = SIGNER

    response = agent.respond("s1", "what is my address?")
    account = response["data"]["account"]

    assert account["address"] == SIGNER
    assert account["source"] == "signer"
    assert account["watch_only"] is False
    # The signer's chain is the configured signing chain, not the fixture's.
    assert account["chain_id"] == SEPOLIA
    assert account["chain_name"] == "Base Sepolia"
    assert account["explorer_url"] == f"https://sepolia.basescan.org/address/{SIGNER}"
    assert "signer account" in response["reply"]


def test_account_view_never_carries_key_material():
    agent = _agent()
    agent.signer_address = SIGNER
    serialized = repr(agent.respond("s1", "who am i")["data"])

    for secret in ("private_key", "mnemonic", "seed", "passphrase"):
        assert secret not in serialized


def test_explicit_registry_request_still_reaches_the_registry():
    """The registry names itself, so it must win over the address keyword."""

    response = _agent().respond("s1", "show the registry addresses")

    assert response["data"]["type"] == "registry"


def test_explorer_address_url_fails_closed_for_untrusted_input():
    with pytest.raises(UnknownChainError):
        explorer_address_url(999, RECIPIENT)
    with pytest.raises(InvalidAddressError):
        explorer_address_url(BASE, "not-an-address")
    with pytest.raises(InvalidAddressError):
        # A transaction hash is the wrong width and must not build a link.
        explorer_address_url(BASE, "0x" + "a" * 64)


# --- native balance binding ---------------------------------------------


def test_native_balance_is_bound_to_the_snapshot_chain():
    """A second native asset must not read the first chain's balance."""

    agent = _agent()
    assert BASE_REGISTRY.resolve("base:sepolia-native").is_native

    # The fixture snapshot covers Base, so the Sepolia native asset is unproven
    # rather than silently answered with the Base balance.
    with pytest.raises(HarnessError, match="chain 8453"):
        agent._balance_amount("base:sepolia-native")

    assert agent._balance_amount("base:native").base_units == "241000000000000000"


def test_native_balance_rejects_an_unknown_asset():
    with pytest.raises(RegistryError):
        _agent()._balance_amount("base:not-real")


def test_registry_sepolia_native_matches_chain_metadata():
    entry = BASE_REGISTRY.native_asset(SEPOLIA)

    assert entry.asset_id == get_chain_metadata(SEPOLIA).native_asset_id
    assert entry.decimals == 18


# --- transfer command parsing -------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (f"send 1 wei to {RECIPIENT} on base", "1"),
        (f"send 0.01 eth to {RECIPIENT} on base", "10000000000000000"),
        (f"send 1 eth to {RECIPIENT} on base", "1000000000000000000"),
        (f"send 1.5 ETH to {RECIPIENT} on Base", "1500000000000000000"),
        # One wei expressed as the smallest representable decimal.
        (f"send 0.000000000000000001 eth to {RECIPIENT} on base", "1"),
        # Leading zeros normalize rather than changing the value.
        (f"send 007 wei to {RECIPIENT} on base", "7"),
    ],
)
def test_amounts_convert_exactly(message: str, expected: str):
    command = _parse(message)

    assert command.error is None
    assert command.candidate == {
        "chain_id": BASE,
        "amount_base_units": expected,
        "recipient": RECIPIENT,
    }


def test_sub_wei_precision_is_rejected():
    command = _parse(f"send 0.0000000000000000001 eth to {RECIPIENT} on base")

    assert command.candidate is None
    assert "precision" in command.error


def test_fractional_wei_is_rejected():
    command = _parse(f"send 1.5 wei to {RECIPIENT} on base")

    assert command.candidate is None
    assert "whole numbers" in command.error


def test_zero_amount_is_rejected():
    for message in (
        f"send 0 wei to {RECIPIENT} on base",
        f"send 0.0 eth to {RECIPIENT} on base",
    ):
        command = _parse(message)
        assert command.candidate is None
        assert "greater than zero" in command.error


def test_command_for_another_chain_fails_closed():
    """Wrong-chain routing is a predeclared hazard, so it must never bind."""

    command = _parse(f"send 1 eth to {RECIPIENT} on base sepolia", live_chain_id=BASE)

    assert command.candidate is None
    assert "Base Sepolia" in command.error


def test_command_binds_the_configured_testnet_chain():
    command = _parse(
        f"send 1 eth to {RECIPIENT} on base sepolia", live_chain_id=SEPOLIA
    )

    assert command.candidate["chain_id"] == SEPOLIA


@pytest.mark.parametrize(
    "message",
    [
        "send 1 eth to somebody on base",
        f"send 1 eth to {RECIPIENT}",
        f"send 1 usdc to {RECIPIENT} on base",
        f"send 1 eth to {RECIPIENT} on ethereum",
        f"send 1 eth to {RECIPIENT[:-1]} on base",
        f"please send 1 eth to {RECIPIENT} on base",
        f"send 1 eth to {RECIPIENT} on base and approve it",
        # An amount alone is not a command.
        "send eth on base",
    ],
)
def test_non_commands_produce_no_candidate(message: str):
    command = _parse(message)

    assert command.candidate is None


def test_transfer_command_is_ignored_while_transactions_are_disabled():
    agent = _agent()
    assert agent.transfer_requests_enabled is False

    response = agent.respond("s1", f"send 1 eth to {RECIPIENT} on base")

    assert response["transaction_request"] is None


def test_enabled_agent_returns_a_review_request_without_approving():
    agent = _agent(transfer_requests_enabled=True)

    response = agent.respond("s1", f"send 0.01 eth to {RECIPIENT} on base")

    assert response["transaction_request"] == {
        "chain_id": BASE,
        "amount_base_units": "10000000000000000",
        "recipient": RECIPIENT,
    }
    assert "not approved" in response["reply"]


# --- offered action derivation ------------------------------------------


def test_model_actions_follow_the_flag_set_after_construction():
    """The application sets this flag once it knows a controller exists."""

    agent = _agent()
    assert "request_native_transfer_review" not in agent.model_actions

    agent.transfer_requests_enabled = True

    assert "request_native_transfer_review" in agent.model_actions
    assert "get_transaction_status" in agent.model_actions


def test_agent_rejects_a_chain_it_cannot_prove():
    with pytest.raises((ValueError, KeyError)):
        _agent(live_chain_id=999)


# --- signing allowlist ---------------------------------------------------


def test_signing_allowlist_is_narrower_than_explorer_metadata():
    assert SIGNABLE_CHAIN_IDS == {BASE, SEPOLIA}
    # Ethereum mainnet has explorer metadata but must not be signable.
    get_chain_metadata(1)
    with pytest.raises(ValueError):
        require_signable_chain_id(1)


@pytest.mark.parametrize("chain_id", [0, -1, 999, True, "8453", None])
def test_signable_chain_rejects_untrusted_values(chain_id):
    with pytest.raises(ValueError):
        require_signable_chain_id(chain_id)
