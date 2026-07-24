"""Account creation through the private signer.

The key is generated inside the signer process and must never cross the MCP
boundary, be logged, or replace an account that could still hold funds.
"""

from __future__ import annotations

import anyio
import pytest
from eth_account import Account

from agentic_wallet.signer.key_store import (
    KEYRING_ACCOUNT,
    KEYRING_SERVICE,
    KeyAlreadyExistsError,
    KeyStoreError,
    OSKeyringKeyStore,
)
from agentic_wallet.signer.service import SignerDenied, SignerService


class FakeSecureBackend:
    """Stands in for an approved OS keyring backend."""

    def __init__(self, stored: str | None = None) -> None:
        self.stored = stored
        self.writes = 0

    def get_password(self, service: str, account: str) -> str | None:
        assert (service, account) == (KEYRING_SERVICE, KEYRING_ACCOUNT)
        return self.stored

    def set_password(self, service: str, account: str, password: str) -> None:
        assert (service, account) == (KEYRING_SERVICE, KEYRING_ACCOUNT)
        self.stored = password
        self.writes += 1


class UnavailableBackend:
    def get_password(self, service: str, account: str) -> str | None:
        raise OSError("keyring is locked")

    def set_password(self, service: str, account: str, password: str) -> None:
        raise OSError("keyring is locked")


def _store(backend: object) -> OSKeyringKeyStore:
    store = OSKeyringKeyStore(backend=backend)
    # The approved-backend allowlist is enforced elsewhere; bypass only that
    # check so these tests exercise creation rather than backend detection.
    store._secure_backend = lambda: backend  # type: ignore[method-assign]
    return store


def test_creates_a_usable_account_and_returns_only_the_address():
    backend = FakeSecureBackend()

    address = _store(backend).create_private_key()

    assert address.startswith("0x") and len(address) == 42
    assert backend.writes == 1
    # The stored value is a real key for exactly the returned address.
    assert Account.from_key(backend.stored).address == address
    # The address is not the key.
    assert backend.stored != address
    assert address not in (backend.stored or "")


def test_two_creations_produce_different_accounts():
    first = _store(FakeSecureBackend()).create_private_key()
    second = _store(FakeSecureBackend()).create_private_key()

    assert first != second


def test_refuses_to_replace_an_existing_key():
    """Overwriting a key would destroy access to whatever it controls."""

    existing = Account.create().key.hex()
    backend = FakeSecureBackend(stored=existing)

    with pytest.raises(KeyAlreadyExistsError, match="already exists"):
        _store(backend).create_private_key()

    assert backend.stored == existing
    assert backend.writes == 0


def test_creation_fails_closed_when_the_store_is_unavailable():
    with pytest.raises(KeyStoreError):
        _store(UnavailableBackend()).create_private_key()


def test_signer_address_reports_none_without_a_key():
    assert _store(FakeSecureBackend()).signer_address() is None


def test_signer_address_returns_the_provisioned_address():
    account = Account.create()
    backend = FakeSecureBackend(stored=account.key.hex())

    assert _store(backend).signer_address() == account.address


class CreatingKeyStore:
    def __init__(self, address: str = "0x" + "4" * 40) -> None:
        self.address = address

    def load_private_key(self) -> str:
        raise KeyStoreError("no signer key is provisioned in the OS secure store")

    def create_private_key(self) -> str:
        return self.address


class ExistingKeyStore(CreatingKeyStore):
    def create_private_key(self) -> str:
        raise KeyAlreadyExistsError("a signer key already exists")


class NonCreatingKeyStore:
    def load_private_key(self) -> str:
        raise KeyStoreError("no signer key is provisioned in the OS secure store")


def _service(key_store: object) -> SignerService:
    return SignerService(
        key_store=key_store,
        rpc=object(),
        approval_hmac_secret=b"0" * 32,
        capability_use_store=None,
        outcome_store=None,
        clock=lambda: 0,
    )


def test_service_returns_only_an_address():
    store = CreatingKeyStore()

    result = anyio.run(_service(store).create_signer_account)

    assert result == {"address": store.address}


def test_service_surfaces_the_existing_key_refusal():
    with pytest.raises(SignerDenied, match="already exists"):
        anyio.run(_service(ExistingKeyStore()).create_signer_account)


def test_service_denies_a_store_that_cannot_create():
    with pytest.raises(SignerDenied, match="cannot create"):
        anyio.run(_service(NonCreatingKeyStore()).create_signer_account)
