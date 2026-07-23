"""OS secure-store adapter for the isolated signer.

There is deliberately no environment variable, file, or plaintext fallback for
private key custody.  Tests inject an in-memory ``KeyStore`` instead.
"""

from __future__ import annotations

from typing import Protocol

import keyring
from eth_account import Account

KEYRING_SERVICE = "agentic-wallet.ethereum-signer"
KEYRING_ACCOUNT = "private-key-v1"

_SECURE_BACKEND_TYPES = {
    "keyring.backends.SecretService.Keyring",
    "keyring.backends.macOS.Keyring",
    "keyring.backends.Windows.WinVaultKeyring",
    "keyring.backends.kwallet.DBusKeyring",
}


class KeyStoreError(RuntimeError):
    """The required OS secure store is absent or cannot safely hold a key."""


class KeyStore(Protocol):
    def load_private_key(self) -> str:
        """Return a key only to the signing process; never log it."""


def _backend_name(backend: object) -> str:
    backend_type = type(backend)
    return f"{backend_type.__module__}.{backend_type.__qualname__}"


def require_secure_keyring_backend(backend: object | None = None) -> object:
    """Return a known OS secure backend or fail closed.

    A permissive unknown-backend policy would make a configured plaintext or
    test backend indistinguishable from a secure OS keystore.
    """

    selected = keyring.get_keyring() if backend is None else backend
    name = _backend_name(selected)
    lowered = name.casefold()
    if any(marker in lowered for marker in ("fail", "null", "plaintext")):
        raise KeyStoreError("keyring backend is not a secure OS store")
    if name not in _SECURE_BACKEND_TYPES:
        raise KeyStoreError("keyring backend is not an approved OS secure store")
    return selected


class OSKeyringKeyStore:
    """Private key access through a fixed service/account OS-keyring entry."""

    def __init__(self, *, backend: object | None = None) -> None:
        # Resolve lazily so the recovery-only MCP lookup can start even when
        # keyring is temporarily unavailable. Any key operation still validates
        # and refuses an insecure backend before access.
        self._backend = backend

    def _secure_backend(self) -> object:
        return require_secure_keyring_backend(self._backend)

    def load_private_key(self) -> str:
        try:
            private_key = self._secure_backend().get_password(
                KEYRING_SERVICE, KEYRING_ACCOUNT
            )
        except Exception as error:  # Keyring implementations have varied errors.
            raise KeyStoreError("OS secure store is unavailable") from error
        if not private_key:
            raise KeyStoreError("no signer key is provisioned in the OS secure store")
        try:
            Account.from_key(private_key)
        except Exception as error:
            raise KeyStoreError("OS secure store contains an invalid signer key") from error
        return private_key

    def provision_private_key(self, private_key: str) -> str:
        """Validate then persist a key; callers must acquire it from a TTY."""

        try:
            address = Account.from_key(private_key).address
        except Exception as error:
            raise KeyStoreError("provided private key is invalid") from error
        try:
            self._secure_backend().set_password(
                KEYRING_SERVICE, KEYRING_ACCOUNT, private_key
            )
        except Exception as error:
            raise KeyStoreError("could not write to OS secure store") from error
        return address
