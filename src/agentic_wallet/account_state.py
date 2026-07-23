"""Canonical relevant-account-state anchors for approval freshness checks."""

from __future__ import annotations

from dataclasses import dataclass
import re

from eth_utils import to_checksum_address

from .digest import canonical_digest


_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


class AccountStateError(ValueError):
    """Raised when facts cannot form a safe, canonical account-state anchor."""


def _address(value: str) -> str:
    if not isinstance(value, str):
        raise AccountStateError("account address must be a string")
    try:
        return to_checksum_address(value)
    except (TypeError, ValueError) as exc:
        raise AccountStateError("account address must be a valid EVM address") from exc


def _block_hash(value: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise AccountStateError("block hash must be a 32-byte 0x-prefixed hex string")
    return value.lower()


def _non_negative(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AccountStateError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RelevantAccountState:
    """Facts that can invalidate a native transfer before it reaches a signer.

    The pending nonce binds the sending account's pending transaction sequence;
    the balance and a concrete latest block bind the spendable-state view.  The
    digest intentionally contains no endpoint, credential, raw transaction, or
    approval capability.
    """

    chain_id: int
    address: str
    pending_nonce: int
    balance: int
    block_number: int
    block_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.chain_id, bool) or not isinstance(self.chain_id, int) or self.chain_id <= 0:
            raise AccountStateError("chain ID must be a positive integer")
        object.__setattr__(self, "address", _address(self.address))
        object.__setattr__(self, "pending_nonce", _non_negative("pending nonce", self.pending_nonce))
        object.__setattr__(self, "balance", _non_negative("balance", self.balance))
        object.__setattr__(self, "block_number", _non_negative("block number", self.block_number))
        object.__setattr__(self, "block_hash", _block_hash(self.block_hash))

    def payload(self) -> dict[str, object]:
        """Transaction-relevant facts committed by :attr:`anchor`.

        Block number and hash are not themselves account-state drift: including
        them here would invalidate every approval on every new block even when
        balance and pending nonce were unchanged. Phase 8 separately binds the
        captured hash in the approval envelope and uses it for EIP-1898
        hash-pinned preflight. The signer repeats call/gas checks against current
        state before broadcasting.
        """

        return {
            "chain_id": self.chain_id,
            "address": self.address,
            "pending_nonce": self.pending_nonce,
            "balance": self.balance,
        }

    @property
    def anchor(self) -> str:
        return canonical_digest(self.payload())

    # This name makes handoff to ApprovalEnvelope explicit without making an
    # RPC snapshot object itself serializable as an approval.
    @property
    def state_anchor(self) -> str:
        return self.anchor


# Shorter compatibility name for callers that use the approval terminology.
AccountStateAnchor = RelevantAccountState
