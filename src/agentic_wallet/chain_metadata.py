"""Code-owned metadata for the small supported-chain signing allowlist.

Explorer links are presentation metadata, never input from a model, client, or
RPC response.  Keeping the mapping here means an unknown chain cannot be made
to look trusted by supplying an arbitrary explorer URL.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_TRANSACTION_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


class ChainMetadataError(ValueError):
    """Base error for a chain or transaction hash that is not trusted."""


class UnknownChainError(ChainMetadataError):
    """Raised when a chain is outside the code-owned signing allowlist."""


class InvalidTransactionHashError(ChainMetadataError):
    """Raised when a value is not an Ethereum transaction hash."""


@dataclass(frozen=True, slots=True)
class ChainMetadata:
    chain_id: int
    name: str
    native_asset_id: str
    explorer_transaction_prefix: str

    def transaction_url(self, transaction_hash: str) -> str:
        return f"{self.explorer_transaction_prefix}{normalize_transaction_hash(transaction_hash)}"


# This is deliberately a small allowlist for the Phase 8 native-transfer POC.
_SUPPORTED_CHAINS: dict[int, ChainMetadata] = {
    1: ChainMetadata(1, "Ethereum", "ethereum:native", "https://etherscan.io/tx/"),
    8453: ChainMetadata(8453, "Base", "base:native", "https://basescan.org/tx/"),
    84532: ChainMetadata(
        84532,
        "Base Sepolia",
        "base:sepolia-native",
        "https://sepolia.basescan.org/tx/",
    ),
}


def get_chain_metadata(chain_id: int) -> ChainMetadata:
    """Return trusted metadata, failing closed for all unsupported chains."""

    # bool is an int subclass but is never a meaningful EVM chain ID.
    if isinstance(chain_id, bool) or not isinstance(chain_id, int):
        raise UnknownChainError("chain ID must be an integer in the supported allowlist")
    try:
        return _SUPPORTED_CHAINS[chain_id]
    except KeyError as exc:
        raise UnknownChainError(f"unsupported chain ID: {chain_id}") from exc


def normalize_transaction_hash(transaction_hash: str) -> str:
    """Validate and canonicalize a transaction hash without accepting bytes."""

    if not isinstance(transaction_hash, str) or not _TRANSACTION_HASH_RE.fullmatch(
        transaction_hash
    ):
        raise InvalidTransactionHashError("transaction hash must be 32-byte 0x-prefixed hex")
    return transaction_hash.lower()


def explorer_transaction_url(chain_id: int, transaction_hash: str) -> str:
    """Build a link using only trusted metadata and a validated hash."""

    return get_chain_metadata(chain_id).transaction_url(transaction_hash)
