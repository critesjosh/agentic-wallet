"""Bounded, lock-protected safe submission metadata store.

This is intentionally application-owned rather than model/chat-history state.
A submission remains discoverable for the lifetime of the process by the same
browser action session. It stores only identifiers, hashes, status, timestamps,
a deterministic error code, and a code-generated explorer URL. Raw
transactions, signatures, capabilities, approval objects, endpoint credentials,
and keys have no representation here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
import re
import threading
import time

from eth_utils import to_checksum_address

from .chain_metadata import explorer_transaction_url, get_chain_metadata, normalize_transaction_hash


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ERROR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


class TransactionStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class TransactionStoreError(ValueError):
    pass


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TransactionStoreError(f"{name} must be a short safe identifier")
    return value


def _digest(name: str, value: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise TransactionStoreError(f"{name} must be a sha256 digest")
    return value


def _hash(name: str, value: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise TransactionStoreError(f"{name} must be a 32-byte 0x-prefixed hash")
    return value.lower()


@dataclass(frozen=True, slots=True)
class TransactionRecord:
    session_id: str
    workflow_id: str
    plan_digest: str
    envelope_digest: str
    chain_id: int
    sender: str
    transaction_hash: str
    signing_hash: str
    status: TransactionStatus
    created_at: int
    updated_at: int
    error_code: str | None
    explorer_url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _identifier("session ID", self.session_id))
        object.__setattr__(self, "workflow_id", _identifier("workflow ID", self.workflow_id))
        object.__setattr__(self, "plan_digest", _digest("plan digest", self.plan_digest))
        object.__setattr__(self, "envelope_digest", _digest("envelope digest", self.envelope_digest))
        if isinstance(self.chain_id, bool) or not isinstance(self.chain_id, int) or self.chain_id <= 0:
            raise TransactionStoreError("chain ID must be a positive integer")
        get_chain_metadata(self.chain_id)
        try:
            object.__setattr__(self, "sender", to_checksum_address(self.sender))
        except (TypeError, ValueError) as exc:
            raise TransactionStoreError("sender must be a valid EVM address") from exc
        transaction_hash = _hash("transaction hash", self.transaction_hash)
        object.__setattr__(self, "transaction_hash", transaction_hash)
        object.__setattr__(self, "signing_hash", _hash("signing hash", self.signing_hash))
        if not isinstance(self.status, TransactionStatus):
            raise TransactionStoreError("status must be a TransactionStatus")
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, int) or self.created_at < 0:
            raise TransactionStoreError("created_at must be a non-negative integer")
        if isinstance(self.updated_at, bool) or not isinstance(self.updated_at, int) or self.updated_at < self.created_at:
            raise TransactionStoreError("updated_at must not precede created_at")
        if self.error_code is not None and (not isinstance(self.error_code, str) or not _ERROR_CODE_RE.fullmatch(self.error_code)):
            raise TransactionStoreError("error code must be a deterministic uppercase code")
        expected_url = explorer_transaction_url(self.chain_id, transaction_hash)
        if self.explorer_url != expected_url:
            raise TransactionStoreError("explorer URL must be code-generated from chain and transaction hash")


class TransactionStore:
    """Thread-safe FIFO store with session-scoped public lookup support."""

    def __init__(self, *, max_records: int = 256) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        self._max_records = max_records
        self._records: OrderedDict[str, TransactionRecord] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def max_records(self) -> int:
        return self._max_records

    def record_submission(
        self,
        *,
        session_id: str,
        workflow_id: str,
        plan_digest: str,
        envelope_digest: str,
        chain_id: int,
        sender: str,
        transaction_hash: str,
        signing_hash: str,
        status: TransactionStatus = TransactionStatus.SUBMITTED,
        now: int | None = None,
        error_code: str | None = None,
    ) -> TransactionRecord:
        timestamp = int(time.time()) if now is None else now
        record = TransactionRecord(
            session_id=session_id,
            workflow_id=workflow_id,
            plan_digest=plan_digest,
            envelope_digest=envelope_digest,
            chain_id=chain_id,
            sender=sender,
            transaction_hash=transaction_hash,
            signing_hash=signing_hash,
            status=status,
            created_at=timestamp,
            updated_at=timestamp,
            error_code=error_code,
            explorer_url=explorer_transaction_url(chain_id, transaction_hash),
        )
        with self._lock:
            # A duplicate from the same workflow is idempotent. Never hand a
            # record owned by a different browser session or workflow back to
            # the caller merely because its transaction hash collided.
            existing = self._records.get(record.transaction_hash)
            if existing is not None:
                identity = (
                    "session_id",
                    "workflow_id",
                    "plan_digest",
                    "envelope_digest",
                    "chain_id",
                    "sender",
                    "signing_hash",
                )
                if any(
                    getattr(existing, field) != getattr(record, field)
                    for field in identity
                ):
                    raise TransactionStoreError(
                        "transaction hash is already bound to another record"
                    )
                return existing
            self._records[record.transaction_hash] = record
            while len(self._records) > self._max_records:
                self._records.popitem(last=False)
        return record

    def lookup(self, transaction_hash: str) -> TransactionRecord | None:
        """Internal application lookup by validated transaction hash."""

        normalized = normalize_transaction_hash(transaction_hash)
        with self._lock:
            return self._records.get(normalized)

    get = lookup
    get_by_transaction_hash = lookup

    def lookup_for_session(
        self, session_id: str, transaction_hash: str
    ) -> TransactionRecord | None:
        """Return a record only to the session that created its workflow."""

        expected_session = _identifier("session ID", session_id)
        record = self.lookup(transaction_hash)
        if record is None or record.session_id != expected_session:
            return None
        return record

    def update_status(
        self,
        transaction_hash: str,
        *,
        status: TransactionStatus,
        now: int | None = None,
        error_code: str | None = None,
    ) -> TransactionRecord:
        if not isinstance(status, TransactionStatus):
            raise TransactionStoreError("status must be a TransactionStatus")
        normalized = normalize_transaction_hash(transaction_hash)
        timestamp = int(time.time()) if now is None else now
        with self._lock:
            record = self._records.get(normalized)
            if record is None:
                raise KeyError(normalized)
            if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < record.updated_at:
                raise TransactionStoreError("status timestamp must not move backwards")
            updated = replace(
                record,
                status=status,
                updated_at=timestamp,
                error_code=error_code,
            )
            self._records[normalized] = updated
            return updated

    def records(self) -> tuple[TransactionRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
