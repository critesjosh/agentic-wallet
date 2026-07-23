"""Durable, secret-free signer outcome journal.

An UNKNOWN record is fsynced before broadcast. Only a matching RPC hash may
promote that exact envelope record to SUBMITTED. Raw transactions, signatures,
capabilities, endpoints, and keys have no representation in this store.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import threading
from pathlib import Path
from typing import Protocol

from ..signer_outcome import SignerOutcome, SignerOutcomeStatus

_ENVELOPE_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")


class OutcomeStoreError(RuntimeError):
    """The outcome journal could not prove a durable safe state."""


class OutcomeStore(Protocol):
    def record_unknown(self, outcome: SignerOutcome) -> None:
        """Durably record possible broadcast before any network submission."""

    def mark_submitted(self, outcome: SignerOutcome) -> None:
        """Promote an existing matching UNKNOWN record to SUBMITTED."""

    def lookup(self, envelope_digest: str) -> SignerOutcome | None:
        """Read the latest durable outcome for one exact envelope."""


def _digest_hex(envelope_digest: str) -> str:
    match = _ENVELOPE_DIGEST.fullmatch(envelope_digest)
    if match is None:
        raise OutcomeStoreError("invalid envelope digest")
    return match.group(1)


class InMemoryOutcomeStore:
    """Lock-protected injected store for service tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outcomes: dict[str, SignerOutcome] = {}

    def record_unknown(self, outcome: SignerOutcome) -> None:
        if outcome.status is not SignerOutcomeStatus.UNKNOWN:
            raise OutcomeStoreError("initial signer outcome must be UNKNOWN")
        with self._lock:
            if outcome.envelope_digest in self._outcomes:
                raise OutcomeStoreError("signer outcome already exists")
            self._outcomes[outcome.envelope_digest] = outcome

    def mark_submitted(self, outcome: SignerOutcome) -> None:
        if outcome.status is not SignerOutcomeStatus.SUBMITTED:
            raise OutcomeStoreError("final signer outcome must be SUBMITTED")
        with self._lock:
            current = self._outcomes.get(outcome.envelope_digest)
            if (
                current is None
                or current.status is not SignerOutcomeStatus.UNKNOWN
                or current.transaction_hash != outcome.transaction_hash
                or current.transaction_signing_hash
                != outcome.transaction_signing_hash
            ):
                raise OutcomeStoreError("submitted outcome does not match journal")
            self._outcomes[outcome.envelope_digest] = outcome

    def lookup(self, envelope_digest: str) -> SignerOutcome | None:
        _digest_hex(envelope_digest)
        with self._lock:
            return self._outcomes.get(envelope_digest)


class AtomicFileOutcomeStore:
    """Owner-only XDG journal with durable atomic UNKNOWN→SUBMITTED updates."""

    _MAX_RECORDS = 4_096

    def __init__(
        self, *, state_dir: Path | None = None, max_records: int = _MAX_RECORDS
    ) -> None:
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or max_records <= 0
        ):
            raise ValueError("max_records must be a positive integer")
        base = state_dir or self._default_state_dir()
        self._directory = Path(base) / "agentic-wallet" / "signer-outcomes"
        self._max_records = max_records
        self._ensure_secure_directory()

    @staticmethod
    def _default_state_dir() -> Path:
        configured = os.environ.get("XDG_STATE_HOME")
        return Path(configured) if configured else Path.home() / ".local" / "state"

    def _ensure_secure_directory(self) -> None:
        self._directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            info = self._directory.lstat()
        except OSError as error:
            raise OutcomeStoreError("signer outcome journal is unavailable") from error
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        ):
            raise OutcomeStoreError("signer outcome journal path is unsafe")

    def _path(self, envelope_digest: str) -> Path:
        return self._directory / f"{_digest_hex(envelope_digest)}.json"

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short journal write")
            written += count
        os.fsync(descriptor)

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(self._directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise OutcomeStoreError("signer outcome journal sync failed") from error

    @staticmethod
    def _payload(outcome: SignerOutcome) -> bytes:
        return (outcome.model_dump_json() + "\n").encode("utf-8")

    def record_unknown(self, outcome: SignerOutcome) -> None:
        if outcome.status is not SignerOutcomeStatus.UNKNOWN:
            raise OutcomeStoreError("initial signer outcome must be UNKNOWN")
        path = self._path(outcome.envelope_digest)
        if len(list(self._directory.glob("*.json"))) >= self._max_records:
            raise OutcomeStoreError("signer outcome journal is full")
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            raise OutcomeStoreError("signer outcome already exists") from error
        except OSError as error:
            raise OutcomeStoreError("signer outcome journal is unavailable") from error
        try:
            self._write_all(descriptor, self._payload(outcome))
        except OSError as error:
            try:
                path.unlink()
            except OSError:
                pass
            raise OutcomeStoreError("signer outcome journal write failed") from error
        finally:
            os.close(descriptor)
        self._fsync_directory()

    def _replace_matching_unknown(self, outcome: SignerOutcome) -> None:
        current = self.lookup(outcome.envelope_digest)
        if (
            current is None
            or current.status is not SignerOutcomeStatus.UNKNOWN
            or current.transaction_hash != outcome.transaction_hash
            or current.transaction_signing_hash != outcome.transaction_signing_hash
        ):
            raise OutcomeStoreError("replacement outcome does not match journal")

        path = self._path(outcome.envelope_digest)
        temporary = self._directory / (
            f".{_digest_hex(outcome.envelope_digest)}."
            f"{os.getpid()}.{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            try:
                self._write_all(descriptor, self._payload(outcome))
            finally:
                os.close(descriptor)
            os.replace(temporary, path)
            self._fsync_directory()
        except OSError as error:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise OutcomeStoreError("signer outcome journal update failed") from error

    def mark_submitted(self, outcome: SignerOutcome) -> None:
        if outcome.status is not SignerOutcomeStatus.SUBMITTED:
            raise OutcomeStoreError("final signer outcome must be SUBMITTED")
        self._replace_matching_unknown(outcome)

    def lookup(self, envelope_digest: str) -> SignerOutcome | None:
        path = self._path(envelope_digest)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise OutcomeStoreError("signer outcome journal is unavailable") from error
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        ):
            raise OutcomeStoreError("signer outcome journal record is unsafe")
        try:
            outcome = SignerOutcome.model_validate_json(path.read_bytes())
        except Exception as error:
            raise OutcomeStoreError("signer outcome journal record is invalid") from error
        if (
            outcome.envelope_digest != envelope_digest
            or outcome.status
            not in {SignerOutcomeStatus.UNKNOWN, SignerOutcomeStatus.SUBMITTED}
        ):
            raise OutcomeStoreError("signer outcome journal record is invalid")
        return outcome
