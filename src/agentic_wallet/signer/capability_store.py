"""Single-use approval-capability claims for isolated signer processes.

Only a SHA-256 fingerprint of a capability is retained.  The production store
uses exclusive file creation, so independently spawned stdio signer processes
cannot both claim one capability.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from pathlib import Path
from typing import Protocol


class CapabilityUseError(RuntimeError):
    """The signer could not safely record a capability use."""


class CapabilityAlreadyUsed(CapabilityUseError):
    """The capability was already claimed by a signer process."""


class CapabilityUseStore(Protocol):
    def claim(self, *, capability_fingerprint: str, expires_at: int, now: int) -> None:
        """Atomically claim a valid capability or raise without key access."""


def capability_fingerprint(token: str) -> str:
    """A ledger-safe identifier; the capability itself is never persisted."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class InMemoryCapabilityUseStore:
    """Lock-protected fake for focused service tests only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: set[str] = set()

    def claim(self, *, capability_fingerprint: str, expires_at: int, now: int) -> None:
        del expires_at, now
        with self._lock:
            if capability_fingerprint in self._claims:
                raise CapabilityAlreadyUsed("approval capability was already used")
            self._claims.add(capability_fingerprint)


class AtomicFileCapabilityUseStore:
    """Bounded, owner-only XDG-state ledger using atomic exclusive creation."""

    _MAX_RECORDS = 4_096

    def __init__(
        self, *, state_dir: Path | None = None, max_records: int = _MAX_RECORDS
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        base = state_dir or self._default_state_dir()
        self._directory = Path(base) / "agentic-wallet" / "signer-capabilities"
        self._max_records = max_records
        self._ensure_secure_directory(self._directory)

    @staticmethod
    def _default_state_dir() -> Path:
        configured = os.environ.get("XDG_STATE_HOME")
        if configured:
            return Path(configured)
        return Path.home() / ".local" / "state"

    @staticmethod
    def _ensure_secure_directory(directory: Path) -> None:
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            info = directory.lstat()
        except OSError as error:
            raise CapabilityUseError("capability-use ledger is unavailable") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CapabilityUseError("capability-use ledger path is unsafe")
        if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise CapabilityUseError("capability-use ledger permissions are unsafe")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise CapabilityUseError("capability-use ledger owner is unsafe")

    @staticmethod
    def _validate_fingerprint(value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise CapabilityUseError("invalid capability fingerprint")
        return value

    def _record_path(self, fingerprint: str) -> Path:
        return self._directory / f"{self._validate_fingerprint(fingerprint)}.claim"

    @staticmethod
    def _read_expiry(path: Path) -> int | None:
        try:
            contents = path.read_text(encoding="ascii")
            value = int(contents.strip())
        except (OSError, UnicodeError, ValueError):
            return None
        return value if value >= 0 else None

    def _discard_expired_and_count(self, now: int) -> int:
        try:
            records = list(self._directory.glob("*.claim"))
        except OSError as error:
            raise CapabilityUseError("capability-use ledger is unavailable") from error
        active = 0
        for record in records:
            expiry = self._read_expiry(record)
            if expiry is None:
                # Fail closed. Another signer may have completed O_EXCL
                # creation but not yet written the expiry. Deleting that file
                # here would let this process claim the same capability.
                # A crash can therefore leave a conservative tombstone; only
                # an operator with access to this owner-only directory may
                # remove corrupt records.
                active += 1
            elif expiry <= now:
                try:
                    record.unlink()
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise CapabilityUseError("capability-use ledger cleanup failed") from error
            else:
                active += 1
        return active

    def claim(self, *, capability_fingerprint: str, expires_at: int, now: int) -> None:
        if now >= expires_at:
            raise CapabilityUseError("approval capability is expired")
        record = self._record_path(capability_fingerprint)
        # Exclusive creation is the cross-process atomic claim.  Capacity is
        # conservatively enforced before creation; a tiny concurrent overshoot
        # remains bounded by concurrent signer processes rather than history.
        if self._discard_expired_and_count(now) >= self._max_records:
            raise CapabilityUseError("capability-use ledger is full")
        try:
            descriptor = os.open(
                record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            raise CapabilityAlreadyUsed("approval capability was already used") from error
        except OSError as error:
            raise CapabilityUseError("capability-use ledger is unavailable") from error
        try:
            os.write(descriptor, f"{expires_at}\n".encode("ascii"))
        except OSError as error:
            try:
                record.unlink()
            except OSError:
                pass
            raise CapabilityUseError("capability-use ledger write failed") from error
        finally:
            os.close(descriptor)
