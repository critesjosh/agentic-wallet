"""Durable, opaque lifecycle records for one-shot blinded evaluations.

The record deliberately stores only commitment-derived identifiers, aggregate
digest, and status. ``claimed`` is itself an irreversible consumed state: a
host loss or SIGKILL can prevent cleanup, but can never make the suite reusable.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BlindedEvaluationAlreadyUsed(RuntimeError):
    """Raised when an evaluation commitment has already been claimed."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _custody_root(commitment: dict[str, Any]) -> Path:
    return Path(commitment["evaluation_config"]["custody_root"])


def blinded_state_path(commitment: dict[str, Any]) -> Path:
    custody_root = _custody_root(commitment)
    return custody_root / "claims" / f"{commitment['sha256']}.json"


def blinded_report_path(commitment: dict[str, Any]) -> Path:
    custody_root = _custody_root(commitment)
    return custody_root / "reports" / f"{commitment['sha256']}.json"


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _durable_create(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise BlindedEvaluationAlreadyUsed(
            "blinded evaluation was already claimed; it cannot be rerun"
        ) from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(data)
            output.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def claim_blinded_evaluation(commitment: dict[str, Any]) -> Path:
    """Atomically consume a commitment before any suite plaintext is opened."""

    state_path = blinded_state_path(commitment)
    _durable_create(
        state_path,
        {
            "candidate_artifact_sha256": commitment["candidate_artifact_sha256"],
            "claimed_at": _timestamp(),
            "commitment_sha256": commitment["sha256"],
            "status": "claimed",
        },
    )
    return state_path


def finish_blinded_evaluation(
    state_path: Path,
    status: str,
    *,
    aggregate_sha256: str | None = None,
) -> None:
    """Immutably refine ``claimed`` to ``completed`` or ``retired``."""

    if status not in {"completed", "retired"}:
        raise ValueError("invalid blinded evaluation terminal status")
    with state_path.open("r+b") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        try:
            record = json.loads(state.read())
            if record.get("status") != "claimed":
                raise RuntimeError("blinded evaluation lifecycle is terminal")
            if status == "completed":
                if (
                    not isinstance(aggregate_sha256, str)
                    or len(aggregate_sha256) != 64
                ):
                    raise ValueError("completed evaluation requires aggregate digest")
                record["aggregate_sha256"] = aggregate_sha256
            elif aggregate_sha256 is not None:
                raise ValueError("retired evaluation cannot bind an aggregate digest")
            record["status"] = status
            record["finished_at"] = _timestamp()
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{state_path.name}.",
                suffix=".transition",
                dir=state_path.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(
                        (json.dumps(record, sort_keys=True) + "\n").encode()
                    )
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, state_path)
                _fsync_directory(state_path.parent)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(state.fileno(), fcntl.LOCK_UN)


def publish_blinded_aggregate(
    commitment: dict[str, Any],
    aggregate: dict[str, Any],
) -> tuple[Path, str]:
    """Atomically publish one aggregate report without replacing an old one."""

    path = blinded_report_path(commitment)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publish", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BlindedEvaluationAlreadyUsed(
                "aggregate report already exists; evaluation cannot be rerun"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path, digest
