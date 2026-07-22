"""Digest-only sealed-suite commitment validation; never reads plaintext."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..benchmark import BenchmarkCase, load_cases


def validate_sealed_commitment(path: str | Path) -> dict[str, Any]:
    metadata = json.loads(Path(path).read_text())
    allowed = {
        "author_independence_attested",
        "author_role",
        "case_count",
        "created_at",
        "rubric_version",
        "sha256",
        "status",
    }
    if set(metadata) != allowed:
        raise ValueError("sealed commitment contains unexpected or plaintext fields")
    if metadata["status"] != "committed-before-training":
        raise ValueError("sealed suite is not committed before training")
    if metadata["author_independence_attested"] is not True:
        raise ValueError("sealed suite lacks independent-author attestation")
    if not isinstance(metadata["author_role"], str) or not metadata["author_role"]:
        raise ValueError("sealed suite lacks author role")
    if not isinstance(metadata["case_count"], int) or metadata["case_count"] < 20:
        raise ValueError("sealed suite has fewer than 20 cases")
    if not isinstance(metadata["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", metadata["sha256"]
    ):
        raise ValueError("sealed suite digest is invalid")
    return metadata


def load_verified_sealed_cases(
    suite_path: str | Path, commitment_path: str | Path
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    """Verify the pre-training commitment before returning external cases."""

    commitment = validate_sealed_commitment(commitment_path)
    suite = Path(suite_path)
    payload = suite.read_bytes()
    if hashlib.sha256(payload).hexdigest() != commitment["sha256"]:
        raise ValueError("sealed suite digest does not match its commitment")
    cases = load_cases(suite)
    if len(cases) != commitment["case_count"]:
        raise ValueError("sealed suite case count does not match its commitment")
    if any(case.family != "sealed" for case in cases):
        raise ValueError("every sealed-suite record must use family=sealed")
    return cases, commitment
