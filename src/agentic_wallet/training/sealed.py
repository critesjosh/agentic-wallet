"""Digest-only sealed-suite commitment validation; never reads plaintext."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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
