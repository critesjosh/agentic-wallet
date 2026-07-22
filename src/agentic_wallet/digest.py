"""Canonical serialization and hashing for the C1 approval-integrity contract.

A plan digest binds user approval to the exact plan, simulation, policy,
expiry, state anchor, and nonce. Any change to any bound field changes the
digest and therefore invalidates approval. See plan.md C1.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DIGEST_ALGORITHM = "sha256"


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, no insignificant whitespace."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_digest(obj: Any) -> str:
    """Return ``"<algorithm>:<hexdigest>"`` over the canonical encoding."""
    h = hashlib.new(DIGEST_ALGORITHM)
    h.update(canonical_bytes(obj))
    return f"{DIGEST_ALGORITHM}:{h.hexdigest()}"
