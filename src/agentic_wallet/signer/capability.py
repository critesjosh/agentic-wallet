"""Short-lived HMAC capabilities for an already-approved envelope.

The capability is not an approval object and cannot authorize a different
envelope.  Its only purpose is to let the isolated signer verify that its
deterministic caller observed explicit approval for this exact digest.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Callable

from ..digest import canonical_bytes

_VERSION = "v1"
_MAX_LIFETIME_SECONDS = 300


class CapabilityError(RuntimeError):
    """A capability is malformed, expired, or not bound to this envelope."""


def decode_approval_hmac_key(encoded: str) -> bytes:
    """Decode a URL-safe base64 capability key with at least 256 bits."""

    if not isinstance(encoded, str) or not encoded:
        raise CapabilityError("approval capability key is not configured")
    try:
        raw = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, UnicodeEncodeError) as error:
        raise CapabilityError("approval capability key must be URL-safe base64") from error
    if len(raw) < 32:
        raise CapabilityError("approval capability key must decode to at least 32 bytes")
    return raw


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, UnicodeEncodeError) as error:
        raise CapabilityError("malformed approval capability") from error


@dataclass(frozen=True)
class ApprovalCapability:
    """Verified, non-secret claims carried by an approval capability."""

    envelope_digest: str
    issued_at: int
    expires_at: int
    capability_id: str

    @classmethod
    def from_payload(cls, payload: object) -> "ApprovalCapability":
        if not isinstance(payload, dict) or set(payload) != {
            "capability_id",
            "envelope_digest",
            "expires_at",
            "issued_at",
        }:
            raise CapabilityError("malformed approval capability")
        digest = payload["envelope_digest"]
        issued_at = payload["issued_at"]
        expires_at = payload["expires_at"]
        capability_id = payload["capability_id"]
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or not isinstance(issued_at, int)
            or isinstance(issued_at, bool)
            or not isinstance(expires_at, int)
            or isinstance(expires_at, bool)
            or not isinstance(capability_id, str)
            or len(capability_id) < 16
        ):
            raise CapabilityError("malformed approval capability")
        if expires_at <= issued_at or expires_at - issued_at > _MAX_LIFETIME_SECONDS:
            raise CapabilityError("invalid approval capability lifetime")
        return cls(
            envelope_digest=digest,
            issued_at=issued_at,
            expires_at=expires_at,
            capability_id=capability_id,
        )


def create_approval_capability(
    *,
    envelope_digest: str,
    envelope_expires_at: int,
    secret: bytes,
    now: int | None = None,
    lifetime_seconds: int = 60,
) -> str:
    """Create a capability for one exact envelope after explicit approval."""

    issued_at = int(time.time()) if now is None else now
    if not isinstance(issued_at, int) or isinstance(issued_at, bool):
        raise ValueError("now must be an integer unix timestamp")
    if not isinstance(lifetime_seconds, int) or not 0 < lifetime_seconds <= _MAX_LIFETIME_SECONDS:
        raise ValueError("capability lifetime must be between 1 and 300 seconds")
    if len(secret) < 32:
        raise ValueError("approval capability secret must be at least 32 bytes")
    expires_at = min(issued_at + lifetime_seconds, envelope_expires_at)
    if expires_at <= issued_at:
        raise CapabilityError("approval envelope is expired")
    payload = {
        "capability_id": secrets.token_urlsafe(18),
        "envelope_digest": envelope_digest,
        "expires_at": expires_at,
        "issued_at": issued_at,
    }
    encoded_payload = _b64encode(canonical_bytes(payload))
    signature = hmac.new(
        secret, f"{_VERSION}.{encoded_payload}".encode("ascii"), hashlib.sha256
    ).digest()
    return f"{_VERSION}.{encoded_payload}.{_b64encode(signature)}"


def verify_approval_capability(
    token: str,
    *,
    secret: bytes,
    envelope_digest: str,
    envelope_expires_at: int,
    now: int | None = None,
    clock: Callable[[], float] = time.time,
) -> ApprovalCapability:
    """Verify an HMAC token with the signer's clock and exact envelope binding."""

    if len(secret) < 32 or not isinstance(token, str):
        raise CapabilityError("invalid approval capability")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _VERSION:
        raise CapabilityError("malformed approval capability")
    encoded_payload, encoded_signature = parts[1:]
    expected = hmac.new(
        secret, f"{_VERSION}.{encoded_payload}".encode("ascii"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, _b64decode(encoded_signature)):
        raise CapabilityError("invalid approval capability")
    try:
        payload = json.loads(_b64decode(encoded_payload))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CapabilityError("malformed approval capability") from error
    claims = ApprovalCapability.from_payload(payload)
    observed_now = int(clock()) if now is None else now
    if observed_now >= claims.expires_at or claims.expires_at > envelope_expires_at:
        raise CapabilityError("approval capability is expired")
    if not hmac.compare_digest(claims.envelope_digest, envelope_digest):
        raise CapabilityError("approval capability is not for this envelope")
    return claims
