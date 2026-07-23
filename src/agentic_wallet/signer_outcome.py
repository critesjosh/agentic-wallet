"""Safe structured outcomes from the private signer boundary."""

from __future__ import annotations

from enum import Enum

from pydantic import ConfigDict, Field, model_validator

from .schemas.common import EvmAddress, StrictModel
from .schemas.signing import Bytes32


class SignerOutcomeStatus(str, Enum):
    RESIMULATION_REQUIRED = "RESIMULATION_REQUIRED"
    UNKNOWN = "UNKNOWN"
    SUBMITTED = "SUBMITTED"


class SignerOutcomeCode(str, Enum):
    RPC_CHAIN_CHANGED = "RPC_CHAIN_CHANGED"
    PENDING_NONCE_CHANGED = "PENDING_NONCE_CHANGED"
    RELEVANT_STATE_CHANGED = "RELEVANT_STATE_CHANGED"
    LIVE_PREFLIGHT_FAILED = "LIVE_PREFLIGHT_FAILED"
    BROADCAST_RESULT_UNKNOWN = "BROADCAST_RESULT_UNKNOWN"
    BROADCAST_HASH_MISMATCH = "BROADCAST_HASH_MISMATCH"
    SUBMITTED = "SUBMITTED"


FRESHNESS_REJECTION_CODES: frozenset[SignerOutcomeCode] = frozenset(
    {
        SignerOutcomeCode.RPC_CHAIN_CHANGED,
        SignerOutcomeCode.PENDING_NONCE_CHANGED,
        SignerOutcomeCode.RELEVANT_STATE_CHANGED,
        SignerOutcomeCode.LIVE_PREFLIGHT_FAILED,
    }
)

UNKNOWN_SUBMISSION_CODES: frozenset[SignerOutcomeCode] = frozenset(
    {
        SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN,
        SignerOutcomeCode.BROADCAST_HASH_MISMATCH,
    }
)


class SignerOutcome(StrictModel):
    """Metadata safe to cross MCP; secret signing material has no field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SignerOutcomeStatus
    code: SignerOutcomeCode
    envelope_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    from_address: EvmAddress
    transaction_hash: Bytes32 | None = None
    transaction_signing_hash: Bytes32 | None = None

    @model_validator(mode="after")
    def _shape_matches_status(self) -> "SignerOutcome":
        hashes = (self.transaction_hash, self.transaction_signing_hash)
        if self.status is SignerOutcomeStatus.RESIMULATION_REQUIRED:
            if self.code not in FRESHNESS_REJECTION_CODES or any(hashes):
                raise ValueError("re-simulation outcome must have a freshness code and no hashes")
        elif self.status is SignerOutcomeStatus.UNKNOWN:
            if self.code not in UNKNOWN_SUBMISSION_CODES or any(
                value is None for value in hashes
            ):
                raise ValueError("unknown submission outcome must retain both local hashes")
        elif (
            self.status is not SignerOutcomeStatus.SUBMITTED
            or self.code is not SignerOutcomeCode.SUBMITTED
            or any(value is None for value in hashes)
        ):
            raise ValueError("submitted outcome must retain both local hashes")
        return self
