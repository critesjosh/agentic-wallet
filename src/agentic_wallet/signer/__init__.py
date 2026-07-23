"""Private, process-isolated Ethereum signing boundary.

This package is intentionally not imported by the web or model-facing layers.
It exposes only a small stdio MCP surface for deterministic application code.
"""

from .capability import ApprovalCapability, CapabilityError, create_approval_capability
from .service import SignerDenied, SignerService, SubmissionResult
from ..signer_outcome import SignerOutcome, SignerOutcomeCode, SignerOutcomeStatus
from .outcome_store import AtomicFileOutcomeStore, InMemoryOutcomeStore, OutcomeStore

__all__ = [
    "ApprovalCapability",
    "AtomicFileOutcomeStore",
    "CapabilityError",
    "InMemoryOutcomeStore",
    "OutcomeStore",
    "SignerDenied",
    "SignerOutcome",
    "SignerOutcomeCode",
    "SignerOutcomeStatus",
    "SignerService",
    "SubmissionResult",
    "create_approval_capability",
]
