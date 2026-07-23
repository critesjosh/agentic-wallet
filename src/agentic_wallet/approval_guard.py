"""C1 approval binding and invalidation, without any signing authority."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas.approval import ApprovalEnvelope
from .signer_outcome import (
    FRESHNESS_REJECTION_CODES,
    SignerOutcome,
    SignerOutcomeStatus,
)
from .state_machine import StateMachine, WorkflowState


class ApprovalInvalidated(RuntimeError):
    pass


@dataclass
class ApprovalGuard:
    """Record an explicit approval digest and verify freshness before handoff."""

    approved_digest: str | None = None

    def record_explicit_approval(self, envelope: ApprovalEnvelope, *, now: int) -> str:
        if now >= envelope.expires_at:
            raise ApprovalInvalidated("approval envelope is already expired")
        if not envelope.policy.allowed:
            raise ApprovalInvalidated("policy does not allow this plan")
        if envelope.policy.violations:
            raise ApprovalInvalidated("policy contains violations")
        if not envelope.simulation.success or envelope.simulation.mismatch:
            raise ApprovalInvalidated("simulation is not approval-safe")
        if envelope.simulation.plan_id != envelope.plan.plan_id:
            raise ApprovalInvalidated("simulation is for a different plan")
        self.approved_digest = envelope.digest()
        return self.approved_digest

    def require_current(
        self,
        envelope: ApprovalEnvelope,
        *,
        now: int,
        state_anchor: str,
        nonce: int,
        state_machine: StateMachine,
        registry_digest: str | None = None,
    ) -> None:
        reasons: list[str] = []
        if self.approved_digest is None:
            reasons.append("no explicit approval")
        elif envelope.digest() != self.approved_digest:
            reasons.append("approval digest changed")
        if now >= envelope.expires_at:
            reasons.append("approval expired")
        if state_anchor != envelope.state_anchor:
            reasons.append("state anchor changed")
        if nonce != envelope.nonce:
            reasons.append("nonce changed")
        if envelope.is_phase_eight_bound:
            if registry_digest is None:
                reasons.append("current registry digest is required")
            elif registry_digest != envelope.registry_digest:
                reasons.append("registry changed")

        if reasons:
            self.approved_digest = None
            if state_machine.state is WorkflowState.READY_TO_SIGN:
                state_machine.transition(WorkflowState.SIMULATING)
            raise ApprovalInvalidated("; ".join(reasons))

    def begin_submission(
        self,
        envelope: ApprovalEnvelope,
        *,
        now: int,
        state_anchor: str,
        nonce: int,
        registry_digest: str,
        state_machine: StateMachine,
    ) -> str:
        """Revalidate a Phase 8 envelope and take the guarded handoff edge.

        This contains no signing or RPC capability.  It only produces the
        exact approved digest after deterministic freshness checks have moved
        the workflow into ``SUBMITTING``.
        """

        if not envelope.is_phase_eight_bound:
            raise ApprovalInvalidated("signing handoff requires a bound EIP-1559 preimage")
        self.require_current(
            envelope,
            now=now,
            state_anchor=state_anchor,
            nonce=nonce,
            registry_digest=registry_digest,
            state_machine=state_machine,
        )
        state_machine._enter_submitting(self)
        return envelope.digest()

    def apply_signer_outcome(
        self,
        envelope: ApprovalEnvelope,
        outcome: SignerOutcome,
        *,
        state_machine: StateMachine,
    ) -> WorkflowState:
        """Consume a structured result from the private signer boundary."""

        if state_machine.state is not WorkflowState.SUBMITTING:
            raise ApprovalInvalidated("signer outcome requires SUBMITTING state")
        if outcome.envelope_digest != envelope.digest():
            raise ApprovalInvalidated("signer outcome is for a different envelope")
        if outcome.from_address.lower() != envelope.plan.from_address.lower():
            raise ApprovalInvalidated("signer outcome is for a different sender")

        if outcome.status is SignerOutcomeStatus.RESIMULATION_REQUIRED:
            if outcome.code not in FRESHNESS_REJECTION_CODES:
                raise ApprovalInvalidated("signer outcome is not a freshness rejection")
            self.approved_digest = None
            return state_machine._invalidate_submitting(self)
        if outcome.status is SignerOutcomeStatus.UNKNOWN:
            # The capability has been burned and a signed transaction may have
            # reached the network. This state has no retry path.
            self.approved_digest = None
            return state_machine._mark_submission_unknown(self)
        if outcome.status is SignerOutcomeStatus.SUBMITTED:
            self.approved_digest = None
            return state_machine.transition(WorkflowState.SUBMITTED)
        raise ApprovalInvalidated("unsupported signer outcome")

    def mark_submission_unknown_after_lookup_failure(
        self, *, state_machine: StateMachine
    ) -> WorkflowState:
        """Enter the non-retry state when post-sign recovery is unavailable."""

        if state_machine.state is not WorkflowState.SUBMITTING:
            raise ApprovalInvalidated(
                "signer outcome lookup failure requires SUBMITTING state"
            )
        self.approved_digest = None
        return state_machine._mark_submission_unknown(self)
