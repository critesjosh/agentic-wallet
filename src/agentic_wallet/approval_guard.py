"""C1 approval binding and invalidation, without any signing authority."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas.approval import ApprovalEnvelope
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

        if reasons:
            self.approved_digest = None
            if state_machine.state is WorkflowState.READY_TO_SIGN:
                state_machine.transition(WorkflowState.SIMULATING)
            raise ApprovalInvalidated("; ".join(reasons))
