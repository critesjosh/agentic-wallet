"""End-to-end unsigned planning, simulation, policy, and approval workflow."""

from __future__ import annotations

from .approval_guard import ApprovalGuard, ApprovalInvalidated
from .harness import MockReadOnlyHarness
from .planning import build_swap_plan, build_transfer_plan
from .policy_engine import WalletPolicy, evaluate_policy
from .registry import Registry
from .schemas.approval import ApprovalEnvelope
from .schemas.common import Amount
from .schemas.policy import PolicyResult
from .schemas.quote import SwapQuote
from .schemas.simulation_result import BalanceChange, SimulationResult
from .schemas.transaction_plan import TransactionPlan
from .simulation import simulate_plan
from .state_machine import StateMachine, WorkflowState


class UnsignedWorkflowError(RuntimeError):
    pass


class UnsignedTransactionWorkflow:
    """A simulated-only flow with no signing or submission capability."""

    def __init__(
        self,
        harness: MockReadOnlyHarness,
        policy: WalletPolicy,
        *,
        approval_ttl_seconds: int = 120,
    ) -> None:
        if approval_ttl_seconds <= 0:
            raise ValueError("approval TTL must be positive")
        self.harness = harness
        self.registry: Registry = harness.registry
        self.wallet_policy = policy
        self.approval_ttl_seconds = approval_ttl_seconds
        self.state_machine = StateMachine(WorkflowState.PLANNING)
        self.plan: TransactionPlan | None = None
        self.simulation: SimulationResult | None = None
        self.policy_result: PolicyResult | None = None
        self.envelope: ApprovalEnvelope | None = None
        self.approval_guard = ApprovalGuard()

    def _require_state(self, expected: WorkflowState) -> None:
        if self.state_machine.state is not expected:
            raise UnsignedWorkflowError(
                f"operation requires {expected.value}, got {self.state_machine.state.value}"
            )

    def plan_transfer(
        self,
        *,
        asset_id: str,
        amount: Amount,
        recipient: str,
        gas_reserve: Amount,
    ) -> TransactionPlan:
        self._require_state(WorkflowState.PLANNING)
        self.plan = build_transfer_plan(
            self.harness,
            asset_id=asset_id,
            amount=amount,
            recipient=recipient,
            gas_reserve=gas_reserve,
        )
        self.state_machine.transition(WorkflowState.PLAN_READY)
        return self.plan

    def plan_swap(
        self, *, quote: SwapQuote, now: int, gas_reserve: Amount
    ) -> TransactionPlan:
        self._require_state(WorkflowState.PLANNING)
        self.plan = build_swap_plan(
            self.harness, quote=quote, now=now, gas_reserve=gas_reserve
        )
        self.state_machine.transition(WorkflowState.PLAN_READY)
        return self.plan

    def simulate_and_check(
        self,
        *,
        now: int,
        block: int,
        state_anchor: str,
        nonce: int,
        gas_used: int,
        gas_fee: Amount,
        observed_changes: list[BalanceChange] | None = None,
        success: bool = True,
    ) -> ApprovalEnvelope | None:
        if self.state_machine.state not in {
            WorkflowState.PLAN_READY,
            WorkflowState.SIMULATING,
        }:
            raise UnsignedWorkflowError(
                "simulation requires PLAN_READY or a mandatory re-simulation, "
                f"got {self.state_machine.state.value}"
            )
        if self.plan is None:
            raise UnsignedWorkflowError("no plan exists")
        if self.state_machine.state is WorkflowState.PLAN_READY:
            self.state_machine.transition(WorkflowState.SIMULATING)
        self.simulation = None
        self.policy_result = None
        self.envelope = None
        if self.plan.quote_expires_at is not None and now >= self.plan.quote_expires_at:
            self.state_machine.transition(WorkflowState.QUOTE_EXPIRED)
            return None

        self.simulation = simulate_plan(
            self.plan,
            block=block,
            gas_used=gas_used,
            gas_fee=gas_fee,
            observed_changes=observed_changes,
            success=success,
        )
        self.policy_result = evaluate_policy(
            self.plan,
            self.simulation,
            policy=self.wallet_policy,
            registry=self.registry,
            now=now,
        )
        if not self.policy_result.allowed:
            self.state_machine.transition(WorkflowState.REJECTED_BY_POLICY)
            return None

        expires_at = now + self.approval_ttl_seconds
        if self.plan.quote_expires_at is not None:
            expires_at = min(expires_at, self.plan.quote_expires_at)
        self.envelope = ApprovalEnvelope(
            chain_id=self.plan.chain_id,
            plan=self.plan,
            simulation=self.simulation,
            policy=self.policy_result,
            expires_at=expires_at,
            state_anchor=state_anchor,
            nonce=nonce,
        )
        self.state_machine.transition(WorkflowState.AWAITING_CONFIRMATION)
        return self.envelope

    def record_user_approval(self, presented_digest: str, *, now: int) -> None:
        self._require_state(WorkflowState.AWAITING_CONFIRMATION)
        if self.envelope is None or presented_digest != self.envelope.digest():
            raise ApprovalInvalidated("user did not approve the exact envelope digest")
        self.approval_guard.record_explicit_approval(self.envelope, now=now)
        self.state_machine.transition(WorkflowState.READY_TO_SIGN)

    def require_current_approval(
        self, *, now: int, state_anchor: str, nonce: int
    ) -> str:
        """Validate handoff readiness; this method does not sign or submit."""

        self._require_state(WorkflowState.READY_TO_SIGN)
        if self.envelope is None:
            raise ApprovalInvalidated("approval envelope is missing")
        self.approval_guard.require_current(
            self.envelope,
            now=now,
            state_anchor=state_anchor,
            nonce=nonce,
            state_machine=self.state_machine,
        )
        return self.envelope.digest()
