"""End-to-end unsigned planning, simulation, policy, and approval workflow."""

from __future__ import annotations

from typing import Any

from .approval_guard import ApprovalGuard, ApprovalInvalidated
from .candidate_binding import bind_transfer_candidate
from .harness import MockReadOnlyHarness
from .planning import build_swap_plan, build_transfer_plan
from .policy_engine import WalletPolicy, evaluate_policy
from .registry import Registry
from .schemas.approval import ApprovalEnvelope
from .schemas.common import Amount
from .schemas.policy import PolicyResult
from .schemas.quote import SwapQuote
from .schemas.simulation_result import BalanceChange, SimulationResult
from .schemas.signing import Eip1559Transaction
from .schemas.transaction_plan import TransactionPlan
from .schemas.tool_call import ToolCall
from .simulation import simulate_plan
from .signer_outcome import SignerOutcome
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
        self.state_machine.bind_submission_guard(self.approval_guard)

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

    def plan_transfer_from_candidate(
        self,
        *,
        call: ToolCall,
        context: dict[str, Any],
        gas_reserve: Amount,
    ) -> TransactionPlan:
        """Resolve a trusted candidate ID, then enter normal deterministic planning."""

        bound = bind_transfer_candidate(call, context)
        if bound.chain_id != self.harness.portfolio.chain_id:
            raise UnsignedWorkflowError(
                "candidate transfer chain does not match wallet"
            )
        asset = self.registry.resolve(bound.asset_id)
        return self.plan_transfer(
            asset_id=bound.asset_id,
            amount=Amount(
                base_units=bound.amount_base_units,
                decimals=asset.decimals,
            ),
            recipient=bound.recipient,
            gas_reserve=gas_reserve,
        )

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
        snapshot_block_hash: str | None = None,
        nonce: int,
        gas_used: int,
        gas_fee: Amount,
        observed_changes: list[BalanceChange] | None = None,
        success: bool = True,
        signing_transaction: Eip1559Transaction | None = None,
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

        if signing_transaction is not None:
            if snapshot_block_hash is None:
                raise UnsignedWorkflowError(
                    "Phase 8 signing requires a canonical snapshot block hash"
                )
            try:
                asset = self.registry.resolve(self.plan.asset_id)
            except Exception as exc:  # Registry errors become workflow errors here.
                raise UnsignedWorkflowError("signing plan asset is not registry-resolved") from exc
            if self.plan.kind != "transfer" or not asset.is_native:
                raise UnsignedWorkflowError(
                    "Phase 8 signing supports only native-token transfers"
                )

        native_asset = self.registry.native_asset(self.plan.chain_id)

        self.simulation = simulate_plan(
            self.plan,
            block=block,
            gas_used=gas_used,
            gas_fee=gas_fee,
            observed_changes=observed_changes,
            success=success,
            signing_transaction=signing_transaction,
            native_asset_id=native_asset.asset_id,
            native_asset_decimals=native_asset.decimals,
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
            registry_digest=(self.registry.version_digest() if signing_transaction else None),
            simulated_transaction_digest=(
                signing_transaction.digest() if signing_transaction else None
            ),
            snapshot_block_hash=(
                snapshot_block_hash.lower()
                if signing_transaction and snapshot_block_hash is not None
                else None
            ),
            signing_transaction=signing_transaction,
        )
        self.state_machine.transition(WorkflowState.AWAITING_CONFIRMATION)
        return self.envelope

    def record_user_approval(self, presented_digest: str, *, now: int) -> None:
        self._require_state(WorkflowState.AWAITING_CONFIRMATION)
        if self.envelope is None or presented_digest != self.envelope.digest():
            raise ApprovalInvalidated("user did not approve the exact envelope digest")
        try:
            self.approval_guard.record_explicit_approval(self.envelope, now=now)
        except ApprovalInvalidated:
            # A freshness-window expiry is not a user rejection.  The displayed
            # envelope is stale and must be rebuilt and re-simulated.
            if now >= self.envelope.expires_at:
                self.state_machine.transition(WorkflowState.SIMULATING)
            raise
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
            registry_digest=self.registry.version_digest(),
            state_machine=self.state_machine,
        )
        return self.envelope.digest()

    def authorize_submission(
        self, *, now: int, state_anchor: str, nonce: int
    ) -> str:
        """Perform the guarded, freshness-checked Phase 8 signing handoff.

        This does not sign, invoke an MCP server, send RPC, or broadcast a
        transaction.  It only moves an already-approved complete envelope to
        ``SUBMITTING`` for a separate deterministic signer boundary.
        """

        self._require_state(WorkflowState.READY_TO_SIGN)
        if self.envelope is None:
            raise ApprovalInvalidated("approval envelope is missing")
        return self.approval_guard.begin_submission(
            self.envelope,
            now=now,
            state_anchor=state_anchor,
            nonce=nonce,
            registry_digest=self.registry.version_digest(),
            state_machine=self.state_machine,
        )

    def handle_signer_outcome(self, outcome: SignerOutcome) -> WorkflowState:
        """Apply a typed private-signer result without exposing signing authority."""

        self._require_state(WorkflowState.SUBMITTING)
        if self.envelope is None:
            raise ApprovalInvalidated("approval envelope is missing")
        return self.approval_guard.apply_signer_outcome(
            self.envelope,
            outcome,
            state_machine=self.state_machine,
        )

    def mark_submission_unknown_after_lookup_failure(self) -> WorkflowState:
        """Conservatively stop retries when signer recovery is unavailable."""

        self._require_state(WorkflowState.SUBMITTING)
        return self.approval_guard.mark_submission_unknown_after_lookup_failure(
            state_machine=self.state_machine
        )
