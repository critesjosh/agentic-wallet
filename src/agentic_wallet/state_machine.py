"""Explicit workflow state machine (plan.md sec 7, plus the Consensus revisions).

The model may recommend a transition, but the application validates it here.
Added over the plan's original list: CANCELLED (distinct from USER_REJECTED),
QUOTE_EXPIRED, and the READY_TO_SIGN -> SIMULATING re-simulation edge that the
C1 approval-integrity contract requires on staleness or approval invalidation.
"""

from __future__ import annotations

from enum import Enum


class WorkflowState(str, Enum):
    IDLE = "IDLE"
    UNDERSTANDING_INTENT = "UNDERSTANDING_INTENT"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    COLLECTING_STATE = "COLLECTING_STATE"
    PLANNING = "PLANNING"
    PLAN_READY = "PLAN_READY"
    SIMULATING = "SIMULATING"
    SIMULATION_FAILED = "SIMULATION_FAILED"
    SIMULATION_MISMATCH = "SIMULATION_MISMATCH"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    USER_REJECTED = "USER_REJECTED"
    CANCELLED = "CANCELLED"
    QUOTE_EXPIRED = "QUOTE_EXPIRED"
    READY_TO_SIGN = "READY_TO_SIGN"
    SUBMITTING = "SUBMITTING"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    REJECTED_BY_POLICY = "REJECTED_BY_POLICY"


_S = WorkflowState

TERMINAL: frozenset[WorkflowState] = frozenset(
    {
        _S.CONFIRMED,
        _S.FAILED,
        _S.USER_REJECTED,
        _S.CANCELLED,
        _S.REJECTED_BY_POLICY,
        _S.SUBMISSION_UNKNOWN,
    }
)

# Explicit allowed transitions. CANCELLED is handled separately: the user may
# cancel from any non-terminal state, so it is not listed per-source here.
TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    _S.IDLE: frozenset({_S.UNDERSTANDING_INTENT}),
    _S.UNDERSTANDING_INTENT: frozenset(
        {_S.NEEDS_CLARIFICATION, _S.COLLECTING_STATE, _S.REJECTED_BY_POLICY}
    ),
    _S.NEEDS_CLARIFICATION: frozenset({_S.UNDERSTANDING_INTENT, _S.COLLECTING_STATE}),
    _S.COLLECTING_STATE: frozenset(
        {_S.PLANNING, _S.NEEDS_CLARIFICATION, _S.REJECTED_BY_POLICY}
    ),
    _S.PLANNING: frozenset({_S.PLAN_READY, _S.NEEDS_CLARIFICATION, _S.REJECTED_BY_POLICY}),
    _S.PLAN_READY: frozenset({_S.SIMULATING, _S.REJECTED_BY_POLICY}),
    _S.SIMULATING: frozenset(
        {
            _S.AWAITING_CONFIRMATION,
            _S.SIMULATION_FAILED,
            _S.SIMULATION_MISMATCH,
            _S.QUOTE_EXPIRED,
            _S.REJECTED_BY_POLICY,
        }
    ),
    _S.SIMULATION_FAILED: frozenset({_S.PLANNING, _S.FAILED}),
    _S.SIMULATION_MISMATCH: frozenset({_S.PLANNING, _S.USER_REJECTED}),
    _S.QUOTE_EXPIRED: frozenset({_S.PLANNING}),
    _S.AWAITING_CONFIRMATION: frozenset(
        {_S.READY_TO_SIGN, _S.SIMULATING, _S.USER_REJECTED, _S.QUOTE_EXPIRED}
    ),
    # READY_TO_SIGN may fall back to SIMULATING on staleness/approval invalidation (C1).
    _S.READY_TO_SIGN: frozenset({_S.SIMULATING, _S.SUBMITTING, _S.USER_REJECTED, _S.QUOTE_EXPIRED}),
    _S.SUBMITTING: frozenset({_S.SUBMITTED, _S.FAILED}),
    _S.SUBMITTED: frozenset({_S.CONFIRMED, _S.FAILED}),
    # Terminal states have no outgoing transitions.
    _S.USER_REJECTED: frozenset(),
    _S.CANCELLED: frozenset(),
    _S.REJECTED_BY_POLICY: frozenset(),
    _S.SUBMISSION_UNKNOWN: frozenset(),
    _S.CONFIRMED: frozenset(),
    _S.FAILED: frozenset(),
}


class TransitionError(RuntimeError):
    """Raised when an illegal state transition is attempted."""


class StateMachine:
    def __init__(self, state: WorkflowState = WorkflowState.IDLE) -> None:
        self.state = state
        self.history: list[WorkflowState] = [state]
        self._submission_guard: object | None = None

    def bind_submission_guard(self, guard: object) -> None:
        """Bind the one deterministic component permitted to enter SUBMITTING.

        ``SUBMITTING`` is not a normal model- or UI-selectable transition.  The
        approval guard binds itself once during workflow construction and must
        re-check approval freshness before using the private transition below.
        """

        if self._submission_guard is not None:
            raise TransitionError("a submission guard is already bound")
        self._submission_guard = guard

    def allowed(self, target: WorkflowState) -> bool:
        if self.state in TERMINAL:
            return False
        if target is WorkflowState.CANCELLED:
            return True  # user may cancel from any non-terminal state
        if target is WorkflowState.SUBMITTING:
            return False  # only ApprovalGuard may perform this guarded edge
        return target in TRANSITIONS.get(self.state, frozenset())

    def transition(self, target: WorkflowState) -> WorkflowState:
        if not self.allowed(target):
            raise TransitionError(
                f"illegal transition {self.state.value} -> {target.value}"
            )
        self.state = target
        self.history.append(target)
        return self.state

    def _enter_submitting(self, guard: object) -> WorkflowState:
        """Guard-owned READY_TO_SIGN -> SUBMITTING transition.

        Kept private so only the approval subsystem reaches the signing handoff;
        callers must never treat a state transition as authorization.
        """

        if guard is not self._submission_guard:
            raise TransitionError("SUBMITTING requires the bound approval guard")
        if self.state is not WorkflowState.READY_TO_SIGN:
            raise TransitionError(
                f"illegal transition {self.state.value} -> {WorkflowState.SUBMITTING.value}"
            )
        self.state = WorkflowState.SUBMITTING
        self.history.append(self.state)
        return self.state

    def _invalidate_submitting(self, guard: object) -> WorkflowState:
        """Guard-owned signer-freshness rejection requiring re-simulation."""

        if guard is not self._submission_guard:
            raise TransitionError("submission invalidation requires the bound approval guard")
        if self.state is not WorkflowState.SUBMITTING:
            raise TransitionError(
                f"illegal transition {self.state.value} -> {WorkflowState.SIMULATING.value}"
            )
        self.state = WorkflowState.SIMULATING
        self.history.append(self.state)
        return self.state

    def _mark_submission_unknown(self, guard: object) -> WorkflowState:
        """Guard-owned terminal edge for an ambiguous post-sign broadcast."""

        if guard is not self._submission_guard:
            raise TransitionError("unknown submission requires the bound approval guard")
        if self.state is not WorkflowState.SUBMITTING:
            raise TransitionError(
                f"illegal transition {self.state.value} -> "
                f"{WorkflowState.SUBMISSION_UNKNOWN.value}"
            )
        self.state = WorkflowState.SUBMISSION_UNKNOWN
        self.history.append(self.state)
        return self.state
