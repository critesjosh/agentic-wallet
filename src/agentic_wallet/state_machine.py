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
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    REJECTED_BY_POLICY = "REJECTED_BY_POLICY"


_S = WorkflowState

TERMINAL: frozenset[WorkflowState] = frozenset(
    {_S.CONFIRMED, _S.FAILED, _S.USER_REJECTED, _S.CANCELLED, _S.REJECTED_BY_POLICY}
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
    _S.AWAITING_CONFIRMATION: frozenset({_S.READY_TO_SIGN, _S.USER_REJECTED, _S.QUOTE_EXPIRED}),
    # READY_TO_SIGN may fall back to SIMULATING on staleness/approval invalidation (C1).
    _S.READY_TO_SIGN: frozenset({_S.SIMULATING, _S.SUBMITTING, _S.USER_REJECTED, _S.QUOTE_EXPIRED}),
    _S.SUBMITTING: frozenset({_S.SUBMITTED, _S.FAILED}),
    _S.SUBMITTED: frozenset({_S.CONFIRMED, _S.FAILED}),
    # Terminal states have no outgoing transitions.
    _S.USER_REJECTED: frozenset(),
    _S.CANCELLED: frozenset(),
    _S.REJECTED_BY_POLICY: frozenset(),
    _S.CONFIRMED: frozenset(),
    _S.FAILED: frozenset(),
}


class TransitionError(RuntimeError):
    """Raised when an illegal state transition is attempted."""


class StateMachine:
    def __init__(self, state: WorkflowState = WorkflowState.IDLE) -> None:
        self.state = state
        self.history: list[WorkflowState] = [state]

    def allowed(self, target: WorkflowState) -> bool:
        if self.state in TERMINAL:
            return False
        if target is WorkflowState.CANCELLED:
            return True  # user may cancel from any non-terminal state
        return target in TRANSITIONS.get(self.state, frozenset())

    def transition(self, target: WorkflowState) -> WorkflowState:
        if not self.allowed(target):
            raise TransitionError(
                f"illegal transition {self.state.value} -> {target.value}"
            )
        self.state = target
        self.history.append(target)
        return self.state
