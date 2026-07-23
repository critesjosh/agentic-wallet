import pytest

from agentic_wallet.state_machine import (
    TERMINAL,
    TRANSITIONS,
    StateMachine,
    TransitionError,
)
from agentic_wallet.state_machine import WorkflowState as S


def test_normal_workflow_path_reaches_ready_to_sign():
    sm = StateMachine()
    for nxt in [
        S.UNDERSTANDING_INTENT,
        S.COLLECTING_STATE,
        S.PLANNING,
        S.PLAN_READY,
        S.SIMULATING,
        S.AWAITING_CONFIRMATION,
        S.READY_TO_SIGN,
    ]:
        sm.transition(nxt)
    assert sm.state is S.READY_TO_SIGN


def test_submitting_cannot_be_entered_by_an_arbitrary_transition():
    sm = StateMachine(S.READY_TO_SIGN)
    assert not sm.allowed(S.SUBMITTING)
    with pytest.raises(TransitionError, match="READY_TO_SIGN -> SUBMITTING"):
        sm.transition(S.SUBMITTING)


def test_planning_cannot_jump_to_ready_to_sign():
    sm = StateMachine(S.PLANNING)
    assert not sm.allowed(S.READY_TO_SIGN)
    with pytest.raises(TransitionError):
        sm.transition(S.READY_TO_SIGN)


def test_cancel_allowed_from_any_non_terminal_only():
    for st in S:
        sm = StateMachine(st)
        assert sm.allowed(S.CANCELLED) is (st not in TERMINAL)


def test_quote_expired_requotes_to_planning():
    sm = StateMachine(S.AWAITING_CONFIRMATION)
    sm.transition(S.QUOTE_EXPIRED)
    assert sm.allowed(S.PLANNING)


def test_ready_to_sign_can_resimulate_on_staleness():
    sm = StateMachine(S.READY_TO_SIGN)
    assert sm.allowed(S.SIMULATING)  # C1 approval-invalidation / staleness


def test_simulation_mismatch_never_signs():
    sm = StateMachine(S.SIMULATION_MISMATCH)
    assert not sm.allowed(S.READY_TO_SIGN)


def test_terminal_states_have_no_exits():
    for st in TERMINAL:
        assert TRANSITIONS[st] == frozenset()
        assert not StateMachine(st).allowed(S.PLANNING)


def test_ambiguous_submission_is_terminal_and_non_retryable():
    assert S.SUBMISSION_UNKNOWN in TERMINAL
    sm = StateMachine(S.SUBMISSION_UNKNOWN)
    assert not sm.allowed(S.SIMULATING)
    assert not sm.allowed(S.SUBMITTING)
