from pathlib import Path

import pytest

from agentic_wallet.approval_guard import ApprovalGuard, ApprovalInvalidated
from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.planning import build_transfer_plan
from agentic_wallet.policy_engine import WalletPolicy, evaluate_policy
from agentic_wallet.registry import BASE_REGISTRY
from agentic_wallet.schemas.approval import ApprovalEnvelope
from agentic_wallet.schemas.common import Amount
from agentic_wallet.simulation import simulate_plan
from agentic_wallet.state_machine import StateMachine, WorkflowState


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
WALLET = "0x1111111111111111111111111111111111111111"
RECIPIENT = "0x3333333333333333333333333333333333333333"


def _envelope():
    harness = MockReadOnlyHarness.from_fixture(FIXTURE)
    plan = build_transfer_plan(
        harness,
        asset_id="base:usdc",
        amount=Amount(base_units="25000000", decimals=6),
        recipient=RECIPIENT,
        gas_reserve=Amount(base_units="50000000000000", decimals=18),
    )
    simulation = simulate_plan(
        plan,
        block=21000001,
        gas_used=65000,
        gas_fee=Amount(base_units="21000000000000", decimals=18),
    )
    policy = evaluate_policy(
        plan,
        simulation,
        policy=WalletPolicy(chain_id=8453, wallet_address=WALLET),
        registry=BASE_REGISTRY,
        now=1900,
    )
    return ApprovalEnvelope(
        chain_id=8453,
        plan=plan,
        simulation=simulation,
        policy=policy,
        expires_at=1950,
        state_anchor="base:21000001:0xabc",
        nonce=7,
    )


def test_exact_approved_envelope_remains_current():
    envelope = _envelope()
    guard = ApprovalGuard()
    approved = guard.record_explicit_approval(envelope, now=1900)
    state = StateMachine(WorkflowState.READY_TO_SIGN)

    guard.require_current(
        envelope,
        now=1901,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        state_machine=state,
    )

    assert approved == envelope.digest()
    assert state.state is WorkflowState.READY_TO_SIGN


@pytest.mark.parametrize(
    ("mutation", "kwargs", "message"),
    [
        ({"expires_at": 1960}, {}, "digest changed"),
        ({}, {"now": 1950}, "expired"),
        ({}, {"state_anchor": "base:21000002:0xdef"}, "state anchor"),
        ({}, {"nonce": 8}, "nonce"),
    ],
)
def test_mutation_or_state_drift_invalidates_and_forces_resimulation(
    mutation, kwargs, message
):
    original = _envelope()
    guard = ApprovalGuard()
    guard.record_explicit_approval(original, now=1900)
    envelope = original.model_copy(update=mutation)
    state = StateMachine(WorkflowState.READY_TO_SIGN)
    current = {
        "now": 1901,
        "state_anchor": "base:21000001:0xabc",
        "nonce": 7,
        **kwargs,
    }

    with pytest.raises(ApprovalInvalidated, match=message):
        guard.require_current(envelope, state_machine=state, **current)

    assert state.state is WorkflowState.SIMULATING
    assert guard.approved_digest is None


def test_expired_envelope_cannot_be_recorded_as_approved():
    with pytest.raises(ApprovalInvalidated, match="already expired"):
        ApprovalGuard().record_explicit_approval(_envelope(), now=1950)
