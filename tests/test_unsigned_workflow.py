from pathlib import Path

import pytest

from agentic_wallet.approval_guard import ApprovalInvalidated
from agentic_wallet.candidate_binding import prepare_inference_context
from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.inference import ScriptedProvider
from agentic_wallet.policy_engine import WalletPolicy
from agentic_wallet.schemas.common import Amount
from agentic_wallet.schemas.simulation_result import BalanceChange
from agentic_wallet.state_machine import WorkflowState
from agentic_wallet.unsigned_workflow import (
    UnsignedTransactionWorkflow,
    UnsignedWorkflowError,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
WALLET = "0x1111111111111111111111111111111111111111"
RECIPIENT = "0x3333333333333333333333333333333333333333"
GAS_RESERVE = Amount(base_units="50000000000000", decimals=18)
GAS_FEE = Amount(base_units="21000000000000", decimals=18)


def _workflow():
    harness = MockReadOnlyHarness.from_fixture(FIXTURE)
    policy = WalletPolicy(chain_id=8453, wallet_address=WALLET)
    return UnsignedTransactionWorkflow(harness, policy, approval_ttl_seconds=60)


def _plan(workflow):
    return workflow.plan_transfer(
        asset_id="base:usdc",
        amount=Amount(base_units="25000000", decimals=6),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )


def test_candidate_bound_transfer_enters_normal_unsigned_workflow():
    workflow = _workflow()
    context = prepare_inference_context(
        {
            "user_request": (
                "Draft 25000000 base units of USDC to " + RECIPIENT
            ),
            "chain_id": 8453,
            "canonical_asset_ids": ["base:native", "base:usdc", "base:weth"],
        }
    )
    provider = ScriptedProvider({})
    call = provider.propose_tool_call_with_repair(
        context, "create_transfer_plan_from_candidate"
    )

    plan = workflow.plan_transfer_from_candidate(
        call=call,
        context=context,
        gas_reserve=GAS_RESERVE,
    )
    assert plan.recipient_address == RECIPIENT
    assert plan.expected_outgoing[0].amount.base_units == "25000000"
    assert workflow.state_machine.state is WorkflowState.PLAN_READY


def test_unsigned_happy_path_stops_at_wallet_handoff_boundary():
    workflow = _workflow()
    _plan(workflow)
    envelope = workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        gas_used=65000,
        gas_fee=GAS_FEE,
    )
    assert envelope is not None
    assert workflow.state_machine.state is WorkflowState.AWAITING_CONFIRMATION

    workflow.record_user_approval(envelope.digest(), now=1001)
    digest = workflow.require_current_approval(
        now=1001, state_anchor="base:21000001:0xabc", nonce=7
    )

    assert digest == envelope.digest()
    assert workflow.state_machine.state is WorkflowState.READY_TO_SIGN
    for forbidden in ("sign", "submit", "send_transaction"):
        assert not hasattr(workflow, forbidden)


def test_policy_failure_never_reaches_confirmation():
    workflow = _workflow()
    plan = _plan(workflow)
    observed = [
        BalanceChange(asset_id="base:usdc", delta_base_units="-25000000"),
        BalanceChange(asset_id="base:native", delta_base_units="-21000000000000"),
        BalanceChange(asset_id="base:weth", delta_base_units="-1"),
    ]
    assert workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        gas_used=65000,
        gas_fee=GAS_FEE,
        observed_changes=observed,
    ) is None

    assert plan.plan_id == workflow.plan.plan_id
    assert workflow.state_machine.state is WorkflowState.REJECTED_BY_POLICY


def test_wrong_display_digest_cannot_record_approval():
    workflow = _workflow()
    _plan(workflow)
    workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        gas_used=65000,
        gas_fee=GAS_FEE,
    )

    with pytest.raises(ApprovalInvalidated, match="exact envelope"):
        workflow.record_user_approval("sha256:wrong", now=1001)
    assert workflow.state_machine.state is WorkflowState.AWAITING_CONFIRMATION


def test_nonce_drift_forces_resimulation_and_prevents_reuse():
    workflow = _workflow()
    _plan(workflow)
    envelope = workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        gas_used=65000,
        gas_fee=GAS_FEE,
    )
    workflow.record_user_approval(envelope.digest(), now=1001)

    with pytest.raises(ApprovalInvalidated, match="nonce changed"):
        workflow.require_current_approval(
            now=1001, state_anchor="base:21000001:0xabc", nonce=8
        )
    assert workflow.state_machine.state is WorkflowState.SIMULATING
    with pytest.raises(UnsignedWorkflowError, match="READY_TO_SIGN"):
        workflow.require_current_approval(
            now=1001, state_anchor="base:21000001:0xabc", nonce=7
        )


def test_invalidation_can_resimulate_and_receive_fresh_approval():
    workflow = _workflow()
    _plan(workflow)
    first = workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        gas_used=65000,
        gas_fee=GAS_FEE,
    )
    workflow.record_user_approval(first.digest(), now=1001)

    with pytest.raises(ApprovalInvalidated, match="nonce changed"):
        workflow.require_current_approval(
            now=1001, state_anchor="base:21000002:0xdef", nonce=8
        )

    second = workflow.simulate_and_check(
        now=1002,
        block=21000002,
        state_anchor="base:21000002:0xdef",
        nonce=8,
        gas_used=65000,
        gas_fee=GAS_FEE,
    )
    assert second is not None
    assert second.digest() != first.digest()
    assert workflow.state_machine.state is WorkflowState.AWAITING_CONFIRMATION

    workflow.record_user_approval(second.digest(), now=1003)
    assert workflow.require_current_approval(
        now=1003, state_anchor="base:21000002:0xdef", nonce=8
    ) == second.digest()
