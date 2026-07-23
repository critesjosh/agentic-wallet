"""Phase 8 deterministic approval bindings and freshness gates."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_wallet.approval_guard import ApprovalInvalidated
from agentic_wallet.harness import MockReadOnlyHarness
from agentic_wallet.planning import build_eip1559_transaction, build_transfer_plan
from agentic_wallet.policy_engine import WalletPolicy
from agentic_wallet.registry import Registry, RegistryEntry
from agentic_wallet.schemas.common import Amount
from agentic_wallet.schemas.policy import PolicyResult
from agentic_wallet.schemas.signing import AccessListEntry, Eip1559Transaction
from agentic_wallet.simulation import expected_balance_changes
from agentic_wallet.signer_outcome import (
    SignerOutcome,
    SignerOutcomeCode,
    SignerOutcomeStatus,
)
from agentic_wallet.state_machine import WorkflowState
from agentic_wallet.unsigned_workflow import UnsignedTransactionWorkflow


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
WALLET = "0x1111111111111111111111111111111111111111"
RECIPIENT = "0x3333333333333333333333333333333333333333"
GAS_RESERVE = Amount(base_units="50000000000000", decimals=18)
GAS_FEE = Amount(base_units="21000000000000", decimals=18)
SNAPSHOT_BLOCK_HASH = "0x" + "ab" * 32


def _phase8_workflow() -> tuple[UnsignedTransactionWorkflow, Eip1559Transaction]:
    workflow = UnsignedTransactionWorkflow(
        MockReadOnlyHarness.from_fixture(FIXTURE),
        WalletPolicy(chain_id=8453, wallet_address=WALLET),
    )
    plan = workflow.plan_transfer(
        asset_id="base:native",
        amount=Amount(base_units="1000000000000000", decimals=18),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )
    return workflow, build_eip1559_transaction(
        plan,
        nonce=7,
        gas_limit=21000,
        max_priority_fee_per_gas=1,
        max_fee_per_gas=2,
    )


def _phase8_envelope():
    workflow, transaction = _phase8_workflow()
    envelope = workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        snapshot_block_hash=SNAPSHOT_BLOCK_HASH,
        nonce=7,
        gas_used=21000,
        gas_fee=GAS_FEE,
        signing_transaction=transaction,
    )
    assert envelope is not None
    return workflow, transaction, envelope


def test_phase8_envelope_binds_full_preimage_registry_and_simulation_digest():
    workflow, transaction, envelope = _phase8_envelope()

    assert envelope.is_phase_eight_bound
    assert envelope.signing_transaction == transaction
    assert envelope.simulated_transaction_digest == transaction.digest()
    assert envelope.simulation.transaction_digest == transaction.digest()
    assert envelope.registry_digest == workflow.registry.version_digest()
    assert envelope.snapshot_block_hash == SNAPSHOT_BLOCK_HASH


def test_phase8_envelope_digest_binds_snapshot_block_hash():
    _, _, envelope = _phase8_envelope()
    changed = envelope.model_copy(
        update={"snapshot_block_hash": "0x" + "cd" * 32}
    )

    assert changed.digest() != envelope.digest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("chain_id", 1),
        ("nonce", "8"),
        ("max_priority_fee_per_gas", "2"),
        ("max_fee_per_gas", "3"),
        ("gas_limit", "22000"),
        ("from_address", "0x2222222222222222222222222222222222222222"),
        ("to_address", "0x4444444444444444444444444444444444444444"),
        ("value", "1000000000000001"),
        ("data", "0x01"),
        (
            "access_list",
            [
                AccessListEntry(
                    address="0x5555555555555555555555555555555555555555",
                    storage_keys=[],
                )
            ],
        ),
    ],
)
def test_every_eip1559_preimage_mutation_breaks_envelope_binding(field, value):
    _, transaction, envelope = _phase8_envelope()
    payload = transaction.model_dump()
    payload[field] = value
    mutated = Eip1559Transaction.model_validate(payload)

    with pytest.raises(ValidationError):
        envelope.model_validate(
            {
                **envelope.model_dump(),
                "signing_transaction": mutated.model_dump(),
            }
        )


def test_registry_drift_invalidates_approval_and_forces_resimulation():
    workflow, _, envelope = _phase8_envelope()
    workflow.record_user_approval(envelope.digest(), now=1001)

    with pytest.raises(ApprovalInvalidated, match="registry changed"):
        workflow.approval_guard.require_current(
            envelope,
            now=1001,
            state_anchor="base:21000001:0xabc",
            nonce=7,
            registry_digest="sha256:" + "0" * 64,
            state_machine=workflow.state_machine,
        )

    assert workflow.state_machine.state is WorkflowState.SIMULATING
    assert workflow.approval_guard.approved_digest is None


def test_expiry_while_waiting_for_confirmation_forces_resimulation():
    workflow, _, envelope = _phase8_envelope()

    with pytest.raises(ApprovalInvalidated, match="already expired"):
        workflow.record_user_approval(envelope.digest(), now=envelope.expires_at)

    assert workflow.state_machine.state is WorkflowState.SIMULATING


def test_guard_alone_can_enter_submitting_after_fresh_phase8_approval():
    workflow, _, envelope = _phase8_envelope()
    workflow.record_user_approval(envelope.digest(), now=1001)

    assert workflow.authorize_submission(
        now=1001, state_anchor="base:21000001:0xabc", nonce=7
    ) == envelope.digest()
    assert workflow.state_machine.state is WorkflowState.SUBMITTING


def _submitting_workflow():
    workflow, _, envelope = _phase8_envelope()
    workflow.record_user_approval(envelope.digest(), now=1001)
    workflow.authorize_submission(
        now=1001, state_anchor="base:21000001:0xabc", nonce=7
    )
    return workflow, envelope


def test_structured_signer_freshness_rejection_clears_approval_and_resimulates():
    workflow, envelope = _submitting_workflow()
    outcome = SignerOutcome(
        status=SignerOutcomeStatus.RESIMULATION_REQUIRED,
        code=SignerOutcomeCode.PENDING_NONCE_CHANGED,
        envelope_digest=envelope.digest(),
        from_address=WALLET,
    )

    assert workflow.handle_signer_outcome(outcome) is WorkflowState.SIMULATING
    assert workflow.approval_guard.approved_digest is None
    assert workflow.state_machine.history[-2:] == [
        WorkflowState.SUBMITTING,
        WorkflowState.SIMULATING,
    ]


def test_ambiguous_submission_is_terminal_and_has_no_retry_path():
    workflow, envelope = _submitting_workflow()
    outcome = SignerOutcome(
        status=SignerOutcomeStatus.UNKNOWN,
        code=SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN,
        envelope_digest=envelope.digest(),
        from_address=WALLET,
        transaction_hash="0x" + "1" * 64,
        transaction_signing_hash="0x" + "2" * 64,
    )

    assert (
        workflow.handle_signer_outcome(outcome)
        is WorkflowState.SUBMISSION_UNKNOWN
    )
    assert workflow.approval_guard.approved_digest is None
    assert not workflow.state_machine.allowed(WorkflowState.SIMULATING)
    assert not workflow.state_machine.allowed(WorkflowState.SUBMITTING)


def test_submitted_signer_outcome_completes_submission_handoff():
    workflow, envelope = _submitting_workflow()
    outcome = SignerOutcome(
        status=SignerOutcomeStatus.SUBMITTED,
        code=SignerOutcomeCode.SUBMITTED,
        envelope_digest=envelope.digest(),
        from_address=WALLET,
        transaction_hash="0x" + "1" * 64,
        transaction_signing_hash="0x" + "2" * 64,
    )

    assert workflow.handle_signer_outcome(outcome) is WorkflowState.SUBMITTED
    assert workflow.approval_guard.approved_digest is None


def test_phase8_submission_rejects_legacy_unsigned_envelope():
    workflow = UnsignedTransactionWorkflow(
        MockReadOnlyHarness.from_fixture(FIXTURE),
        WalletPolicy(chain_id=8453, wallet_address=WALLET),
    )
    workflow.plan_transfer(
        asset_id="base:usdc",
        amount=Amount(base_units="25000000", decimals=6),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )
    envelope = workflow.simulate_and_check(
        now=1000,
        block=21000001,
        state_anchor="base:21000001:0xabc",
        nonce=7,
        gas_used=65000,
        gas_fee=GAS_FEE,
    )
    assert envelope is not None
    workflow.record_user_approval(envelope.digest(), now=1001)

    with pytest.raises(ApprovalInvalidated, match="bound EIP-1559 preimage"):
        workflow.authorize_submission(
            now=1001, state_anchor="base:21000001:0xabc", nonce=7
        )
    assert workflow.state_machine.state is WorkflowState.READY_TO_SIGN


def test_native_asset_identity_comes_from_the_registry_not_a_literal():
    registry = Registry(
        [
            RegistryEntry("example:coin", 8453, "native", "EX", 18, is_native=True),
            RegistryEntry(
                "example:token",
                8453,
                "0x6666666666666666666666666666666666666666",
                "TKN",
                6,
            ),
        ]
    )
    harness = MockReadOnlyHarness.from_fixture(FIXTURE, registry=registry)
    plan = build_transfer_plan(
        harness,
        asset_id="example:coin",
        amount=Amount(base_units="1", decimals=18),
        recipient=RECIPIENT,
        gas_reserve=GAS_RESERVE,
    )

    changes = expected_balance_changes(
        plan,
        gas_fee=GAS_FEE,
        native_asset_id=registry.native_asset(8453).asset_id,
        native_asset_decimals=18,
    )
    assert {(change.asset_id, change.delta_base_units) for change in changes} == {
        ("example:coin", "-21000000000001")
    }


def test_policy_result_cannot_allow_violations():
    with pytest.raises(ValidationError, match="cannot contain violations"):
        PolicyResult(allowed=True, violations=["wrong-chain"])
