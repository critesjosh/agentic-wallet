from agentic_wallet.schemas.approval import ApprovalEnvelope
from agentic_wallet.schemas.common import Amount
from agentic_wallet.schemas.policy import PolicyResult
from agentic_wallet.schemas.simulation_result import SimulationResult
from agentic_wallet.schemas.transaction_plan import TransactionPlan


def _envelope(nonce: int = 1, expires: int = 1000) -> ApprovalEnvelope:
    plan = TransactionPlan(
        plan_id="p1",
        chain_id=8453,
        kind="transfer",
        from_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        to_address="0xdddddddddddddddddddddddddddddddddddddddd",
        asset_id="base:usdc",
        value=Amount(base_units="0", decimals=18),
    )
    sim = SimulationResult(plan_id="p1", success=True, block=100, gas_used=21000)
    return ApprovalEnvelope(
        chain_id=8453,
        plan=plan,
        simulation=sim,
        policy=PolicyResult(allowed=True),
        expires_at=expires,
        state_anchor="0xblockhash",
        nonce=nonce,
    )


def test_digest_is_deterministic():
    assert _envelope().digest() == _envelope().digest()


def test_digest_changes_on_field_mutation():
    assert _envelope(nonce=1).digest() != _envelope(nonce=2).digest()


def test_digest_changes_on_expiry_change():
    assert _envelope(expires=1000).digest() != _envelope(expires=2000).digest()


def test_digest_is_algorithm_prefixed():
    assert _envelope().digest().startswith("sha256:")
