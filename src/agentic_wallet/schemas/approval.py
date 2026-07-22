"""The approval envelope (plan.md C1).

User approval binds to an immutable digest of the exact plan, simulation,
policy, expiry, state anchor, and nonce. Any change invalidates approval and
forces re-simulation before signing.
"""

from __future__ import annotations

from pydantic import ConfigDict, model_validator

from ..digest import canonical_digest
from .common import StrictModel
from .policy import PolicyResult
from .simulation_result import SimulationResult
from .signing import Eip1559Transaction
from .transaction_plan import TransactionPlan


class ApprovalEnvelope(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chain_id: int
    plan: TransactionPlan
    simulation: SimulationResult
    policy: PolicyResult
    expires_at: int  # unix seconds; freshness window
    state_anchor: str  # block hash/number the plan and simulation were built against
    nonce: int
    signing_transaction: Eip1559Transaction | None = None

    @model_validator(mode="after")
    def _bound_fields_agree(self) -> "ApprovalEnvelope":
        if self.chain_id != self.plan.chain_id:
            raise ValueError("envelope and plan chain IDs must match")
        if self.simulation.plan_id != self.plan.plan_id:
            raise ValueError("simulation must be for the enclosed plan")
        transaction = self.signing_transaction
        if transaction is not None:
            if transaction.chain_id != self.chain_id:
                raise ValueError("signing transaction chain must match envelope")
            if int(transaction.nonce) != self.nonce:
                raise ValueError("signing transaction nonce must match envelope")
            if transaction.from_address.lower() != self.plan.from_address.lower():
                raise ValueError("signing transaction sender must match plan")
            if transaction.to_address.lower() != self.plan.to_address.lower():
                raise ValueError("signing transaction target must match plan")
            if transaction.value != self.plan.value.base_units:
                raise ValueError("signing transaction value must match plan")
            if transaction.data.lower() != self.plan.calldata.lower():
                raise ValueError("signing transaction data must match plan")
        return self

    def digest(self) -> str:
        """Deterministic digest over the whole envelope."""
        return canonical_digest(self.model_dump(mode="json"))
