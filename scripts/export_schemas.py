"""Emit JSON Schema for each pydantic model into /schemas.

The pydantic models are the source of truth; these JSON files are generated
artifacts for cross-language consumers. Run: ``python scripts/export_schemas.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_wallet.schemas.approval import ApprovalEnvelope
from agentic_wallet.schemas.dialogue import ModelDialogueTurn
from agentic_wallet.schemas.intent import Intent
from agentic_wallet.schemas.policy import PolicyResult
from agentic_wallet.schemas.quote import SwapQuote
from agentic_wallet.schemas.portfolio import Portfolio
from agentic_wallet.schemas.simulation_result import SimulationResult
from agentic_wallet.schemas.signing import Eip1559Transaction, SignedTransactionResult
from agentic_wallet.schemas.tool_call import ToolCall
from agentic_wallet.schemas.transaction_plan import TransactionPlan

OUT = Path(__file__).resolve().parents[1] / "schemas"

MODELS = {
    "intent": Intent,
    "tool-call": ToolCall,
    "model-dialogue-turn": ModelDialogueTurn,
    "portfolio": Portfolio,
    "transaction-plan": TransactionPlan,
    "simulation-result": SimulationResult,
    "policy-result": PolicyResult,
    "swap-quote": SwapQuote,
    "eip1559-transaction": Eip1559Transaction,
    "signed-transaction-result": SignedTransactionResult,
    "approval-envelope": ApprovalEnvelope,
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, model in MODELS.items():
        path = OUT / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        print("wrote", path.relative_to(OUT.parent))


if __name__ == "__main__":
    main()
