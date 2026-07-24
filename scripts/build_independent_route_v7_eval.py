"""Materialize the disjoint V7 routing suite.

Authored routing cases with novel wording and novel addresses are compiled into
the exact V7 production contract context, so the only difference from the
training distribution is surface form. A disjointness check fails closed if any
request text or identifier also appears in the V7 training data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from agentic_wallet.training import CoverageDimensions, TrainingExample
from agentic_wallet.web.chat import _BASE_MODEL_ACTIONS, _TRANSFER_MODEL_ACTIONS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "benchmark" / "independent-route-v7.source.json"
OUTPUT = ROOT / "data" / "benchmark" / "independent-route-v7.jsonl"
MANIFEST = OUTPUT.with_suffix(".manifest.json")
TRAINING = ROOT / "data" / "training" / "sft-v7-account-identity.jsonl"

LIVE_ACTIONS = [*_BASE_MODEL_ACTIONS, *_TRANSFER_MODEL_ACTIONS]
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40,64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger(state: str) -> dict:
    return {
        "workflow_state": state,
        "chain_id": 8453,
        "resolved_intent": {
            "chain_id": 8453,
            "asset_id": "base:native",
            "amount": None,
            "amount_base_units": None,
            "recipient": None,
        },
        "missing_fields": [],
        "active_plan_id": None,
        "active_quote_id": None,
        "corrections": [],
        "verified_facts": [],
        "prior_proposals": [],
        "recent_messages": [],
    }


def _context(case: dict) -> dict:
    state = case.get("state", "IDLE")
    transfer = case.get("transfer_candidate")
    status = case.get("status_candidate")
    context = {
        "user_request": case["request"],
        "phase": "route_dialogue",
        "chain_id": 8453,
        "canonical_asset_ids": ["base:native", "base:usdc", "base:weth"],
        "transaction_review_enabled": True,
        "read_only": False,
        "conversation_ledger": _ledger(state),
        "parsed_native_transfer_candidate": (
            {**transfer, "provenance": "exact_current_user_message"}
            if transfer
            else None
        ),
        "parsed_transaction_status_candidate": (
            {**status, "provenance": "exact_current_user_message"}
            if status
            else None
        ),
    }
    if case.get("untrusted"):
        context["untrusted_data"] = case["untrusted"]
    return context


def _assert_disjoint(cases: list[dict]) -> None:
    """Fail closed if any request or address also appears in training."""

    training_text = TRAINING.read_text().casefold()
    for case in cases:
        request = case["request"].casefold()
        # A whole-request match would mean a copied training utterance.
        if request in training_text:
            raise ValueError(f"case {case['id']} request appears in training data")
        for address in _ADDRESS.findall(json.dumps(case)):
            if address.casefold() in training_text:
                raise ValueError(
                    f"case {case['id']} reuses training identifier {address}"
                )


def main() -> None:
    payload_in = json.loads(SOURCE.read_text())
    cases = payload_in["cases"]
    if len(cases) != 40:
        raise ValueError("disjoint V7 suite must contain exactly 40 cases")
    _assert_disjoint(cases)

    examples: list[TrainingExample] = []
    for index, case in enumerate(cases, 1):
        if case["id"] != f"{index:03d}":
            raise ValueError("case IDs must be contiguous")
        expected = case["expected"]
        examples.append(
            TrainingExample(
                id=f"sft-independent-v7-{case['id']}",
                kind="dialogue_route",
                scenario_class=f"independent-v7-route-{case['id']}",
                context=_context(case),
                available_actions=list(LIVE_ACTIONS),
                target={"proposed_action": expected},
                split="validation",
                action_exposure="production",
                coverage=CoverageDimensions(
                    workflow_state=case.get("state", "IDLE"),
                    intended_action=expected if expected != "none" else "none",
                    risk_category=case.get("risk", "none"),
                    conversational_intent=(
                        "propose_tool" if expected != "none" else "conversation"
                    ),
                    adversarial_condition=(
                        case["risk"] if case.get("risk") else "none"
                    ),
                ),
            )
        )

    body = "".join(
        json.dumps(example.model_dump(), sort_keys=True, separators=(",", ":")) + "\n"
        for example in examples
    )
    OUTPUT.write_text(body)
    MANIFEST.write_text(
        json.dumps(
            {
                "suite": "independent-route-v7",
                "role": "development-evaluation-only-not-sealed",
                "author": "assistant, disjoint from the curriculum generator",
                "case_count": len(examples),
                "source_sha256": _sha256(SOURCE),
                "dataset_sha256": _sha256(OUTPUT),
                "disjoint_from": "sft-v7-account-identity",
                "release_claim_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["expected"]] = counts.get(case["expected"], 0) + 1
    print(json.dumps({"cases": len(examples), "by_expected_action": counts}, indent=2))


if __name__ == "__main__":
    main()
