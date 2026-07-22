from __future__ import annotations

import json
from pathlib import Path

from agentic_wallet.training import load_training_examples

DATA = Path(__file__).resolve().parents[1] / "data"


def test_independent_route_suite_is_frozen_and_training_isolated() -> None:
    suite_path = DATA / "benchmark" / "independent-route-v1.jsonl"
    manifest = json.loads(
        suite_path.with_suffix(".manifest.json").read_text()
    )
    suite = load_training_examples(suite_path)
    training = load_training_examples(
        DATA / "training" / "sft-v5-candidate-binding.jsonl"
    )

    assert len(suite) == 40
    assert manifest["case_count"] == 40
    assert manifest["authored_before_v5_training"] is True
    assert manifest["release_claim_eligible"] is False
    assert all(item.kind == "dialogue_route" for item in suite)
    assert all(set(item.target) == {"proposed_action"} for item in suite)
    assert all(item.split == "validation" for item in suite)
    suite_requests = {item.context["user_request"] for item in suite}
    training_requests = {item.context.get("user_request") for item in training}
    assert suite_requests.isdisjoint(training_requests)
    assert any(
        item.coverage.adversarial_condition != "none" for item in suite
    )
