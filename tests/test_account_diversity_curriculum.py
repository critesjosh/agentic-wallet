"""The V8 diversity-augmented account, read, and refusal curriculum."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agentic_wallet.training.account_curriculum_v8 import (
    _account_additions,
    _assert_disjoint_from_suite,
    _route,
    account_cluster_diversity,
    load_account_diversity_curriculum,
    validate_account_diversity_coverage,
)
from agentic_wallet.training.data import CoverageDimensions, validate_training_dataset
from agentic_wallet.training.transaction_curriculum import (
    load_transaction_candidate_curriculum,
)
from agentic_wallet.benchmark import load_cases

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "training" / "natural_v3_source.jsonl"
V7_FILE = ROOT / "data" / "training" / "sft-v7-account-identity.jsonl"
V8_FILE = ROOT / "data" / "training" / "sft-v8-account-diversity.jsonl"
SUITE = ROOT / "data" / "benchmark" / "independent-route-v7.source.json"


@pytest.fixture(scope="module")
def curriculum():
    return load_account_diversity_curriculum(SOURCE)


def _additions(examples):
    return [item for item in examples if item.id.startswith("sft-v8-")]


def test_curriculum_shape(curriculum):
    additions = _additions(curriculum)
    assert 50 <= len(additions) <= 100
    # The plan's target: grow the ~12-template V7 cluster many-fold.
    assert len(additions) >= 4 * 12
    assert len(curriculum) == 268 + len(additions)


def test_v6_base_is_inherited_unchanged(curriculum):
    before = {item.id: item for item in load_transaction_candidate_curriculum(SOURCE)}
    inherited = [item for item in curriculum if not item.id.startswith("sft-v8-")]
    assert len(inherited) == 268
    for item in inherited:
        original = before[item.id]
        assert item.target == original.target
        assert item.context == original.context
        assert item.coverage == original.coverage
        assert item.split == original.split


def test_routing_families_resolve_to_the_named_tool(curriculum):
    routes = {
        item.context["user_request"]: item.target["proposed_action"]
        for item in _additions(curriculum)
        if set(item.target) == {"proposed_action"}
    }
    assert routes["What's my wallet address?"] == "get_account"
    assert routes["Lay out everything I currently own."] == "get_portfolio"
    assert routes["Check just my stablecoin balance."] == "get_balance"
    assert routes["Which spenders can still move my tokens?"] == "get_allowances"
    assert routes["List the token identifiers you treat as canonical."] == "get_registry"
    assert routes["What features do you support?"] == "show_help"
    # State-changing requests this contract cannot serve must refuse explicitly.
    assert routes["Go ahead and bridge my funds to Arbitrum."] == "reject_state_changing"


def test_key_and_seed_requests_route_nowhere(curriculum):
    secrets = [
        item
        for item in _additions(curriculum)
        if item.coverage.risk_category == "key-disclosure"
    ]
    assert len(secrets) >= 8
    for item in secrets:
        assert item.target["proposed_action"] == "none"


def test_no_key_or_mnemonic_shaped_values_anywhere(curriculum):
    key_shaped = re.compile(r"(?:0x)?[0-9a-fA-F]{64}")
    mnemonic = re.compile(r"(?:\b[a-z]{3,8}\b[ ]){11,}\b[a-z]{3,8}\b")
    for item in _additions(curriculum):
        text = str(item.model_dump())
        assert not key_shaped.search(text)
        assert not mnemonic.search(text)


def test_randomized_identifiers_are_reproducible():
    """A fixed seed is what keeps the committed dataset digest stable."""

    first = [json.dumps(e.model_dump(), sort_keys=True) for e in _account_additions()]
    second = [json.dumps(e.model_dump(), sort_keys=True) for e in _account_additions()]
    assert first == second
    addresses = re.findall(r"0x[0-9a-fA-F]{40}", "".join(first))
    assert len(set(addresses)) >= 4, "identifiers should vary across examples"


def test_additions_are_disjoint_from_the_held_out_suite(curriculum):
    suite = json.loads(SUITE.read_text())
    suite_requests = {c["request"].casefold().strip() for c in suite["cases"]}
    for item in _additions(curriculum):
        request = str(item.context.get("user_request", "")).casefold().strip()
        assert request not in suite_requests


def test_disjointness_gate_rejects_a_leaked_request():
    """A poisoned utterance copied from the eval suite must fail closed."""

    leaked = _route(
        identifier="leak-00",
        split="train",
        scenario="account-route",
        request="Which wallet is this hooked up to right now?",
        proposed_action="get_account",
        coverage=CoverageDimensions(
            workflow_state="IDLE",
            intended_action="get_account",
            conversational_intent="propose_tool",
        ),
    )
    with pytest.raises(ValueError, match="disjoint suite request"):
        _assert_disjoint_from_suite([leaked])


def test_coverage_validator_rejects_a_key_request_that_routes_to_a_tool(curriculum):
    escalated = []
    for item in curriculum:
        if item.coverage.risk_category == "key-disclosure":
            item = item.model_copy(update={"target": {"proposed_action": "get_account"}})
        escalated.append(item)
    with pytest.raises(ValueError, match="must not route to a tool"):
        validate_account_diversity_coverage(escalated)


def test_coverage_validator_rejects_dropped_adversarial_coverage(curriculum):
    relabelled = []
    for item in curriculum:
        if item.coverage.adversarial_condition == "key-disclosure-request":
            item = item.model_copy(
                update={
                    "coverage": item.coverage.model_copy(
                        update={"adversarial_condition": "none"}
                    )
                }
            )
        relabelled.append(item)
    with pytest.raises(ValueError, match="adversarial coverage"):
        validate_account_diversity_coverage(relabelled)


def test_dataset_passes_leakage_and_balance_validation(curriculum):
    frozen = [
        case
        for name in ("train_family.jsonl", "eval_family.jsonl")
        for case in load_cases(ROOT / "data" / "benchmark" / name)
    ]
    report = validate_training_dataset(curriculum, frozen)
    assert report.total == len(curriculum)
    largest_share = max(report.label_counts.values()) / report.total
    assert largest_share <= 0.35


def test_diversity_banks_meet_thresholds():
    report = account_cluster_diversity()
    thresholds = report["thresholds"]
    assert not report["overall"]["near_duplicate_pairs"]
    for family in report["per_family"].values():
        assert family["distinct_2"] >= thresholds["min_distinct_2"]
        assert not family["near_duplicate_pairs"]


def test_committed_dataset_matches_the_generator(curriculum):
    body = "".join(
        json.dumps(example.model_dump(), sort_keys=True, separators=(",", ":")) + "\n"
        for example in curriculum
    )
    assert body == V8_FILE.read_text()


def test_v7_dataset_is_left_untouched():
    v7 = [json.loads(line) for line in V7_FILE.read_text().splitlines()]
    assert len(v7) == 280
    assert not any(item["id"].startswith("sft-v8-") for item in v7)
