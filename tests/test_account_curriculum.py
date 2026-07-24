"""The v7 account-identity curriculum and its inherited v6 records."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agentic_wallet.training.account_curriculum import (
    LIVE_ACTIONS,
    load_account_curriculum,
    validate_account_curriculum_coverage,
)
from agentic_wallet.training.transaction_curriculum import (
    load_transaction_candidate_curriculum,
)
from agentic_wallet.web.chat import _BASE_MODEL_ACTIONS, _TRANSFER_MODEL_ACTIONS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "training" / "natural_v3_source.jsonl"
V6_FILE = ROOT / "data" / "training" / "sft-v6-transaction-boundary.jsonl"
V7_FILE = ROOT / "data" / "training" / "sft-v7-account-identity.jsonl"


@pytest.fixture(scope="module")
def curriculum():
    return load_account_curriculum(SOURCE)


def _additions(examples):
    return [item for item in examples if item.id.startswith("sft-v7-")]


def test_curriculum_shape(curriculum):
    assert len(curriculum) == 280
    assert len(_additions(curriculum)) == 12


def test_live_actions_track_the_application(curriculum):
    """A hard-coded copy of the allowlist is exactly how these drift apart."""

    assert LIVE_ACTIONS == [*_BASE_MODEL_ACTIONS, *_TRANSFER_MODEL_ACTIONS]
    assert "get_account" in LIVE_ACTIONS


def test_inherited_production_records_offer_get_account(curriculum):
    production = [
        item
        for item in curriculum
        if item.action_exposure == "production"
        and "get_portfolio" in item.available_actions
    ]
    assert production, "expected inherited production routing records"
    for item in production:
        assert "get_account" in item.available_actions


def test_v6_records_are_otherwise_unchanged(curriculum):
    """Only the allowlist may differ, so v6 behaviour is not silently rewritten."""

    before = {item.id: item for item in load_transaction_candidate_curriculum(SOURCE)}
    inherited = [item for item in curriculum if not item.id.startswith("sft-v7-")]
    assert len(inherited) == 268

    for item in inherited:
        original = before[item.id]
        assert item.target == original.target
        assert item.context == original.context
        assert item.coverage == original.coverage
        assert item.kind == original.kind
        assert item.split == original.split


def test_account_routing_and_disambiguation(curriculum):
    targets = {
        item.context["user_request"]: item.target["proposed_action"]
        for item in _additions(curriculum)
        if "proposed_action" in item.target and item.available_actions
    }
    assert targets["What is my address?"] == "get_account"
    assert targets["show me my wallet address"] == "get_account"
    # Words that overlap must still resolve to the tool the user named.
    assert targets["show the registry addresses"] == "get_registry"
    assert targets["what do I hold right now"] == "get_portfolio"


def test_key_and_seed_requests_route_nowhere(curriculum):
    secrets = [
        item
        for item in _additions(curriculum)
        if item.coverage.risk_category == "key-disclosure"
    ]
    assert len(secrets) >= 2
    for item in secrets:
        assert item.target["proposed_action"] == "none"


def test_untrusted_text_never_becomes_the_account(curriculum):
    untrusted = [
        item
        for item in _additions(curriculum)
        if item.coverage.adversarial_condition == "untrusted-account-claim"
    ]
    assert untrusted
    for item in untrusted:
        assert item.target["proposed_action"] == "none"
        # The attacker address appears only inside the untrusted envelope.
        assert item.context["untrusted_data"]["content"]
        assert "untrusted_data" in item.context
        assert item.context["user_request"].count("0x") == 0


def test_fixture_narration_refuses_to_call_it_the_users_account(curriculum):
    fixture = next(
        item
        for item in _additions(curriculum)
        if item.scenario_class == "v7-fixture-account-result"
    )
    message = fixture.target["message"]
    assert "No real account is loaded" in message
    assert "no funds should be sent to it" in message


def test_no_key_or_mnemonic_shaped_values_anywhere(curriculum):
    """This corpus deliberately contains key request text, so shape is the guard."""

    key_shaped = re.compile(r"(?:0x)?[0-9a-fA-F]{64}")
    for item in _additions(curriculum):
        assert not key_shaped.search(str(item.model_dump()))


def test_coverage_validator_rejects_dropped_refusal_coverage(curriculum):
    """Relabel rather than delete, so the count check cannot mask the gap."""

    relabelled = []
    for item in curriculum:
        if item.coverage.adversarial_condition == "key-disclosure-request":
            item = item.model_copy(
                update={"coverage": item.coverage.model_copy(
                    update={"adversarial_condition": "none"}
                )}
            )
        relabelled.append(item)

    with pytest.raises(ValueError, match="adversarial coverage"):
        validate_account_curriculum_coverage(relabelled)


def test_coverage_validator_rejects_a_key_request_that_routes_to_a_tool(curriculum):
    escalated = []
    for item in curriculum:
        if item.coverage.risk_category == "key-disclosure":
            item = item.model_copy(
                update={"target": {"proposed_action": "get_account"}}
            )
        escalated.append(item)

    with pytest.raises(ValueError, match="must not route to a tool"):
        validate_account_curriculum_coverage(escalated)


def test_committed_corpora_match_the_generators():
    """v6 must keep reproducing even though v7 inherits and rewrites it."""

    v6 = [json.loads(line) for line in V6_FILE.read_text().splitlines()]
    v7 = [json.loads(line) for line in V7_FILE.read_text().splitlines()]
    assert len(v6) == 268
    assert len(v7) == 280

    v6_ids = {item["id"] for item in v6}
    v7_ids = {item["id"] for item in v7}
    assert v6_ids < v7_ids

    # The committed v6 file still carries its own frozen allowlist.
    v6_production = [
        item
        for item in v6
        if item["action_exposure"] == "production"
        and "get_portfolio" in item["available_actions"]
    ]
    assert v6_production
    for item in v6_production:
        assert "get_account" not in item["available_actions"]
