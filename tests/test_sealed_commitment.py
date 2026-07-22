from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_wallet.training import (
    load_verified_sealed_cases,
    validate_sealed_commitment,
)


def test_placeholder_sealed_commitment_blocks_training():
    with pytest.raises(ValueError, match="not committed before training"):
        validate_sealed_commitment(
            Path("data/benchmark/sealed-suite-v1.commitment.json")
        )


def test_valid_digest_only_commitment_is_accepted(tmp_path):
    path = tmp_path / "commitment.json"
    metadata = {
        "author_independence_attested": True,
        "author_role": "independent human evaluator",
        "case_count": 20,
        "created_at": "2026-07-22T12:00:00+00:00",
        "rubric_version": "sealed-wallet-eval-v1",
        "sha256": "a" * 64,
        "status": "committed-before-training",
    }
    path.write_text(json.dumps(metadata))

    assert validate_sealed_commitment(path) == metadata


def test_commitment_rejects_plaintext_or_unexpected_fields(tmp_path):
    path = tmp_path / "commitment.json"
    path.write_text(
        json.dumps(
            {
                "author_independence_attested": True,
                "author_role": "independent human evaluator",
                "case_count": 20,
                "created_at": "2026-07-22T12:00:00+00:00",
                "rubric_version": "sealed-wallet-eval-v1",
                "sha256": "a" * 64,
                "status": "committed-before-training",
                "cases": [{"user_request": "plaintext must not be committed"}],
            }
        )
    )

    with pytest.raises(ValueError, match="unexpected or plaintext"):
        validate_sealed_commitment(path)


def test_verified_sealed_cases_preserve_external_registry(tmp_path):
    suite = tmp_path / "sealed.jsonl"
    records = [
        {
            "id": f"sealed-{index:02d}",
            "family": "sealed",
            "scenario_id": f"independent-{index:02d}",
            "user_request": f"Independent request {index}",
            "workflow_state": "IDLE",
            "available_actions": ["reject_request"],
            "expected_action": "reject_request",
            "context": {"canonical_asset_ids": ["independent:alpha"]},
        }
        for index in range(20)
    ]
    suite.write_text("".join(json.dumps(record) + "\n" for record in records))
    commitment = tmp_path / "commitment.json"
    commitment.write_text(
        json.dumps(
            {
                "author_independence_attested": True,
                "author_role": "independent human evaluator",
                "case_count": 20,
                "created_at": "2026-07-22T12:00:00+00:00",
                "rubric_version": "sealed-wallet-eval-v1",
                "sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
                "status": "committed-before-training",
            }
        )
    )

    cases, _ = load_verified_sealed_cases(suite, commitment)
    assert len(cases) == 20
    assert cases[0].context["canonical_asset_ids"] == ["independent:alpha"]

    suite.write_text(suite.read_text() + "\n")
    with pytest.raises(ValueError, match="digest"):
        load_verified_sealed_cases(suite, commitment)
