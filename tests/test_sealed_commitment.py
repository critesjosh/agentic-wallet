from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_wallet.training import validate_sealed_commitment


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
