from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_wallet.benchmark import load_cases
from agentic_wallet.training.blinded import (
    BLINDED_COMMITMENT_STATUS,
    BLINDED_EVALUATION_CONFIG,
    BLINDED_POST_COMMIT_FAILURE_POLICY,
    BLINDED_RUBRIC_VERSION,
    BLINDED_SEQUENCE_MODE,
    audit_blinded_disjointness,
    load_verified_blinded_cases,
    validate_blinded_commitment,
)
from agentic_wallet.benchmark.blinded_scenarios import (
    BLINDED_SCENARIO_CATALOG_VERSION,
)


def _records() -> list[dict]:
    return [
        {
            "id": f"sealed-model-{index:03d}",
            "family": "sealed",
            "scenario_id": f"blind-composition-{index:03d}",
            "user_request": f"Novel utterance {index} concerning ZIRCON.",
            "workflow_state": "IDLE",
            "available_actions": ["reject_request"],
            "expected_action": "reject_request",
            "context": {"canonical_asset_ids": ["blind:zircon"]},
        }
        for index in range(60)
    ]


def _commitment(payload: bytes) -> dict:
    return {
        "author_generation_config": {
            "batch_count": 8,
            "interface": "openrouter-chat-completions-json-schema",
            "provider_data_collection": "not-restricted-synthetic-prompts-only",
            "provider_require_parameters": True,
            "temperature": "provider-default-unsupported-with-structured-output",
            "whole_suite_regeneration_only": True,
        },
        "author_model": "openrouter/anthropic/claude-fable-5",
        "author_prompt_sha256": "a" * 64,
        "author_request_script_sha256": "e" * 64,
        "author_role": "model-authored blinded evaluator",
        "authoring_attempt_count": 1,
        "blinding_scope": "Model-authored and aggregate-only, not human independent.",
        "candidate_artifact_sha256": "b" * 64,
        "candidate_checkpoint": "checkpoint-25",
        "candidate_selection_commit": "fc0547e",
        "case_count": 60,
        "created_at": "2026-07-23T12:00:00+00:00",
        "disjointness": {
            "address_overlap": 0,
            "asset_id_overlap": 0,
            "exact_request_overlap": 0,
            "identifier_overlap": 0,
            "long_text_overlap": 0,
            "max_request_similarity": 0.1,
            "scenario_id_overlap": 0,
        },
        "evaluation_config": BLINDED_EVALUATION_CONFIG,
        "evaluation_script_sha256": "c" * 64,
        "harness_commit": "frozen-commit",
        "harness_sha256": "d" * 64,
        "human_independence_attested": False,
        "post_commit_failure_policy": BLINDED_POST_COMMIT_FAILURE_POLICY,
        "release_claim_eligible": False,
        "rubric_version": BLINDED_RUBRIC_VERSION,
        "scenario_catalog_version": BLINDED_SCENARIO_CATALOG_VERSION,
        "sequence_mode": BLINDED_SEQUENCE_MODE,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": BLINDED_COMMITMENT_STATUS,
    }


def test_model_authored_commitment_cannot_claim_human_or_release_status(
    tmp_path: Path,
):
    path = tmp_path / "commitment.json"
    metadata = _commitment(b"payload")
    metadata["human_independence_attested"] = True
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="cannot claim human"):
        validate_blinded_commitment(path)

    metadata["human_independence_attested"] = False
    metadata["release_claim_eligible"] = True
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_blinded_commitment(path)


def test_blinded_plaintext_is_digest_and_count_bound(tmp_path: Path):
    suite = tmp_path / "suite.jsonl"
    payload = "".join(json.dumps(record) + "\n" for record in _records()).encode()
    suite.write_bytes(payload)
    commitment = tmp_path / "commitment.json"
    commitment.write_text(json.dumps(_commitment(payload)))

    cases, metadata = load_verified_blinded_cases(suite, commitment)

    assert len(cases) == 60
    assert metadata["release_claim_eligible"] is False
    suite.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="digest"):
        load_verified_blinded_cases(suite, commitment)


def test_disjointness_audit_rejects_reused_assets_and_requests(tmp_path: Path):
    data = tmp_path / "data" / "training"
    data.mkdir(parents=True)
    prior = {
        "scenario_id": "old-scenario",
        "user_request": "Previously used request for OLD.",
        "context": {"canonical_asset_ids": ["old:asset"]},
    }
    (data / "prior.jsonl").write_text(json.dumps(prior) + "\n")
    cases = load_cases(
        _write_suite(
            tmp_path,
            [
                {
                    **_records()[0],
                    "user_request": prior["user_request"],
                    "context": {"canonical_asset_ids": ["old:asset"]},
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="repeats an existing request"):
        audit_blinded_disjointness(cases, root=tmp_path)


def _write_suite(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "suite.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path
