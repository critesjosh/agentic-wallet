from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_wallet.benchmark import load_cases
from agentic_wallet.training.blinded import (
    BLINDED_AUTHOR_GENERATION_CONFIG,
    BLINDED_AUTHOR_MODEL,
    BLINDED_AUTHOR_ROLE,
    BLINDED_BLINDING_SCOPE,
    BLINDED_CANDIDATE_ARTIFACT_SHA256,
    BLINDED_CANDIDATE_CHECKPOINT,
    BLINDED_CANDIDATE_SELECTION_COMMIT,
    BLINDED_CASE_COUNT,
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
        for index in range(BLINDED_CASE_COUNT)
    ]


def _commitment(payload: bytes) -> dict:
    return {
        "author_generation_config": BLINDED_AUTHOR_GENERATION_CONFIG,
        "author_model": BLINDED_AUTHOR_MODEL,
        "author_prompt_sha256": "a" * 64,
        "author_procedure_sha256": "e" * 64,
        "author_role": BLINDED_AUTHOR_ROLE,
        "author_shard_sha256": [f"{index:064x}" for index in range(8)],
        "author_validator_sha256": "8" * 64,
        "authoring_attempt_count": 1,
        "blinding_scope": BLINDED_BLINDING_SCOPE,
        "candidate_artifact_sha256": BLINDED_CANDIDATE_ARTIFACT_SHA256,
        "candidate_checkpoint": BLINDED_CANDIDATE_CHECKPOINT,
        "candidate_selection_commit": BLINDED_CANDIDATE_SELECTION_COMMIT,
        "case_count": BLINDED_CASE_COUNT,
        "commit_script_sha256": "9" * 64,
        "created_at": "2026-07-23T12:00:00+00:00",
        "disjointness": {
            "address_overlap": 0,
            "asset_id_overlap": 0,
            "exact_request_overlap": 0,
            "identifier_overlap": 0,
            "long_text_overlap": 0,
            "max_request_similarity": 0.1,
            "scenario_id_overlap": 0,
            "text_overlap": 0,
        },
        "evaluation_config": BLINDED_EVALUATION_CONFIG,
        "evaluation_script_sha256": "c" * 64,
        "harness_commit": "f" * 40,
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

    assert len(cases) == BLINDED_CASE_COUNT
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


def test_disjointness_audit_reads_nested_source_json_typed_and_untrusted_values(
    tmp_path: Path,
):
    source = tmp_path / "data" / "benchmark" / "development.source.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "typed": {"trusted_recipient_id": "dev:recipient-9"},
                        "context": {
                            "untrusted_data": {
                                "opaque_blob": "Unique nested source warning."
                            }
                        },
                    }
                ]
            }
        )
    )
    cases = load_cases(
        _write_suite(
            tmp_path,
            [
                {
                    **_records()[0],
                    "expected_arguments": {"trusted_recipient_id": "dev:recipient-9"},
                    "context": {
                        "canonical_asset_ids": ["blind:zircon"],
                        "untrusted_data": {"memo": "fresh text"},
                    },
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="typed identifier"):
        audit_blinded_disjointness(cases, root=tmp_path)


def test_disjointness_audit_rejects_nested_untrusted_source_text(tmp_path: Path):
    source = tmp_path / "data" / "training" / "development.source.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "context": {
                            "untrusted_data": {
                                "memo": "Unique nested source warning."
                            }
                        }
                    }
                ]
            }
        )
    )
    cases = load_cases(
        _write_suite(
            tmp_path,
            [
                {
                    **_records()[0],
                    "context": {
                        "canonical_asset_ids": ["blind:zircon"],
                        "untrusted_data": {
                            "opaque_blob": "Unique nested source warning."
                        }
                    },
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="typed or untrusted text"):
        audit_blinded_disjointness(cases, root=tmp_path)


def test_disjointness_audit_rejects_nested_similar_request(
    tmp_path: Path,
):
    source = tmp_path / "data" / "training" / "nested.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "user_request": "Novel utterance 0 concerning zircon!",
                    }
                ]
            }
        )
    )
    cases = load_cases(_write_suite(tmp_path, [_records()[0]]))

    with pytest.raises(ValueError, match="too similar"):
        audit_blinded_disjointness(cases, root=tmp_path)


def test_disjointness_audit_rejects_nested_scenario_id(tmp_path: Path):
    source = tmp_path / "data" / "training" / "nested.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {"records": [{"scenario_id": "blind-composition-000"}]}
        )
    )
    cases = load_cases(_write_suite(tmp_path, [_records()[0]]))

    with pytest.raises(ValueError, match="typed identifier|scenario ID"):
        audit_blinded_disjointness(cases, root=tmp_path)


def test_commitment_requires_exact_author_config_and_zero_overlap(tmp_path: Path):
    path = tmp_path / "commitment.json"
    metadata = _commitment(b"payload")
    metadata["author_generation_config"] = {
        **BLINDED_AUTHOR_GENERATION_CONFIG,
        "interface": "openrouter",
    }
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="author generation"):
        validate_blinded_commitment(path)

    metadata = _commitment(b"payload")
    metadata["disjointness"]["text_overlap"] = 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="nonzero"):
        validate_blinded_commitment(path)


def test_disjointness_audit_excludes_result_commitment_and_manifest_aggregates(
    tmp_path: Path,
):
    benchmark = tmp_path / "data" / "benchmark"
    results = benchmark / "results"
    results.mkdir(parents=True)
    ignored = {
        "scenario_id": _records()[0]["scenario_id"],
        "user_request": _records()[0]["user_request"],
        "context": {"canonical_asset_ids": ["blind:zircon"]},
    }
    (results / "derived-result.json").write_text(json.dumps(ignored))
    (benchmark / "sealed-suite-v1.commitment.json").write_text(json.dumps(ignored))
    (benchmark / "train.manifest.json").write_text(json.dumps(ignored))
    cases = load_cases(_write_suite(tmp_path, [_records()[0]]))

    result = audit_blinded_disjointness(cases, root=tmp_path)

    assert result["exact_request_overlap"] == 0
    assert result["identifier_overlap"] == 0
    assert result["text_overlap"] == 0


def _write_suite(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "suite.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path
