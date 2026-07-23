from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agentic_wallet.blinded_evaluation import (
    BlindedEvaluationAlreadyUsed,
    blinded_report_path,
    blinded_state_path,
    claim_blinded_evaluation,
    finish_blinded_evaluation,
    publish_blinded_aggregate,
)


def _commitment(tmp_path: Path) -> dict:
    return {
        "sha256": "a" * 64,
        "candidate_artifact_sha256": "b" * 64,
        "evaluation_config": {"custody_root": str(tmp_path / "custody")},
    }


def test_claim_is_keyed_by_commitment_and_record_is_opaque(tmp_path: Path):
    commitment = _commitment(tmp_path)
    state = claim_blinded_evaluation(commitment)

    assert state == blinded_state_path(commitment)
    record = json.loads(state.read_text())
    assert record["status"] == "claimed"
    assert record["commitment_sha256"] == "a" * 64
    assert "suite" not in record
    assert "path" not in record

    with pytest.raises(BlindedEvaluationAlreadyUsed):
        claim_blinded_evaluation(commitment)


def test_claimed_evaluation_completes_once_with_aggregate_digest(tmp_path: Path):
    commitment = _commitment(tmp_path)
    state = claim_blinded_evaluation(commitment)
    _, digest = publish_blinded_aggregate(
        commitment, {"suite_sha256": commitment["sha256"], "total": 64}
    )
    finish_blinded_evaluation(state, "completed", aggregate_sha256=digest)

    record = json.loads(state.read_text())
    assert record["status"] == "completed"
    assert record["aggregate_sha256"] == digest
    with pytest.raises(RuntimeError, match="terminal"):
        finish_blinded_evaluation(state, "retired")


def test_claimed_is_irreversibly_used_even_without_cleanup(tmp_path: Path):
    commitment = _commitment(tmp_path)
    state = claim_blinded_evaluation(commitment)

    assert json.loads(state.read_text())["status"] == "claimed"
    with pytest.raises(BlindedEvaluationAlreadyUsed):
        claim_blinded_evaluation(commitment)


def test_retired_evaluation_cannot_bind_aggregate(tmp_path: Path):
    commitment = _commitment(tmp_path)
    state = claim_blinded_evaluation(commitment)
    with pytest.raises(ValueError, match="cannot bind"):
        finish_blinded_evaluation(state, "retired", aggregate_sha256="c" * 64)
    finish_blinded_evaluation(state, "retired")
    assert json.loads(state.read_text())["status"] == "retired"


def test_aggregate_publication_is_create_once_and_digest_bound(tmp_path: Path):
    commitment = _commitment(tmp_path)
    aggregate = {"suite_sha256": commitment["sha256"], "total": 64}

    path, digest = publish_blinded_aggregate(commitment, aggregate)

    assert path == blinded_report_path(commitment)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(BlindedEvaluationAlreadyUsed):
        publish_blinded_aggregate(commitment, aggregate)


def test_evaluator_claims_before_plaintext_and_retires_on_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = Path(__file__).parents[1] / "scripts" / "evaluate_blinded.py"
    spec = importlib.util.spec_from_file_location("evaluate_blinded", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commitment = {
        **_commitment(tmp_path),
        "evaluation_config": {
            "base_model": "google/gemma-4-E2B-it",
            "base_model_revision": "revision",
            "custody_root": str(tmp_path / "custody"),
            "device": "cuda",
            "load_in_4bit": True,
                "max_new_tokens": 256,
                "runtime_constraints": {},
                "source_root": str(tmp_path / "workspace"),
            },
        "harness_sha256": "harness",
        "evaluation_script_sha256": "evaluator",
    }

    def fail_after_claim(*_):
        assert blinded_state_path(commitment).exists()
        raise RuntimeError("no plaintext")

    monkeypatch.setattr(module, "validate_blinded_commitment", lambda _: commitment)
    monkeypatch.setattr(module, "_require_frozen_environment", lambda _: None)
    monkeypatch.setattr(module, "blinded_harness_sha256", lambda _: "harness")
    monkeypatch.setattr(module, "sha256_named_files", lambda *_: "evaluator")
    monkeypatch.setattr(module, "blinded_adapter_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(module, "load_verified_blinded_cases", fail_after_claim)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_blinded.py",
            "--suite",
            "/not/read",
            "--commitment",
            "commitment.json",
            "--adapter-path",
            "adapter",
        ],
    )

    with pytest.raises(RuntimeError, match="no plaintext"):
        module.main()
    assert json.loads(blinded_state_path(commitment).read_text())["status"] == (
        "retired"
    )
