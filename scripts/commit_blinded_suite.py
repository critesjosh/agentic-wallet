"""Commit digest-only metadata for an external model-authored blinded suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_wallet.benchmark import load_cases
from agentic_wallet.training.blinded import (
    BLINDED_COMMITMENT_STATUS,
    BLINDED_EVALUATION_CONFIG,
    BLINDED_POST_COMMIT_FAILURE_POLICY,
    BLINDED_RUBRIC_VERSION,
    BLINDED_SEQUENCE_MODE,
    MIN_BLINDED_CASES,
    audit_blinded_disjointness,
    blinded_harness_sha256,
    sha256_named_files,
)
from agentic_wallet.benchmark.blinded_scenarios import (
    BLINDED_SCENARIO_CATALOG_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "data" / "benchmark" / "claude-blinded-suite-v7.commitment.json"
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--candidate-artifact-sha256", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-selection-commit", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument(
        "--author-prompt", type=Path, action="append", required=True
    )
    parser.add_argument("--authoring-attempt-count", type=int, required=True)
    parser.add_argument(
        "--author-model", default="openrouter/anthropic/claude-fable-5"
    )
    parser.add_argument(
        "--author-role", default="model-authored blinded evaluator"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--acknowledge-model-authored", action="store_true")
    args = parser.parse_args()

    if not args.acknowledge_model_authored:
        raise SystemExit("model-authored acknowledgement is required")
    if _is_within(args.suite, ROOT):
        raise SystemExit("blinded plaintext must remain outside the checkout")
    payload = args.suite.read_bytes()
    cases = load_cases(args.suite)
    if len(cases) < MIN_BLINDED_CASES:
        raise SystemExit(
            f"blinded suite must contain at least {MIN_BLINDED_CASES} cases"
        )
    if any(case.family != "sealed" for case in cases):
        raise SystemExit("every blinded-suite record must use family=sealed")
    disjointness = audit_blinded_disjointness(cases, root=ROOT)
    if args.authoring_attempt_count not in {1, 2}:
        raise SystemExit("authoring attempts are capped at two whole-suite attempts")
    prompt_digest = hashlib.sha256()
    for prompt in args.author_prompt:
        payload = prompt.read_bytes()
        prompt_digest.update(prompt.name.encode())
        prompt_digest.update(b"\0")
        prompt_digest.update(str(len(payload)).encode())
        prompt_digest.update(b"\0")
        prompt_digest.update(payload)
    commitment = {
        "author_generation_config": {
            "batch_count": 8,
            "interface": "openrouter-chat-completions-json-schema",
            "provider_data_collection": "not-restricted-synthetic-prompts-only",
            "provider_require_parameters": True,
            "temperature": "provider-default-unsupported-with-structured-output",
            "whole_suite_regeneration_only": True,
        },
        "author_model": args.author_model,
        "author_prompt_sha256": prompt_digest.hexdigest(),
        "author_request_script_sha256": sha256_named_files(
            ROOT, ("scripts/request_claude_blinded_shard.py",)
        ),
        "author_role": args.author_role,
        "authoring_attempt_count": args.authoring_attempt_count,
        "blinding_scope": (
            "Claude was not given repository training or development plaintext; "
            "the evaluator receives no case-level output. The developer operates "
            "the workflow, so this is not independent-human evidence."
        ),
        "candidate_artifact_sha256": args.candidate_artifact_sha256,
        "candidate_checkpoint": args.candidate_checkpoint,
        "candidate_selection_commit": args.candidate_selection_commit,
        "case_count": len(cases),
        "created_at": datetime.now(UTC).isoformat(),
        "disjointness": disjointness,
        "evaluation_config": BLINDED_EVALUATION_CONFIG,
        "evaluation_script_sha256": sha256_named_files(
            ROOT, ("scripts/evaluate_blinded.py",)
        ),
        "harness_commit": args.harness_commit,
        "harness_sha256": blinded_harness_sha256(ROOT),
        "human_independence_attested": False,
        "post_commit_failure_policy": BLINDED_POST_COMMIT_FAILURE_POLICY,
        "release_claim_eligible": False,
        "rubric_version": BLINDED_RUBRIC_VERSION,
        "scenario_catalog_version": BLINDED_SCENARIO_CATALOG_VERSION,
        "sequence_mode": BLINDED_SEQUENCE_MODE,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": BLINDED_COMMITMENT_STATUS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(commitment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(commitment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
