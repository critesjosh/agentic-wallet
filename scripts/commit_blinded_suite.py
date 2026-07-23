"""Commit digest-only metadata for an external model-authored blinded suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from agentic_wallet.training.blinded import (
    BLINDED_AUTHOR_GENERATION_CONFIG,
    BLINDED_AUTHOR_MODEL,
    BLINDED_AUTHOR_ROLE,
    BLINDED_BLINDING_SCOPE,
    BLINDED_CANDIDATE_ARTIFACT_SHA256,
    BLINDED_CANDIDATE_CHECKPOINT,
    BLINDED_CANDIDATE_SELECTION_COMMIT,
    BLINDED_COMMITMENT_STATUS,
    BLINDED_EVALUATION_CONFIG,
    BLINDED_HASHED_HARNESS_FILES,
    BLINDED_POST_COMMIT_FAILURE_POLICY,
    BLINDED_RUBRIC_VERSION,
    BLINDED_SEQUENCE_MODE,
    audit_blinded_disjointness,
    blinded_harness_sha256,
    sha256_named_files,
)
from agentic_wallet.training.blinded_authoring import (
    canonical_suite_bytes,
    materialize_author_shards,
)
from agentic_wallet.benchmark.blinded_scenarios import (
    BLINDED_SCENARIO_CATALOG_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "data" / "benchmark" / "terra-blinded-suite-v10.commitment.json"
)
AUTHOR_PROMPTS = (
    "docs/terra-blinded-author-shared-v1.md",
    "docs/terra-blinded-author-shard-1a-v1.md",
    "docs/terra-blinded-author-shard-1b-v1.md",
    "docs/terra-blinded-author-shard-2a-v1.md",
    "docs/terra-blinded-author-shard-2b-v1.md",
    "docs/terra-blinded-author-shard-3a-v1.md",
    "docs/terra-blinded-author-shard-3b-v1.md",
    "docs/terra-blinded-author-shard-4a-v1.md",
    "docs/terra-blinded-author-shard-4b-v1.md",
)
PROTECTED_TRACKED_PATHS = (
    *BLINDED_HASHED_HARNESS_FILES,
    *AUTHOR_PROMPTS,
    "scripts/commit_blinded_suite.py",
    "scripts/evaluate_blinded.py",
    "scripts/materialize_blinded_suite.py",
    "docs/terra-blinded-author-procedure-v1.md",
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def _require_frozen_git_state() -> str:
    head = _git("rev-parse", "HEAD").stdout.strip()
    if _git(
        "merge-base",
        "--is-ancestor",
        BLINDED_CANDIDATE_SELECTION_COMMIT,
        head,
        check=False,
    ).returncode:
        raise SystemExit("candidate selection commit is not an ancestor of HEAD")
    for cached in (False, True):
        arguments = ["diff", "--quiet"]
        if cached:
            arguments.append("--cached")
        arguments.extend(("--", *PROTECTED_TRACKED_PATHS))
        if _git(*arguments, check=False).returncode:
            raise SystemExit("frozen evaluator scope has uncommitted changes")
    return head


def _prompt_digest() -> str:
    digest = hashlib.sha256()
    for name in sorted(AUTHOR_PROMPTS):
        payload = (ROOT / name).read_bytes()
        digest.update(Path(name).name.encode())
        digest.update(b"\0")
        digest.update(str(len(payload)).encode())
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise SystemExit("blinded commitment already exists") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(data)
            output.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--authoring-attempt-count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--acknowledge-model-authored", action="store_true")
    args = parser.parse_args()

    if not args.acknowledge_model_authored:
        raise SystemExit("model-authored acknowledgement is required")
    if _is_within(args.suite, ROOT) or any(
        _is_within(path, ROOT) for path in args.source
    ):
        raise SystemExit("blinded plaintext must remain outside the checkout")
    if args.authoring_attempt_count not in {1, 2}:
        raise SystemExit("authoring attempts are capped at two whole-suite attempts")

    harness_commit = _require_frozen_git_state()
    try:
        cases, receipt = materialize_author_shards(args.source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    suite_payload = args.suite.read_bytes()
    if suite_payload != canonical_suite_bytes(cases):
        raise SystemExit("suite does not match the eight frozen author shards")
    if any(case.family != "sealed" for case in cases):
        raise SystemExit("every blinded-suite record must use family=sealed")
    disjointness = audit_blinded_disjointness(cases, root=ROOT)

    commitment = {
        "author_generation_config": BLINDED_AUTHOR_GENERATION_CONFIG,
        "author_model": BLINDED_AUTHOR_MODEL,
        "author_prompt_sha256": _prompt_digest(),
        "author_procedure_sha256": sha256_named_files(
            ROOT, ("docs/terra-blinded-author-procedure-v1.md",)
        ),
        "author_role": BLINDED_AUTHOR_ROLE,
        "author_shard_sha256": receipt["source_sha256"],
        "authoring_attempt_count": args.authoring_attempt_count,
        "blinding_scope": BLINDED_BLINDING_SCOPE,
        "candidate_artifact_sha256": BLINDED_CANDIDATE_ARTIFACT_SHA256,
        "candidate_checkpoint": BLINDED_CANDIDATE_CHECKPOINT,
        "candidate_selection_commit": BLINDED_CANDIDATE_SELECTION_COMMIT,
        "case_count": len(cases),
        "commit_script_sha256": sha256_named_files(
            ROOT, ("scripts/commit_blinded_suite.py",)
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "disjointness": disjointness,
        "evaluation_config": BLINDED_EVALUATION_CONFIG,
        "evaluation_script_sha256": sha256_named_files(
            ROOT, ("scripts/evaluate_blinded.py",)
        ),
        "harness_commit": harness_commit,
        "harness_sha256": blinded_harness_sha256(ROOT),
        "human_independence_attested": False,
        "post_commit_failure_policy": BLINDED_POST_COMMIT_FAILURE_POLICY,
        "release_claim_eligible": False,
        "rubric_version": BLINDED_RUBRIC_VERSION,
        "scenario_catalog_version": BLINDED_SCENARIO_CATALOG_VERSION,
        "sequence_mode": BLINDED_SEQUENCE_MODE,
        "sha256": hashlib.sha256(suite_payload).hexdigest(),
        "status": BLINDED_COMMITMENT_STATUS,
    }
    _write_new_json(args.output, commitment)
    print(json.dumps(commitment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
