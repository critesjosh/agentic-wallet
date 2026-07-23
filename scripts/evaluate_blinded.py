# /// script
# requires-python = "==3.12.12"
# dependencies = [
#   "accelerate==1.14.0",
#   "bitsandbytes==0.49.2",
#   "eth-utils==6.0.0",
#   "packaging==26.2",
#   "peft==0.19.1",
#   "pydantic==2.13.4",
#   "torch==2.13.0",
#   "torchvision==0.28.0",
#   "transformers==5.10.1",
# ]
# ///
"""Evaluate a precommitted blinded suite exactly once.

Plaintext suite data and raw case results remain outside the checkout. The
script emits only aggregate metrics and never serializes per-case results.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

from packaging.specifiers import SpecifierSet

from agentic_wallet.benchmark import run_benchmark
from agentic_wallet.blinded_evaluation import (
    BlindedEvaluationAlreadyUsed,
    claim_blinded_evaluation,
    finish_blinded_evaluation,
    publish_blinded_aggregate,
)
from agentic_wallet.providers import LocalTransformersProvider
from agentic_wallet.training import (
    blinded_adapter_sha256,
    blinded_harness_sha256,
    load_verified_blinded_cases,
    sha256_named_files,
)
from agentic_wallet.training.blinded import validate_blinded_commitment


def _require_frozen_environment(commitment: dict) -> None:
    config = commitment["evaluation_config"]
    if platform.python_version() != config["python_version"]:
        raise SystemExit("Python version does not match frozen evaluation config")
    if os.environ.get("PYTHONPATH") != config["pythonpath"]:
        raise SystemExit("PYTHONPATH does not match frozen evaluation config")
    for package, specifier in config["runtime_constraints"].items():
        installed = importlib.metadata.version(package)
        if installed not in SpecifierSet(specifier):
            raise SystemExit(
                f"installed {package}={installed} violates frozen constraint {specifier}"
            )
    import torch

    if torch.version.cuda != config["cuda_version"]:
        raise SystemExit("CUDA build does not match frozen evaluation config")


def _build_aggregate(report, provider, cases, commitment: dict) -> dict:
    generated_argument_results = [
        item
        for case, item in zip(cases, report.results, strict=True)
        if "create_transfer_plan_from_candidate" not in case.available_actions
        and len(case.expected_arguments) > 0
    ]
    candidate_bound_results = [
        item
        for case, item in zip(cases, report.results, strict=True)
        if "create_transfer_plan_from_candidate" in case.available_actions
    ]
    return {
        "action_passed": sum(item.action_ok for item in report.results),
        "argument_passed": sum(item.arguments_ok for item in report.results),
        "author_role": commitment["author_role"],
        "by_argument_count": {
            bucket: asdict(metrics)
            for bucket, metrics in report.by_argument_count.items()
        },
        "by_hard_zero": {
            category: asdict(metrics)
            for category, metrics in report.by_hard_zero.items()
        },
        "candidate_artifact_sha256": commitment["candidate_artifact_sha256"],
        "candidate_checkpoint": commitment["candidate_checkpoint"],
        "critical_failures": len(report.critical_failures),
        "deterministic_candidate_binding": {
            "passed": sum(item.arguments_ok for item in candidate_bound_results),
            "total": len(candidate_bound_results),
        },
        "exact_passed": report.passed,
        "generated_arguments": {
            "passed": sum(item.arguments_ok for item in generated_argument_results),
            "total": len(generated_argument_results),
        },
        "human_independence_attested": False,
        "model": provider.model_id,
        "native_constrained_decoding": provider.native_constrained_decoding,
        "provider": provider.name,
        "release_claim_eligible": False,
        "rubric_version": commitment["rubric_version"],
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in commitment["evaluation_config"]["runtime_constraints"]
        },
        "schema_valid": sum(item.schema_valid for item in report.results),
        "sequence_mode": commitment["sequence_mode"],
        "sequence_accuracy": report.sequence_accuracy,
        "suite_sha256": commitment["sha256"],
        "syntax_valid": sum(item.syntax_valid for item in report.results),
        "total": report.total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    args = parser.parse_args()

    # Digest-only and runtime checks are retryable because no plaintext has been
    # opened and no claim has been made.
    commitment = validate_blinded_commitment(args.commitment)
    config = commitment["evaluation_config"]
    _require_frozen_environment(commitment)
    root = Path(config["source_root"])
    if blinded_harness_sha256(root) != commitment["harness_sha256"]:
        raise SystemExit("frozen harness digest mismatch")
    if (
        sha256_named_files(root, ("scripts/evaluate_blinded.py",))
        != commitment["evaluation_script_sha256"]
    ):
        raise SystemExit("frozen evaluator digest mismatch")
    if (
        blinded_adapter_sha256(args.adapter_path)
        != commitment["candidate_artifact_sha256"]
    ):
        raise SystemExit("candidate adapter digest mismatch")

    try:
        state_path = claim_blinded_evaluation(commitment)
    except BlindedEvaluationAlreadyUsed as exc:
        raise SystemExit(str(exc)) from exc

    try:
        # The first suite plaintext read occurs only after the durable claim.
        cases, loaded_commitment = load_verified_blinded_cases(
            args.suite, args.commitment
        )
        if loaded_commitment != commitment:
            raise RuntimeError("commitment changed after evaluation claim")
        provider = LocalTransformersProvider(
            model_id=config["base_model"],
            revision=config["base_model_revision"],
            adapter_path=str(args.adapter_path),
            load_in_4bit=config["load_in_4bit"],
            max_new_tokens=config["max_new_tokens"],
            device=config["device"],
        )
        provider.load()
        report = run_benchmark(provider, cases)
        aggregate = _build_aggregate(report, provider, cases, commitment)
        _, aggregate_sha256 = publish_blinded_aggregate(
            commitment, aggregate
        )
    except BaseException:
        # ``claimed`` already means consumed. Best-effort refinement improves
        # audit clarity; even SIGKILL cannot reopen the suite.
        try:
            finish_blinded_evaluation(state_path, "retired")
        except BaseException:
            pass
        raise
    else:
        finish_blinded_evaluation(
            state_path,
            "completed",
            aggregate_sha256=aggregate_sha256,
        )
        print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
