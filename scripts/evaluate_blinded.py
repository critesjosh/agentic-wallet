# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.10",
#   "bitsandbytes>=0.49",
#   "eth-utils>=5,<7",
#   "peft>=0.19",
#   "pydantic>=2.6",
#   "torch>=2.7",
#   "torchvision>=0.22",
#   "transformers>=5.10.1",
# ]
# ///
"""Evaluate committed blinded plaintext once and emit aggregate metrics only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from agentic_wallet.benchmark import run_benchmark
from agentic_wallet.providers import LocalTransformersProvider
from agentic_wallet.training import (
    blinded_adapter_sha256,
    blinded_harness_sha256,
    load_verified_blinded_cases,
    sha256_named_files,
)
from agentic_wallet.training.config import BASE_MODEL_ID, BASE_MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    cases, commitment = load_verified_blinded_cases(
        args.suite, args.commitment
    )
    root = Path(__file__).resolve().parents[1]
    if blinded_harness_sha256(root) != commitment["harness_sha256"]:
        raise SystemExit("frozen harness digest mismatch; suite is retired")
    if sha256_named_files(
        root, ("scripts/evaluate_blinded.py",)
    ) != commitment["evaluation_script_sha256"]:
        raise SystemExit("frozen evaluator digest mismatch; suite is retired")
    if (
        blinded_adapter_sha256(args.adapter_path)
        != commitment["candidate_artifact_sha256"]
    ):
        raise SystemExit("candidate adapter digest mismatch; suite is retired")
    provider = LocalTransformersProvider(
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        adapter_path=str(args.adapter_path),
    )
    provider.load()
    report = run_benchmark(provider, cases)
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
    aggregate = {
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
        "candidate_artifact_sha256": commitment[
            "candidate_artifact_sha256"
        ],
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
            for package in (
                "accelerate",
                "bitsandbytes",
                "peft",
                "torch",
                "transformers",
            )
        },
        "schema_valid": sum(item.schema_valid for item in report.results),
        "sequence_mode": commitment["sequence_mode"],
        "sequence_accuracy": report.sequence_accuracy,
        "suite_sha256": commitment["sha256"],
        "syntax_valid": sum(item.syntax_valid for item in report.results),
        "total": report.total,
    }
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
