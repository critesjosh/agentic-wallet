"""Verify and evaluate external sealed plaintext, emitting aggregates only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agentic_wallet.benchmark import run_benchmark
from agentic_wallet.providers import LocalTransformersProvider, OllamaProvider
from agentic_wallet.training import load_verified_sealed_cases
from agentic_wallet.training.config import BASE_MODEL_ID, BASE_MODEL_REVISION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMITMENT = ROOT / "data" / "benchmark" / "sealed-suite-v1.commitment.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, default=DEFAULT_COMMITMENT)
    parser.add_argument(
        "--provider", choices=("ollama", "transformers"), default="transformers"
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    cases, commitment = load_verified_sealed_cases(args.suite, args.commitment)

    if args.provider == "ollama":
        if args.adapter_path is not None:
            raise SystemExit("--adapter-path is available only for Transformers")
        provider = OllamaProvider(
            model=args.model_id or "gemma4:e2b", base_url=args.base_url
        )
    else:
        provider = LocalTransformersProvider(
            model_id=args.model_id or BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            adapter_path=(
                str(args.adapter_path) if args.adapter_path is not None else None
            ),
        )
        provider.load()

    report = run_benchmark(provider, cases)
    aggregate = {
        "suite_sha256": commitment["sha256"],
        "rubric_version": commitment["rubric_version"],
        "provider": provider.name,
        "model": getattr(provider, "model", getattr(provider, "model_id", None)),
        "native_constrained_decoding": provider.native_constrained_decoding,
        "total": report.total,
        "exact_passed": report.passed,
        "syntax_valid": sum(item.syntax_valid for item in report.results),
        "schema_valid": sum(item.schema_valid for item in report.results),
        "action_passed": sum(item.action_ok for item in report.results),
        "argument_passed": sum(item.arguments_ok for item in report.results),
        "sequence_accuracy": report.sequence_accuracy,
        "critical_failures": len(report.critical_failures),
        "by_argument_count": {
            bucket: asdict(metrics)
            for bucket, metrics in report.by_argument_count.items()
        },
        "by_hard_zero": {
            category: asdict(metrics)
            for category, metrics in report.by_hard_zero.items()
        },
    }
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
