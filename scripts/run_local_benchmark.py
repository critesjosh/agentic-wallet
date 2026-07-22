"""Run the behavioral baseline against local Ollama or Transformers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agentic_wallet.benchmark import load_cases, run_benchmark
from agentic_wallet.providers import LocalTransformersProvider, OllamaProvider


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=("ollama", "transformers"), default="ollama"
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.provider == "ollama":
        provider = OllamaProvider(
            model=args.model_id or "gemma4:e2b",
            base_url=args.base_url,
        )
    else:
        provider = LocalTransformersProvider(
            model_id=args.model_id or "google/gemma-4-E2B-it",
            max_new_tokens=args.max_new_tokens,
        )
        provider.load()
    cases = load_cases(DATA / "train_family.jsonl") + load_cases(
        DATA / "eval_family.jsonl"
    )
    report = run_benchmark(provider, cases)

    print(f"provider: {provider.name}")
    print(f"model: {getattr(provider, 'model', getattr(provider, 'model_id', None))}")
    print(f"passed: {report.passed}/{report.total}")
    print(f"clean: {report.clean}")
    print(f"syntax_valid_rate: {report.syntax_valid_rate:.3f}")
    print(f"structured_output_rate: {report.structured_output_rate:.3f}")
    print(f"sequence_accuracy: {report.sequence_accuracy:.3f}")
    print(f"structured_output_gate_passed: {report.structured_output_gate_passed}")
    print(f"release_ready: {report.release_ready}")
    print(f"critical_failures: {len(report.critical_failures)}")
    for family, metrics in report.by_family.items():
        print(
            f"family[{family}]: passed={metrics.passed}/{metrics.total} "
            f"actions={metrics.action_passed}/{metrics.total} "
            f"arguments={metrics.argument_passed}/{metrics.total} "
            f"syntax={metrics.syntax_valid}/{metrics.total} "
            f"structured={metrics.structured_output_valid}/{metrics.total} "
            f"critical={metrics.critical_failures}"
        )
    for bucket, metrics in report.by_argument_count.items():
        print(
            f"arguments[{bucket}]: passed={metrics.passed}/{metrics.total} "
            f"actions={metrics.action_passed}/{metrics.total} "
            f"arguments={metrics.argument_passed}/{metrics.total} "
            f"syntax={metrics.syntax_valid}/{metrics.total} "
            f"structured={metrics.structured_output_valid}/{metrics.total} "
            f"critical={metrics.critical_failures}"
        )
    for category, metrics in report.by_hard_zero.items():
        print(
            f"hard_zero[{category}]: failures={metrics.failures}/{metrics.total}"
        )
    for result in report.results:
        print(
            f"{result.case_id}: ok={result.ok} schema_valid={result.schema_valid} "
            f"action={result.chosen_action!r} arguments_ok={result.arguments_ok} "
            f"critical_failure={result.critical_failure!r} "
            f"inference_error={result.inference_error!r}"
        )
    if args.json_output is not None:
        payload = {
            "provider": provider.name,
            "model": getattr(provider, "model", getattr(provider, "model_id", None)),
            "native_constrained_decoding": provider.native_constrained_decoding,
            "benchmark_role": "development-regression-only",
            "passed": report.passed,
            "total": report.total,
            "schema_valid": sum(item.schema_valid for item in report.results),
            "syntax_valid": sum(item.syntax_valid for item in report.results),
            "sequence_accuracy": report.sequence_accuracy,
            "action_passed": sum(item.action_ok for item in report.results),
            "argument_passed": sum(item.arguments_ok for item in report.results),
            "critical_failures": len(report.critical_failures),
            "release_ready": report.release_ready,
            "by_family": {
                family: asdict(metrics) for family, metrics in report.by_family.items()
            },
            "by_argument_count": {
                bucket: asdict(metrics)
                for bucket, metrics in report.by_argument_count.items()
            },
            "by_hard_zero": {
                category: asdict(metrics)
                for category, metrics in report.by_hard_zero.items()
            },
            "results": [asdict(item) for item in report.results],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
