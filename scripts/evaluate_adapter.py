"""Evaluate a PEFT adapter through the unchanged benchmark provider contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agentic_wallet.benchmark import load_cases, run_benchmark
from agentic_wallet.providers import LocalTransformersProvider
from agentic_wallet.training.config import BASE_MODEL_ID, BASE_MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    provider = LocalTransformersProvider(
        model_id=args.model_id,
        revision=args.revision,
        adapter_path=str(args.adapter_path) if args.adapter_path is not None else None,
        max_new_tokens=args.max_new_tokens,
    )
    provider.load()
    cases = load_cases(DATA / "train_family.jsonl") + load_cases(
        DATA / "eval_family.jsonl"
    )
    report = run_benchmark(provider, cases)
    print(f"adapter: {args.adapter_path or 'none (untuned base)'}")
    print(f"passed: {report.passed}/{report.total}")
    print(f"structured_output_rate: {report.structured_output_rate:.3f}")
    print(f"critical_failures: {len(report.critical_failures)}")
    print(f"release_ready: {report.release_ready}")
    for family, metrics in report.by_family.items():
        print(
            f"family[{family}]: passed={metrics.passed}/{metrics.total} "
            f"actions={metrics.action_passed}/{metrics.total} "
            f"arguments={metrics.argument_passed}/{metrics.total} "
            f"structured={metrics.structured_output_valid}/{metrics.total} "
            f"critical={metrics.critical_failures}"
        )
    for result in report.results:
        print(
            f"{result.case_id}: ok={result.ok} schema_valid={result.schema_valid} "
            f"action={result.chosen_action!r} arguments={result.chosen_arguments!r} "
            f"critical_failure={result.critical_failure!r} "
            f"inference_error={result.inference_error!r}"
        )
    if args.json_output is not None:
        payload = {
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "passed": report.passed,
            "total": report.total,
            "structured_output_rate": report.structured_output_rate,
            "critical_failures": len(report.critical_failures),
            "release_ready": report.release_ready,
            "by_family": {
                family: asdict(metrics)
                for family, metrics in report.by_family.items()
            },
            "results": [asdict(result) for result in report.results],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
