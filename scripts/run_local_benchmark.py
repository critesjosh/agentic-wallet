"""Run the behavioral baseline against local Ollama or Transformers."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    if args.provider == "ollama":
        provider = OllamaProvider(
            model=args.model_id or "gemma4:e2b",
            base_url=args.base_url,
        )
    else:
        provider = LocalTransformersProvider(
            model_id=args.model_id or "google/gemma-4-E2B",
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
    print(f"structured_output_rate: {report.structured_output_rate:.3f}")
    print(f"structured_output_gate_passed: {report.structured_output_gate_passed}")
    print(f"release_ready: {report.release_ready}")
    print(f"critical_failures: {len(report.critical_failures)}")
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
            f"action={result.chosen_action!r} arguments_ok={result.arguments_ok} "
            f"critical_failure={result.critical_failure!r} "
            f"inference_error={result.inference_error!r}"
        )


if __name__ == "__main__":
    main()
