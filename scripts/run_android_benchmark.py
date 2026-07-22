"""Run and checkpoint the benchmark through an Android llama.cpp server."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from agentic_wallet.benchmark import load_cases, run_benchmark
from agentic_wallet.providers import LlamaCppHTTPProvider


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark"


def _write_report(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "android-benchmark.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    completed: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume and args.output.exists():
        completed = json.loads(args.output.read_text(encoding="utf-8"))["results"]
        completed_ids = {item["case_id"] for item in completed}

    provider = LlamaCppHTTPProvider(
        args.base_url,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )
    cases = load_cases(DATA / "train_family.jsonl") + load_cases(
        DATA / "eval_family.jsonl"
    )
    for case in cases:
        if case.id in completed_ids:
            continue
        started = time.monotonic()
        report = run_benchmark(provider, [case])
        result = asdict(report.results[0])
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["runtime"] = provider.last_response_metadata
        completed.append(result)
        _write_report(args.output, completed)
        print(json.dumps(result, sort_keys=True), flush=True)

    passed = sum(bool(item["ok"]) for item in completed)
    critical = [item for item in completed if item["critical_failure"]]
    print(
        f"passed: {passed}/{len(completed)} clean: {not critical} "
        f"critical_failures: {len(critical)}"
    )


if __name__ == "__main__":
    main()
