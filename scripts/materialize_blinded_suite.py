"""Compile external author scenarios into deterministic benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agentic_wallet.benchmark.blinded_scenarios import (
    benchmark_case_dict,
    compile_blinded_source,
)
from agentic_wallet.training.blinded import MIN_BLINDED_CASES

ROOT = Path(__file__).resolve().parents[1]


def _within_checkout(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _within_checkout(args.source) or _within_checkout(args.output):
        raise SystemExit(
            "blinded source and plaintext suite must stay outside checkout"
        )

    cases = []
    source_ids: set[str] = set()
    scenario_ids: set[str] = set()
    for line_number, line in enumerate(args.source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"line {line_number} is not an object")
        case = compile_blinded_source(value)
        if case.id in source_ids or case.scenario_id in scenario_ids:
            raise SystemExit("blinded case and scenario IDs must be unique")
        source_ids.add(case.id)
        scenario_ids.add(case.scenario_id)
        cases.append(case)
    if len(cases) < MIN_BLINDED_CASES:
        raise SystemExit(f"need at least {MIN_BLINDED_CASES} valid cases")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(benchmark_case_dict(case), sort_keys=True) + "\n"
            for case in cases
        )
    )
    trajectories = {case.trajectory_id for case in cases if case.trajectory_id}
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "hard_zero_count": sum(
                    case.hard_zero_category is not None for case in cases
                ),
                "scenario_counts": dict(
                    sorted(Counter(
                        json.loads(line)["scenario_type"]
                        for line in args.source.read_text().splitlines()
                        if line.strip()
                    ).items())
                ),
                "trajectory_count": len(trajectories),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
