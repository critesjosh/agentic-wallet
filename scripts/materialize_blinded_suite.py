"""Compile eight external author shards into deterministic benchmark JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentic_wallet.training.blinded_authoring import (
    canonical_suite_bytes,
    materialize_author_shards,
)

ROOT = Path(__file__).resolve().parents[1]


def _within_checkout(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SystemExit("blinded plaintext output already exists") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(_within_checkout(path) for path in [*args.source, args.output]):
        raise SystemExit(
            "blinded sources and plaintext suite must stay outside checkout"
        )

    try:
        cases, receipt = materialize_author_shards(args.source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    _write_new(args.output, canonical_suite_bytes(cases))
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "hard_zero_count": sum(
                    case.hard_zero_category is not None for case in cases
                ),
                "scenario_counts": receipt["scenario_counts"],
                "trajectory_count": receipt["trajectory_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
