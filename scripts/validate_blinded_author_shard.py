"""Aggregate-only validation for one external blinded author shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_wallet.training.blinded_authoring import (
    EXPECTED_SHARD_PREFIXES,
    SOURCE_JSON_INVALID_CODE,
    _read_jsonl,
    author_shard_validation_report,
)

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
    parser.add_argument("--prefix", choices=EXPECTED_SHARD_PREFIXES, required=True)
    args = parser.parse_args()
    if _within_checkout(args.source):
        raise SystemExit("author shard must stay outside checkout")
    try:
        report = author_shard_validation_report(
            _read_jsonl(args.source), args.prefix
        )
    except Exception:
        print(
            json.dumps(
                {
                    "issues": [{"code": SOURCE_JSON_INVALID_CODE}],
                    "valid": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    if not report["valid"]:
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(1)
    print(
        json.dumps(
            {"case_count": 8, "issues": [], "valid": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
