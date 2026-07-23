"""Aggregate-only validation for one external blinded author shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_wallet.training.blinded_authoring import (
    EXPECTED_SHARD_PREFIXES,
    _read_jsonl,
    _validate_source_shard,
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
        cases, _ = _validate_source_shard(
            _read_jsonl(args.source), args.prefix
        )
    except Exception:
        print(json.dumps({"valid": False}, sort_keys=True))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {"case_count": len(cases), "valid": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
