"""Create digest-only metadata for a sealed suite stored outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_wallet.benchmark import load_cases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMITMENT = ROOT / "data" / "benchmark" / "sealed-suite-v1.commitment.json"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--author-role", required=True)
    parser.add_argument("--rubric-version", default="sealed-wallet-eval-v1")
    parser.add_argument("--output", type=Path, default=DEFAULT_COMMITMENT)
    parser.add_argument("--attest-independent-author", action="store_true")
    args = parser.parse_args()

    if not args.attest_independent_author:
        raise SystemExit("independent-author attestation is required")
    if _is_within(args.suite, ROOT):
        raise SystemExit("sealed plaintext must remain outside the training checkout")
    payload = args.suite.read_bytes()
    cases = load_cases(args.suite)
    if any(case.family != "sealed" for case in cases):
        raise SystemExit("every sealed-suite record must use family=sealed")
    case_count = len(cases)
    if case_count < 20:
        raise SystemExit("sealed suite must contain at least 20 non-empty records")
    commitment = {
        "author_independence_attested": True,
        "author_role": args.author_role,
        "case_count": case_count,
        "created_at": datetime.now(UTC).isoformat(),
        "rubric_version": args.rubric_version,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": "committed-before-training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(commitment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(commitment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
