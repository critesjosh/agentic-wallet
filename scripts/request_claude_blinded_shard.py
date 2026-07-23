"""Request one schema-constrained blinded shard through Claude Code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

MODEL = "sonnet"
SHARD_SIZE = 8
ROOT = Path(__file__).resolve().parents[1]
SHARED_PROMPT = ROOT / "docs" / "claude-blinded-author-shared-v2.md"
SHARD_PROMPTS = frozenset(
    ROOT / "docs" / f"claude-blinded-author-shard-{name}-v2.md"
    for name in ("1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b")
)
TOP_LEVEL_FIELDS = (
    "id",
    "scenario_id",
    "scenario_type",
    "user_request",
    "workflow_state",
    "context",
    "trajectory_id",
    "turn_index",
)
SCENARIO_TYPES = (
    "cancel_workflow",
    "conceptual_help",
    "duplicate_plan",
    "exact_approval",
    "prompt_injection",
    "quote_expired",
    "read_allowances",
    "read_balance",
    "read_portfolio",
    "read_registry",
    "signing_boundary",
    "simulation_match",
    "simulation_mismatch",
    "stale_portfolio",
    "swap_quote",
    "transfer_ambiguous_asset",
    "transfer_complete",
    "transfer_missing",
    "transfer_missing_recipient",
    "transfer_untrusted_directory",
    "transfer_wrong_chain",
    "unlimited_approval_attack",
    "unsupported_request",
)


def _within_checkout(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["cases"],
        "additionalProperties": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-prompt", type=Path, required=True)
    parser.add_argument("--shard-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _within_checkout(args.output):
        raise SystemExit("blinded shard output must stay outside checkout")
    if args.shared_prompt.resolve() != SHARED_PROMPT.resolve():
        raise SystemExit("shared prompt does not match the frozen author packet")
    if args.shard_prompt.resolve() not in {
        path.resolve() for path in SHARD_PROMPTS
    }:
        raise SystemExit("shard prompt does not match the frozen author packet")
    if args.output.exists():
        raise SystemExit("blinded shard output already exists")
    prompt = (
        args.shared_prompt.read_text()
        + "\n\n"
        + args.shard_prompt.read_text()
        + "\n\nReturn a JSON object whose cases array contains the records."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            MODEL,
            "--tools",
            "",
            "--safe-mode",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(_schema(), separators=(",", ":")),
            prompt,
        ],
        cwd=args.output.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise SystemExit("Claude Code author request failed")
    result = json.loads(completed.stdout)
    if result.get("is_error"):
        raise SystemExit("Claude Code author request failed")
    value = result.get("structured_output")
    if not isinstance(value, dict):
        raise SystemExit("Claude Code did not return structured output")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != SHARD_SIZE:
        raise SystemExit(f"Claude response did not contain exactly {SHARD_SIZE} cases")
    normalized: list[dict[str, Any]] = []
    for encoded_case in cases:
        try:
            case = json.loads(encoded_case)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SystemExit("Claude returned an invalid encoded case") from exc
        if not isinstance(case, dict) or set(case) != set(TOP_LEVEL_FIELDS):
            raise SystemExit("Claude encoded case has invalid top-level fields")
        if case["scenario_type"] not in SCENARIO_TYPES:
            raise SystemExit("Claude encoded case has an invalid scenario type")
        if not isinstance(case["context"], dict):
            raise SystemExit("Claude encoded context must be an object")
        normalized.append(case)
    payload = "".join(
        json.dumps(case, sort_keys=True) + "\n" for case in normalized
    ).encode()
    try:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SystemExit("blinded shard output already exists") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "interface": "claude-code-cli",
                "model": MODEL,
                "structured_output": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
