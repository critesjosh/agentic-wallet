"""Request one schema-constrained Claude blinded-suite shard from OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "anthropic/claude-fable-5"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
SHARD_SIZE = 8
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
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    prompt = (
        args.shared_prompt.read_text()
        + "\n\n"
        + args.shard_prompt.read_text()
        + "\n\nReturn a JSON object whose cases array contains the records."
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 24_000,
        "provider": {
            "require_parameters": True,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "claude_blinded_wallet_shard",
                "strict": True,
                "schema": _schema(),
            },
        },
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        error = exc.read().decode(errors="replace")
        raise SystemExit(f"OpenRouter request failed ({exc.code}): {error}") from exc
    choice = result["choices"][0]
    content = choice["message"]["content"]
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "Claude structured response was invalid; "
            f"finish_reason={choice.get('finish_reason')!r}"
        ) from exc
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in normalized)
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "model": result.get("model", MODEL),
                "provider": result.get("provider"),
                "structured_output": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
