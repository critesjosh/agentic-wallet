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
SHARD_SIZE = 16
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
                "minItems": SHARD_SIZE,
                "maxItems": SHARD_SIZE,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "scenario_id": {"type": "string", "minLength": 1},
                        "scenario_type": {
                            "type": "string",
                            "enum": list(SCENARIO_TYPES),
                        },
                        "user_request": {"type": "string", "minLength": 1},
                        "workflow_state": {"type": "string", "minLength": 1},
                        "context": {
                            "type": "object",
                            "minProperties": 1,
                            "additionalProperties": True,
                        },
                        "trajectory_id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}]
                        },
                        "turn_index": {
                            "anyOf": [
                                {"type": "integer", "minimum": 0},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": list(TOP_LEVEL_FIELDS),
                    "additionalProperties": False,
                },
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
        "temperature": 0.7,
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
    content = result["choices"][0]["message"]["content"]
    value = json.loads(content)
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != SHARD_SIZE:
        raise SystemExit("Claude response did not contain exactly 16 cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
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
