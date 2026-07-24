# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.10",
#   "bitsandbytes>=0.49",
#   "datasets>=4.0",
#   "eth-utils>=5,<7",
#   "peft>=0.19",
#   "pydantic>=2.6",
#   "torch>=2.7",
#   "torchvision>=0.22",
#   "transformers>=5.10.1",
# ]
# ///
"""Sweep several skill variants against one adapter in a single job.

The adapter is loaded once; each variant is evaluated on the same development
split by swapping the provider's routing and argument skills. This keeps the
comparison clean (one model load, one dataset) and cheap (one job).

Never trains, never signs. The repository is mounted read-only at ``/workspace``
and the adapter at ``AGENTIC_WALLET_ADAPTER``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path(os.environ.get("AGENTIC_WALLET_WORKSPACE", "/workspace"))
OUTPUT_ROOT = Path(os.environ.get("AGENTIC_WALLET_OUTPUT_ROOT", "/outputs"))
ADAPTER = Path(os.environ.get("AGENTIC_WALLET_ADAPTER", "/adapter"))
DATASET_PATH = Path(
    os.environ.get(
        "AGENTIC_WALLET_DATASET",
        str(WORKSPACE / "data" / "training" / "sft-v7-account-identity.jsonl"),
    )
)

sys.path.insert(0, str(WORKSPACE / "src"))

SKILLS = WORKSPACE / "src" / "agentic_wallet" / "skills"
ROUTE_V1 = WORKSPACE / "src" / "agentic_wallet" / "SKILL.md"

# (name, routing skill file, argument skill file). None means that phase gets
# no skill.
VARIANTS = [
    ("baseline", None, None),
    ("route_v1", ROUTE_V1, None),
    ("route_v2", SKILLS / "route_v2.md", None),
    ("arguments_v1", None, SKILLS / "arguments_v1.md"),
    ("route_v2_arguments_v1", SKILLS / "route_v2.md", SKILLS / "arguments_v1.md"),
]


def main() -> None:
    from agentic_wallet.providers import LocalTransformersProvider
    from agentic_wallet.skill import load_skill
    from agentic_wallet.training import (
        evaluate_development_examples,
        load_training_examples,
    )

    if not (ADAPTER / "adapter_config.json").is_file():
        raise SystemExit(f"adapter mount not found at {ADAPTER}")

    examples = [
        example
        for example in load_training_examples(DATASET_PATH)
        if example.split == "validation"
    ]
    if not examples:
        raise SystemExit("dataset has no validation examples")

    provider = LocalTransformersProvider(adapter_path=str(ADAPTER))
    provider.load()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"v7-skill-sweep-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, route_file, arg_file in VARIANTS:
        provider.skill = load_skill(route_file) if route_file else None
        provider.argument_skill = load_skill(arg_file) if arg_file else None
        report = evaluate_development_examples(provider, examples).to_dict()
        payload = {
            "variant": name,
            "route_skill": str(route_file) if route_file else None,
            "argument_skill": str(arg_file) if arg_file else None,
            "adapter": str(ADAPTER),
            **report,
        }
        (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))
        summary[name] = {
            "exact_accuracy": report["exact_accuracy"],
            "action_accuracy": report["action_accuracy"],
            "schema_valid_rate": report["schema_valid_rate"],
            "safety_failures": report["safety_failures"],
            "by_argument_count": {
                bucket: report["by_argument_count"][bucket]["exact_passed"]
                for bucket in ("zero", "single", "multi")
            },
        }
        print(json.dumps({name: summary[name]}))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
