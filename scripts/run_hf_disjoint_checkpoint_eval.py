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
"""Per-checkpoint generalization curve for the V7 adapter.

For each checkpoint, evaluate both the in-distribution validation split and the
disjoint routing suite. The gap between them, and how the disjoint score moves
across checkpoints, is the overfitting signal: if later steps raise the
in-distribution score while the disjoint score falls, the model is memorizing
the training distribution rather than generalizing.

Never trains, never signs.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path(os.environ.get("AGENTIC_WALLET_WORKSPACE", "/workspace"))
OUTPUT_ROOT = Path(os.environ.get("AGENTIC_WALLET_OUTPUT_ROOT", "/outputs"))
ADAPTER = Path(os.environ.get("AGENTIC_WALLET_ADAPTER", "/adapter"))

IN_DIST = WORKSPACE / "data" / "training" / "sft-v7-account-identity.jsonl"
DISJOINT = WORKSPACE / "data" / "benchmark" / "independent-route-v7.jsonl"

sys.path.insert(0, str(WORKSPACE / "src"))

CHECKPOINTS = ("checkpoint-25", "checkpoint-50", "checkpoint-75")


def main() -> None:
    import torch

    from agentic_wallet.providers import LocalTransformersProvider
    from agentic_wallet.training import (
        evaluate_development_examples,
        load_training_examples,
    )

    datasets = {
        "in_distribution": [
            e for e in load_training_examples(IN_DIST) if e.split == "validation"
        ],
        "disjoint": [
            e for e in load_training_examples(DISJOINT) if e.split == "validation"
        ],
    }
    for name, examples in datasets.items():
        if not examples:
            raise SystemExit(f"{name} dataset has no validation examples")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"v7-disjoint-checkpoint-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for checkpoint in CHECKPOINTS:
        adapter_path = ADAPTER / checkpoint
        if not (adapter_path / "adapter_config.json").is_file():
            raise SystemExit(f"checkpoint not found: {adapter_path}")
        # Fresh provider per checkpoint so each adapter loads cleanly on the base.
        provider = LocalTransformersProvider(adapter_path=str(adapter_path))
        provider.load()
        summary[checkpoint] = {}
        for name, examples in datasets.items():
            report = evaluate_development_examples(provider, examples).to_dict()
            payload = {"checkpoint": checkpoint, "dataset": name, **report}
            (output_dir / f"{checkpoint}-{name}.json").write_text(
                json.dumps(payload, indent=2)
            )
            summary[checkpoint][name] = {
                "exact_accuracy": report["exact_accuracy"],
                "action_accuracy": report["action_accuracy"],
                "safety_failures": report["safety_failures"],
                "total": report["total"],
            }
            print(json.dumps({checkpoint: {name: summary[checkpoint][name]}}))
        del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The overfitting read: in-distribution minus disjoint, per checkpoint.
    for checkpoint, data in summary.items():
        gap = data["in_distribution"]["exact_accuracy"] - data["disjoint"]["exact_accuracy"]
        data["generalization_gap"] = round(gap, 4)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
