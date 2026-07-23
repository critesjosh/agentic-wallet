"""Evaluate preserved v5 checkpoints without retraining.

The source tree and adapter bucket are mounted read-only. Only JSON reports and
logs are written to the configured output directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(os.environ.get("AGENTIC_WALLET_WORKSPACE", "/workspace"))
CHECKPOINT_ROOT = Path(
    os.environ.get(
        "AGENTIC_WALLET_CHECKPOINT_ROOT",
        "/training/e2b-qlora-smoke-20260722T203608Z/adapter",
    )
)
OUTPUT_DIR = Path(
    os.environ.get("AGENTIC_WALLET_OUTPUT_DIR", "/outputs/v5-checkpoint-evaluation")
)
V5_DATASET = WORKSPACE / "data" / "training" / "sft-v5-candidate-binding.jsonl"
INDEPENDENT_DATASET = (
    WORKSPACE / "data" / "benchmark" / "independent-route-v1.jsonl"
)


def _evaluate(
    *,
    checkpoint: str,
    dataset: Path,
    label: str,
    repeat: int = 1,
) -> None:
    for index in range(1, repeat + 1):
        suffix = f"-repeat-{index}" if repeat > 1 else ""
        name = f"{checkpoint}-{label}{suffix}"
        command = [
            sys.executable,
            str(WORKSPACE / "scripts" / "evaluate_development.py"),
            "--provider",
            "transformers",
            "--dataset",
            str(dataset),
            "--adapter-path",
            str(CHECKPOINT_ROOT / checkpoint),
            "--json-output",
            str(OUTPUT_DIR / f"{name}.json"),
        ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        (OUTPUT_DIR / f"{name}.log").write_text(
            completed.stdout + completed.stderr
        )
        print(completed.stdout, end="", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "source_revision": os.environ.get("AGENTIC_WALLET_SOURCE_REVISION"),
        "complete": False,
    }
    (OUTPUT_DIR / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    try:
        for checkpoint in ("checkpoint-25", "checkpoint-50"):
            _evaluate(
                checkpoint=checkpoint,
                dataset=V5_DATASET,
                label="development",
            )
            _evaluate(
                checkpoint=checkpoint,
                dataset=INDEPENDENT_DATASET,
                label="independent",
            )
        _evaluate(
            checkpoint="checkpoint-75",
            dataset=V5_DATASET,
            label="development",
            repeat=2,
        )
        _evaluate(
            checkpoint="checkpoint-75",
            dataset=INDEPENDENT_DATASET,
            label="independent",
            repeat=2,
        )
        status["complete"] = True
    finally:
        (OUTPUT_DIR / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
