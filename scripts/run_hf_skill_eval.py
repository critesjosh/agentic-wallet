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
"""Skill on/off A/B for an already-trained adapter on Hugging Face Jobs.

The repository is mounted read-only at ``/workspace`` and the trained adapter at
``/adapter``. This evaluates the same adapter twice on the same development
split, changing only whether the inference-time routing skill is prepended, and
writes both reports to ``/outputs``. It never trains and never signs.
"""

from __future__ import annotations

import json
import os
import subprocess
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


def _run(output: Path, *, skill: bool) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKSPACE / "src")
    command = [
        sys.executable,
        str(WORKSPACE / "scripts" / "evaluate_development.py"),
        "--provider", "transformers",
        "--dataset", str(DATASET_PATH),
        "--adapter-path", str(ADAPTER),
        "--json-output", str(output),
    ]
    if skill:
        command.append("--skill")
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    (output.with_suffix(".log")).write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"evaluation failed (skill={skill})")


def main() -> None:
    if not (WORKSPACE / "scripts" / "evaluate_development.py").is_file():
        raise SystemExit(f"workspace mount not found at {WORKSPACE}")
    if not (ADAPTER / "adapter_config.json").is_file():
        raise SystemExit(f"adapter mount not found at {ADAPTER}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"v7-skill-eval-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _run(output_dir / "adapter_skill_off.json", skill=False)
    _run(output_dir / "adapter_skill_on.json", skill=True)

    (output_dir / "status.json").write_text(
        json.dumps(
            {
                "adapter": str(ADAPTER),
                "dataset": str(DATASET_PATH),
                "skill_off": True,
                "skill_on": True,
                "source_revision": os.environ.get("AGENTIC_WALLET_SOURCE_REVISION"),
            },
            indent=2,
        )
    )
    print(json.dumps({"output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
