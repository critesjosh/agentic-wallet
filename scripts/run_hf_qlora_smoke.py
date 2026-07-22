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
"""Bounded Hugging Face Jobs smoke test for the Gemma 4 E2B adapter path.

The repository is expected at ``/workspace`` on a read-only mount and a private
artifact bucket at ``/outputs``. The bounded step count and dataset are supplied
explicitly, then the script reloads the adapter and runs the frozen suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


WORKSPACE = Path(os.environ.get("AGENTIC_WALLET_WORKSPACE", "/workspace"))
OUTPUT_ROOT = Path(os.environ.get("AGENTIC_WALLET_OUTPUT_ROOT", "/outputs"))
MAX_STEPS = int(os.environ.get("AGENTIC_WALLET_MAX_STEPS", "20"))
DATASET_PATH = Path(
    os.environ.get(
        "AGENTIC_WALLET_DATASET",
        str(WORKSPACE / "data" / "training" / "sft-v1.jsonl"),
    )
)
EXISTING_ADAPTER = os.environ.get("AGENTIC_WALLET_EXISTING_ADAPTER")
EVALUATE_BASE = os.environ.get("AGENTIC_WALLET_EVALUATE_BASE") == "1"


def _source_tree_sha256() -> str:
    """Hash training-relevant source and data without relying on Git metadata."""

    paths = [
        WORKSPACE / "pyproject.toml",
        WORKSPACE / "scripts" / "evaluate_adapter.py",
        WORKSPACE / "scripts" / "train_qlora.py",
        *sorted((WORKSPACE / "src").rglob("*.py")),
        *sorted((WORKSPACE / "data" / "benchmark").glob("*.jsonl")),
        *sorted((WORKSPACE / "data" / "training").glob("*.jsonl")),
        *sorted((WORKSPACE / "data" / "training").glob("*.manifest.json")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(WORKSPACE).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _emit(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)


def main() -> None:
    if not (WORKSPACE / "scripts" / "train_qlora.py").is_file():
        raise SystemExit(f"sanitized repository mount not found at {WORKSPACE}")
    if not 1 <= MAX_STEPS <= 200:
        raise SystemExit("bounded experiment max steps must remain between 1 and 200")
    if not DATASET_PATH.is_file():
        raise SystemExit(f"training dataset not found: {DATASET_PATH}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"e2b-qlora-smoke-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    source_tree_sha256 = _source_tree_sha256()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE / "src")
    env["AGENTIC_WALLET_SOURCE_TREE_SHA256"] = source_tree_sha256

    import torch

    environment = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_bf16_supported": (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        ),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "dataset": str(DATASET_PATH),
        "max_steps": MAX_STEPS,
        "python": sys.version,
        "source_revision": os.environ.get("AGENTIC_WALLET_SOURCE_REVISION"),
        "source_tree_sha256": source_tree_sha256,
        "torch": torch.__version__,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(environment, indent=2, sort_keys=True), flush=True)

    status = {"output_dir": str(output_dir), "training": False, "evaluation": False}
    try:
        adapter_path: Path | None
        if EVALUATE_BASE:
            adapter_path = None
            status["training"] = "not-applicable"
            status["evaluation_target"] = "untuned-base"
        elif EXISTING_ADAPTER:
            adapter_path = Path(EXISTING_ADAPTER)
            if not adapter_path.is_dir():
                raise SystemExit(f"existing adapter not found: {adapter_path}")
            status["training"] = "reused"
            status["existing_adapter"] = str(adapter_path)
        else:
            adapter_path = output_dir / "adapter"
            training = subprocess.run(
                [
                    sys.executable,
                    str(WORKSPACE / "scripts" / "train_qlora.py"),
                    "--execute",
                    "--acknowledge-p2-gate",
                    "--max-steps",
                    str(MAX_STEPS),
                    "--dataset",
                    str(DATASET_PATH),
                    "--output-dir",
                    str(adapter_path),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            _emit(training)
            (output_dir / "training.log").write_text(
                training.stdout + training.stderr
            )
            status["training"] = True

        evaluation_command = [
            sys.executable,
            str(WORKSPACE / "scripts" / "evaluate_adapter.py"),
            "--max-new-tokens",
            "256",
            "--json-output",
            str(output_dir / "evaluation.json"),
        ]
        if adapter_path is not None:
            evaluation_command.extend(["--adapter-path", str(adapter_path)])
        evaluation = subprocess.run(
            evaluation_command,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
        _emit(evaluation)
        (output_dir / "evaluation.log").write_text(
            evaluation.stdout + evaluation.stderr
        )
        status["evaluation"] = True
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr, flush=True)
        status["failed_command"] = exc.cmd
        status["returncode"] = exc.returncode
        (output_dir / "failure.log").write_text(
            (exc.stdout or "") + (exc.stderr or "")
        )
        raise
    finally:
        (output_dir / "smoke_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
