#!/usr/bin/env python3
"""Stage a sanitized workspace and launch the bounded QLoRA job on HF Jobs.

Only the files the job actually reads are staged, and the set is taken from
tracked Git content so an untracked secret cannot ride along. Nothing is
uploaded or spent without ``--execute``; the default prints the exact file
manifest and command instead.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirrors the workspace paths scripts/run_hf_qlora_smoke.py reads, so the job's
# own source digest is computed over the same set that was staged.
_PATTERNS = (
    "pyproject.toml",
    "scripts/evaluate_adapter.py",
    "scripts/evaluate_development.py",
    "scripts/train_qlora.py",
    "src/**/*.py",
    "src/agentic_wallet/SKILL.md",
    "src/agentic_wallet/skills/*.md",
    "data/benchmark/*.jsonl",
    "data/training/*.jsonl",
    "data/training/*.manifest.json",
)

# A staged file matching any of these never leaves the machine.
_FORBIDDEN_NAME = re.compile(r"(?:^|/)\.env$|\.pem$|\.key$|(?:^|/)id_(?:rsa|ed25519)$")
_FORBIDDEN_CONTENT = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
)


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True, capture_output=True, text=True,
    )
    return set(result.stdout.split())


def staged_paths() -> list[Path]:
    tracked = tracked_files()
    selected: set[Path] = set()
    for pattern in _PATTERNS:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            # Untracked files are excluded by construction: .env and any local
            # scratch artifact can never be staged.
            if relative in tracked:
                selected.add(path)
    return sorted(selected)


def assert_no_secrets(paths: list[Path]) -> None:
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if _FORBIDDEN_NAME.search(relative):
            raise SystemExit(f"refusing to stage {relative}")
        data = path.read_bytes()
        for pattern in _FORBIDDEN_CONTENT:
            if pattern.search(data):
                raise SystemExit(f"refusing to stage {relative}: contains a secret-shaped value")


def stage(paths: list[Path], destination: Path) -> str:
    if destination.exists():
        shutil.rmtree(destination)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest.update(relative.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="data/training/sft-v7-account-identity.jsonl"
    )
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--flavor", default="l4x1")
    parser.add_argument(
        "--evaluate-base-only",
        action="store_true",
        help=(
            "Evaluate the untuned base INSTEAD of training. "
            "AGENTIC_WALLET_EVALUATE_BASE short-circuits the training branch, "
            "so this cannot be combined with a training run."
        ),
    )
    parser.add_argument(
        "--skill-eval-adapter",
        default=None,
        help=(
            "Instead of training, run the skill on/off A/B against this adapter "
            "bucket path, e.g. e2b-qlora-smoke-20260724T104217Z/adapter."
        ),
    )
    parser.add_argument(
        "--skill-sweep-adapter",
        default=None,
        help=(
            "Instead of training, sweep several skill variants against this "
            "adapter bucket path in a single job."
        ),
    )
    parser.add_argument(
        "--output-bucket", default="hf://buckets/critesjosh/agentic-wallet-smoke"
    )
    parser.add_argument("--stage-dir", type=Path, default=ROOT / ".hf-workspace")
    parser.add_argument("--name", default="agentic-wallet-v7-qlora")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not (ROOT / args.dataset).is_file():
        raise SystemExit(f"dataset not found: {args.dataset}")

    paths = staged_paths()
    assert_no_secrets(paths)
    tree_digest = stage(paths, args.stage_dir)
    dataset_digest = hashlib.sha256((ROOT / args.dataset).read_bytes()).hexdigest()

    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    # Resolve the CLI next to this interpreter so the run does not depend on
    # whichever PATH the caller's shell happens to have.
    hf_cli = Path(sys.executable).parent / "hf"
    if not hf_cli.is_file():
        raise SystemExit(f"hf CLI not found at {hf_cli}")

    if args.skill_eval_adapter or args.skill_sweep_adapter:
        # The adapter already lives inside the output bucket, so read it through
        # the same /outputs mount rather than mounting the bucket twice.
        adapter = args.skill_eval_adapter or args.skill_sweep_adapter
        if args.skill_sweep_adapter:
            job_script = "run_hf_skill_sweep.py"
            mode = f"skill variant sweep against {adapter}, NO TRAINING"
            name = f"{args.name}-skill-sweep"
        else:
            job_script = "run_hf_skill_eval.py"
            mode = f"skill on/off A/B against {adapter}, NO TRAINING"
            name = f"{args.name}-skill-eval"
        command = [
            str(hf_cli), "jobs", "uv", "run",
            "--flavor", args.flavor,
            "--name", name,
            "-s", "HF_TOKEN",
            "-v", f"{args.stage_dir}:/workspace:ro",
            "-v", f"{args.output_bucket}:/outputs",
            "-e", f"AGENTIC_WALLET_DATASET=/workspace/{args.dataset}",
            "-e", f"AGENTIC_WALLET_ADAPTER=/outputs/{adapter}",
            "-e", f"AGENTIC_WALLET_SOURCE_REVISION={revision}",
            str(ROOT / "scripts" / job_script),
        ]
    else:
        command = [
            str(hf_cli), "jobs", "uv", "run",
            "--flavor", args.flavor,
            "--name", args.name,
            "-s", "HF_TOKEN",
            "-v", f"{args.stage_dir}:/workspace:ro",
            "-v", f"{args.output_bucket}:/outputs",
            "-e", f"AGENTIC_WALLET_DATASET=/workspace/{args.dataset}",
            "-e", f"AGENTIC_WALLET_MAX_STEPS={args.max_steps}",
            "-e", f"AGENTIC_WALLET_SOURCE_REVISION={revision}",
            # train_qlora.py checkpoints and evaluates every 25 steps by
            # default, which is what makes early-stopping comparison possible.
            *(
                ["-e", "AGENTIC_WALLET_EVALUATE_BASE=1",
                 "-e", "AGENTIC_WALLET_EVALUATE_DEVELOPMENT_BASE=1"]
                if args.evaluate_base_only
                else []
            ),
            str(ROOT / "scripts" / "run_hf_qlora_smoke.py"),
        ]
        mode = (
            "evaluate untuned base, NO TRAINING" if args.evaluate_base_only
            else f"train {args.max_steps} steps, checkpoints every 25"
        )

    print(f"staged files      : {len(paths)}")
    print(f"stage directory   : {args.stage_dir}")
    print(f"source tree sha256: {tree_digest}")
    print(f"dataset sha256    : {dataset_digest}")
    print(f"dataset           : {args.dataset}")
    print(f"flavor            : {args.flavor}")
    print(f"mode              : {mode}")
    print()
    print(" ".join(command))

    if not args.execute:
        print("\nDry run. Re-run with --execute to upload and start the job.")
        return 0

    print("\nlaunching...")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
