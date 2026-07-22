"""Generate and validate the deterministic SFT dataset and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agentic_wallet.benchmark import BENCHMARK_DATASET_ROLE, load_cases
from agentic_wallet.training import (
    ERROR_DRIVEN_GENERATOR_VERSION,
    GENERATOR_VERSION,
    NATURAL_CURRICULUM_VERSION,
    PIPELINE_CURRICULUM_VERSION,
    generate_error_driven_training_examples,
    generate_training_examples,
    load_natural_curriculum,
    load_pipeline_curriculum,
    validate_training_dataset,
)
from agentic_wallet.training.config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DATASET_VERSION,
    ERROR_DRIVEN_DATASET_VERSION,
    WORKFLOW_DATASET_VERSION,
    PIPELINE_DATASET_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark"
PROFILES = {
    "v1": {
        "output": ROOT / "data" / "training" / "sft-v1.jsonl",
        "tool_count": 96,
        "dialogue_count": 48,
        "seed": 17,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generate": generate_training_examples,
    },
    "error-driven-v2": {
        "output": ROOT / "data" / "training" / "sft-v2-error-driven.jsonl",
        "tool_count": 504,
        "dialogue_count": 72,
        "seed": 29,
        "dataset_version": ERROR_DRIVEN_DATASET_VERSION,
        "generator_version": ERROR_DRIVEN_GENERATOR_VERSION,
        "generate": generate_error_driven_training_examples,
    },
    "workflow-v3": {
        "output": ROOT / "data" / "training" / "sft-v3-workflow.jsonl",
        "source": ROOT / "data" / "training" / "natural_v3_source.jsonl",
        "tool_count": None,
        "dialogue_count": None,
        "seed": None,
        "dataset_version": WORKFLOW_DATASET_VERSION,
        "generator_version": NATURAL_CURRICULUM_VERSION,
        "generate": None,
        "load": load_natural_curriculum,
    },
    "pipeline-v4": {
        "output": ROOT / "data" / "training" / "sft-v4-pipeline.jsonl",
        "source": ROOT / "data" / "training" / "natural_v3_source.jsonl",
        "tool_count": None,
        "dialogue_count": None,
        "seed": None,
        "dataset_version": PIPELINE_DATASET_VERSION,
        "generator_version": PIPELINE_CURRICULUM_VERSION,
        "generate": None,
        "load": load_pipeline_curriculum,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tool-count", type=int, default=None)
    parser.add_argument("--dialogue-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    output = args.output or profile["output"]
    tool_count = args.tool_count or profile["tool_count"]
    dialogue_count = args.dialogue_count or profile["dialogue_count"]
    seed = args.seed if args.seed is not None else profile["seed"]

    benchmark_paths = [
        BENCHMARK / "train_family.jsonl",
        BENCHMARK / "eval_family.jsonl",
    ]
    frozen = [case for path in benchmark_paths for case in load_cases(path)]
    source = profile.get("source")
    if source is not None:
        if any(value is not None for value in (args.tool_count, args.dialogue_count, args.seed)):
            raise ValueError("static natural curriculum does not accept count or seed overrides")
        examples = profile["load"](source)
    else:
        examples = profile["generate"](
            tool_count=tool_count,
            dialogue_count=dialogue_count,
            seed=seed,
        )
    report = validate_training_dataset(examples, frozen)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(example.model_dump(), sort_keys=True, separators=(",", ":")) + "\n"
        for example in examples
    )
    output.write_text(payload)
    manifest = {
        "dataset_version": profile["dataset_version"],
        "generator_version": profile["generator_version"],
        "profile": args.profile,
        "seed": seed,
        "source": (
            {"path": str(source.relative_to(ROOT)), "sha256": _sha256(source)}
            if source is not None
            else None
        ),
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "dataset_sha256": _sha256(output),
        "frozen_benchmark": {
            path.name: _sha256(path) for path in benchmark_paths
        },
        "benchmark_role": BENCHMARK_DATASET_ROLE,
        "validation": {
            "total": report.total,
            "tool_calls": report.tool_calls,
            "dialogue_turns": report.dialogue_turns,
            "dialogue_routes": report.dialogue_routes,
            "label_counts": report.label_counts,
            "max_benchmark_similarity": round(report.max_benchmark_similarity, 6),
            "coverage_counts": report.coverage_counts,
            "split_counts": report.split_counts,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
