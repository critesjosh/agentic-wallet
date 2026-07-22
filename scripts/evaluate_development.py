"""Evaluate a fixed non-sealed validation split through a local runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_wallet.providers import LocalTransformersProvider, OllamaProvider
from agentic_wallet.training import (
    evaluate_development_examples,
    load_training_examples,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "training" / "sft-v3-workflow.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=("ollama", "transformers"), default="ollama"
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    examples = [
        example
        for example in load_training_examples(args.dataset)
        if example.split == "validation"
    ]
    if not examples:
        raise SystemExit("dataset has no validation examples")
    if args.provider == "ollama":
        if args.adapter_path is not None:
            raise SystemExit("--adapter-path is available only for Transformers")
        provider = OllamaProvider(
            model=args.model_id or "gemma4:e2b", base_url=args.base_url
        )
    else:
        provider = LocalTransformersProvider(
            model_id=args.model_id or "google/gemma-4-E2B-it",
            adapter_path=(
                str(args.adapter_path) if args.adapter_path is not None else None
            ),
        )
        provider.load()

    report = evaluate_development_examples(provider, examples)
    payload = {
        "benchmark_role": "development-validation-only",
        "provider": provider.name,
        "model": getattr(provider, "model", getattr(provider, "model_id", None)),
        "native_constrained_decoding": provider.native_constrained_decoding,
        **report.to_dict(),
    }
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2, sort_keys=True))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
