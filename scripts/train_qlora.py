"""Reproducible, completion-only QLoRA plumbing for Gemma 4 E2B.

Without ``--execute`` this performs a CPU-safe dataset/configuration dry run.
Actual training additionally requires an explicit P2 acknowledgement and a
BF16-capable CUDA GPU. It never pushes an artifact to the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from agentic_wallet.benchmark import load_cases
from agentic_wallet.training import (
    balanced_semantic_subset,
    evaluate_development_examples,
    load_training_examples,
    tokenize_completion_only,
    validate_sealed_commitment,
    validate_training_dataset,
)
from agentic_wallet.training.config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    LORA_EXCLUDE_PATTERN,
    LORA_TARGET_MODULES,
    SUPPORTED_DATASET_VERSIONS,
    PIPELINE_DATASET_VERSION,
    WORKFLOW_DATASET_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "training" / "sft-v1.jsonl"
DEFAULT_SEALED_COMMITMENT = (
    ROOT / "data" / "benchmark" / "sealed-suite-v1.commitment.json"
)
BENCHMARK_DIR = ROOT / "data" / "benchmark"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str | None:
    explicit = os.environ.get("AGENTIC_WALLET_SOURCE_REVISION")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_inputs(dataset: Path, manifest: Path) -> tuple[list, dict, Any]:
    examples = load_training_examples(dataset)
    benchmark_paths = [
        BENCHMARK_DIR / "train_family.jsonl",
        BENCHMARK_DIR / "eval_family.jsonl",
    ]
    frozen = [case for path in benchmark_paths for case in load_cases(path)]
    report = validate_training_dataset(examples, frozen)
    metadata = json.loads(manifest.read_text())
    if metadata["dataset_version"] not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError("unexpected dataset version")
    if metadata["dataset_sha256"] != _sha256(dataset):
        raise ValueError("dataset digest does not match its manifest")
    for path in benchmark_paths:
        if metadata["frozen_benchmark"].get(path.name) != _sha256(path):
            raise ValueError(f"frozen benchmark digest changed: {path.name}")
    if metadata["base_model_id"] != BASE_MODEL_ID:
        raise ValueError("manifest base model does not match training config")
    if metadata["base_model_revision"] != BASE_MODEL_REVISION:
        raise ValueError("manifest base revision does not match training config")
    return examples, metadata, report


def _collator(tokenizer: Any):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor tokenizer has no pad token id")

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        keys = set.intersection(*(set(feature) for feature in features))
        for key in keys:
            padding = -100 if key == "labels" else (pad_token_id if key == "input_ids" else 0)
            values = [torch.tensor(feature[key], dtype=torch.long) for feature in features]
            batch[key] = pad_sequence(values, batch_first=True, padding_value=padding)
        return batch

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "e2b-qlora-smoke")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--semantic-eval-limit", type=int, default=20)
    parser.add_argument(
        "--sealed-commitment", type=Path, default=DEFAULT_SEALED_COMMITMENT
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-p2-gate", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest or args.dataset.with_suffix(".manifest.json")
    examples, dataset_metadata, report = _validate_inputs(args.dataset, manifest)
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [
        example for example in examples if example.split == "validation"
    ]
    if dataset_metadata["dataset_version"] in {
        WORKFLOW_DATASET_VERSION,
        PIPELINE_DATASET_VERSION,
    }:
        if not train_examples or not validation_examples:
            raise ValueError(
                "workflow datasets require explicit train and validation splits"
            )
    if args.eval_steps <= 0 or args.save_steps <= 0:
        raise ValueError("eval and save steps must be positive")
    if args.eval_steps != args.save_steps:
        raise ValueError("eval and save steps must match for checkpoint selection")

    run_plan = {
        "mode": "execute" if args.execute else "dry-run",
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "dataset_version": dataset_metadata["dataset_version"],
        "dataset_sha256": dataset_metadata["dataset_sha256"],
        "examples": report.total,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "rank": args.rank,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "semantic_eval_limit": args.semantic_eval_limit,
        "target_modules": list(LORA_TARGET_MODULES),
        "completion_only_loss": True,
        "automatic_hub_push": False,
        "p2_gate": "open-real-device-evidence-required",
    }
    print(json.dumps(run_plan, indent=2, sort_keys=True))
    if not args.execute:
        return
    if not args.acknowledge_p2_gate:
        raise SystemExit(
            "Refusing training: --acknowledge-p2-gate is required for the mechanical "
            "smoke run; it does not authorize dataset-scale training."
        )
    sealed_commitment = None
    if dataset_metadata["dataset_version"] == WORKFLOW_DATASET_VERSION:
        sealed_commitment = validate_sealed_commitment(args.sealed_commitment)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('Install the ML dependencies with pip install -e ".[ml]"') from exc
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA execution requires CUDA; use a free GPU runtime for the smoke run")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("QLoRA execution requires a BF16-capable CUDA GPU")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, revision=BASE_MODEL_REVISION
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        device_map="auto",
        quantization_config=quantization,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
            exclude_modules=LORA_EXCLUDE_PATTERN,
        ),
    )
    model.config.use_cache = False
    train_records = [
        tokenize_completion_only(processor, example, max_length=args.max_length)
        for example in train_examples
    ]
    validation_records = [
        tokenize_completion_only(processor, example, max_length=args.max_length)
        for example in validation_examples
    ]
    train_dataset = Dataset.from_list(train_records)
    validation_dataset = (
        Dataset.from_list(validation_records) if validation_records else None
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps" if validation_dataset is not None else "no",
        eval_steps=args.eval_steps if validation_dataset is not None else None,
        load_best_model_at_end=validation_dataset is not None,
        metric_for_best_model=(
            "eval_semantic_exact_accuracy"
            if validation_dataset is not None
            else None
        ),
        greater_is_better=True if validation_dataset is not None else None,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="paged_adamw_8bit",
    )
    class InstrumentedTrainer(Trainer):
        def evaluate(self, *eval_args: Any, **eval_kwargs: Any) -> dict[str, float]:
            torch.cuda.empty_cache()
            metrics = super().evaluate(*eval_args, **eval_kwargs)
            if not validation_examples:
                return metrics
            from agentic_wallet.providers import LocalTransformersProvider

            provider = LocalTransformersProvider(
                model_id=BASE_MODEL_ID,
                revision=BASE_MODEL_REVISION,
                max_new_tokens=256,
                device="cuda",
            )
            provider._processor = processor
            provider._model = self.model
            was_training = self.model.training
            self.model.eval()
            semantic_examples = balanced_semantic_subset(
                validation_examples, args.semantic_eval_limit
            )
            previous_use_cache = self.model.config.use_cache
            self.model.config.use_cache = True
            torch.cuda.empty_cache()
            try:
                report = evaluate_development_examples(provider, semantic_examples)
            finally:
                self.model.config.use_cache = previous_use_cache
            semantic = report.to_dict(include_results=False)
            semantic_metrics = {
                f"eval_semantic_{key}": float(value)
                for key, value in semantic.items()
                if isinstance(value, (int, float))
            }
            metrics.update(semantic_metrics)
            self.log(semantic_metrics)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = args.output_dir / "development_metrics.jsonl"
            with metrics_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "global_step": self.state.global_step,
                            "loss_metrics": metrics,
                            "semantic": semantic,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if was_training:
                self.model.train()
            return metrics

    trainer = InstrumentedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=_collator(processor.tokenizer),
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    training_metadata = {
        **run_plan,
        "source_commit": _source_commit(),
        "source_tree_sha256": os.environ.get("AGENTIC_WALLET_SOURCE_TREE_SHA256"),
        "training_complete": True,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "checkpoint_selection_source": "development-validation-only",
        "sealed_suite_used_for_selection": False,
        "sealed_commitment": sealed_commitment,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
