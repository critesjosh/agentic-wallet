# Gemma 4 E2B fine-tuning feasibility

Status: **the bounded GPU smoke test passed mechanically; dataset-scale training
and mobile conversion remain P2-gated**.

## Purpose

The first experiment asks whether narrow supervised fine-tuning can improve
schema adherence, exact tool arguments, safe dialogue routing, and signing
boundaries enough for local mobile use. It does not grant the model authority:
typed validation, workflow state, policy, approval digests, and the wallet
signer remain deterministic boundaries.

## Frozen evaluation boundary

Every case in `data/benchmark/train_family.jsonl` and
`data/benchmark/eval_family.jsonl` is evaluation-only. The historical `train`
name means the familiar synthetic registry universe, not SFT eligibility. The
29-case benchmark now includes at least two cases per declared hard-zero
category, including `wrong-asset`.

Generated training examples use only `TRAIN_REGISTRY`. Validation rejects:

- any held-out DAI, cbETH, or Aerodrome ID, symbol, or address;
- exact or near-duplicate frozen benchmark prompts;
- duplicate records or IDs;
- unavailable actions, invalid typed arguments, and unknown canonical IDs;
- excessive label imbalance.

`data/training/sft-v1.manifest.json` binds the generated dataset digest to both
frozen benchmark file digests, the generator version, seed, base model ID, and
base model revision.

## Current checkpoint decision

The mechanical smoke path is pinned to:

- model: `google/gemma-4-E2B-it`;
- revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`;
- NF4 4-bit load, BF16 compute, double quantization;
- LoRA rank 8 over text `q_proj`, `k_proj`, `v_proj`, and `o_proj`, with the
  unsupported Gemma 4 vision/audio projection wrappers explicitly excluded;
- completion-only loss with every prompt token masked to `-100`.

This deliberately selects the instruction-tuned checkpoint for the plumbing
test. The final mobile lineage is unresolved: the Android GGUF spike used a QAT
Q4 artifact, while LiteRT-LM is a separate candidate. Do not merge or quantize
an adapter until the real-device P2 gate and runtime choice are resolved.

## Commands

CPU-safe generation, validation, and dry run:

```bash
python scripts/generate_training_data.py
python scripts/train_qlora.py
```

The dry run imports no heavyweight model and performs no training. A mechanical
free-GPU smoke run, after explicitly acknowledging that P2 is still open, is:

```bash
python scripts/train_qlora.py \
  --execute --acknowledge-p2-gate \
  --max-steps 20 \
  --output-dir artifacts/e2b-qlora-smoke
```

This machine has no usable CUDA GPU, so that command fails closed locally.
`scripts/run_hf_qlora_smoke.py` packages the same bounded run for Hugging Face
Jobs. It never uploads `.env`, local weights, or Git metadata, and it writes
adapter artifacts to a private bucket rather than pushing a public model.

Evaluate a future adapter through the unchanged benchmark contract:

```bash
python scripts/evaluate_adapter.py \
  --adapter-path artifacts/e2b-qlora-smoke
```

Pass `--json-output result.json` to retain per-case evidence. Omitting
`--adapter-path` evaluates the pinned untuned base through the identical
Transformers provider path.

## 2026-07-22 Hugging Face L4 smoke result

The bounded run used one NVIDIA L4 (24 GB), Python 3.12.12, PyTorch
`2.13.0+cu130`, BF16 compute, NF4 4-bit base weights, rank 8, batch size 1 with
eight-step gradient accumulation, and 20 optimizer steps. The pinned base model
download was 10.2 GB. Training took 93.04 seconds (0.215 optimizer steps/second),
with observed per-step loss moving from 5.347 on the first step to 1.067 on the
last and an average training loss of 2.061. The saved adapter weight file is
10.7 MB.

The full load/train/save/reload/evaluate job ran for 437 seconds. Across all
diagnostic, successful, detailed-evaluation, and controlled-base jobs, billed
GPU running time was about 22.5 minutes, or approximately $0.30 at the listed
L4 rate of $0.0133/minute. Queue time is excluded from that estimate.

Controlled frozen-suite results using the same pinned checkpoint, provider,
4-bit load, L4, prompt, and greedy decoding path:

| Run | Exact action + arguments | Schema-valid | Critical failures |
| --- | ---: | ---: | ---: |
| Untuned Transformers control | 0/29 | 0/29 (0.0%) | 0 |
| 20-step QLoRA smoke adapter | 8/29 | 12/29 (41.4%) | 1 |

The adapter's familiar family scored 5/19 exact and 9/19 schema-valid, with one
critical failure. The held-out family scored 3/10 exact and 3/10 schema-valid,
with no critical failure. Case `t10` still selected `proceed_to_signing` instead
of requesting digest-bound user confirmation. Seventeen responses failed
closed on malformed JSON or typed-schema/argument errors. Common errors included
extra `args` or reason fields, missing `missing_fields`/`plan_id`, non-canonical
transfer argument names, and prose surrounding JSON.

This is evidence that narrow SFT can materially improve the exact target
behavior from a very weak direct-Transformers baseline. It is not a quality
result: the dataset has only 144 synthetic examples, the adapter still violates
a hard-zero boundary, structured validity is far below 100%, and no mobile
conversion or physical-device measurement has occurred. The separate Ollama
QAT baseline (5/29 exact, 7/29 schema-valid, two critical failures) is useful
mobile-like context but is not a controlled comparison to this adapter.

The non-weight evidence is committed in:

- `data/training/results/hf-l4-base-20260722-evaluation.json`;
- `data/training/results/hf-l4-smoke-20260722-evaluation.json`;
- `data/training/results/hf-l4-smoke-20260722-environment.json`;
- `data/training/results/hf-l4-smoke-20260722-training-metadata.json`;
- `data/training/results/hf-l4-smoke-20260722-run-summary.json`.

The sanitized training snapshot intentionally omitted `.git`, so the historical
training metadata has `source_commit: null`. The dataset, both benchmark hashes,
model revision, and run configuration remain pinned, but the exact historical
source-tree hash was not captured. The HF runner now records a deterministic
digest of training-relevant source, scripts, dataset, benchmark, and dependency
configuration for future jobs, independently of Git metadata.

The disposable adapter and full logs remain private at
`hf://buckets/critesjosh/agentic-wallet-smoke/e2b-qlora-smoke-20260722T134205Z`.
The successful job ID is `critesjosh/6a60c89513e6ef894d54bc05`; the detailed
adapter evaluation is `critesjosh/6a60caa5d09dc1f57c6c2664`, and the controlled
base evaluation is `critesjosh/6a60ccc3d09dc1f57c6c26c2`.

Three earlier bounded jobs failed before any optimizer step and exposed the
Gemma 4/PEFT targeting detail now covered by tests: multimodal projections use
an unsupported `Gemma4ClippableLinear` wrapper, while the language projections
are supported ordinary linear layers. The final configuration targets the
language projection suffixes and excludes `vision_tower` and `audio_tower`.

The smoke dataset contains no `request_user_confirmation`,
`proceed_to_signing`, or `reject_simulation` targets. That omission is
acceptable for proving mechanics but helps explain the retained `t10` failure;
signing-boundary contrastive examples are mandatory before a larger run.
Scenario classes also overlap structurally between the deterministic SFT
generator and frozen suite even though prompts and asset/protocol universes are
isolated. Before making a quality claim, add independently authored scenario
classes and retain a generator-independent held-out evaluation set.

## Gates before a real training run

1. Close physical-device P2: cold load, peak memory, sustained generation,
   thermals, battery, crash rate, and fail-closed structured validity.
2. Decide GGUF/llama.cpp versus LiteRT-LM as the mobile runtime.
3. Use an environment with enough memory for merging and conversion; the local
   BF16-to-GGUF attempt already OOM-killed near 16.1 GB RSS.
4. Obtain explicit approval before any additional paid GPU work beyond bounded
   evaluation or smoke testing.
5. Expand the dataset beyond the 144-example plumbing set before claiming a
   quality experiment.

After conversion, viability still requires 100% structured output on the
frozen suite and zero hard-zero critical failures. Training loss alone is not a
success metric.
