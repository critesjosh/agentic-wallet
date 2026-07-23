# Gemma 4 E2B fine-tuning feasibility

Status: **the bounded GPU smoke test passed mechanically; dataset-scale training
and mobile conversion remain P2-gated**.

## Purpose

The first experiment asks whether narrow supervised fine-tuning can improve
schema adherence, exact tool arguments, safe dialogue routing, and signing
boundaries enough for local mobile use. It does not grant the model authority:
typed validation, workflow state, policy, approval digests, and the wallet
signer remain deterministic boundaries.

## Development regression boundary

Every case in `data/benchmark/train_family.jsonl` and
`data/benchmark/eval_family.jsonl` remains excluded from SFT text. The historical
`train` name means the familiar synthetic registry universe, not SFT
eligibility. Because v2 scenarios were chosen after inspecting failures on
these cases, the suite is now development regression data rather than an
independent quality evaluation. A new sealed suite must follow
`docs/sealed-eval-protocol.md` before another quality-training run.

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
5. Hash-commit an independently authored sealed suite and replace repetitive
   template morphology with naturally phrased canonical-argument examples.

After conversion, viability still requires 100% structured output on the sealed
suite and zero hard-zero critical failures. Training loss alone is not a success
metric.

## 2026-07-22 error-driven v2 result

The second bounded job used the same pinned base checkpoint, NF4/BF16 load,
rank-8 projection targets, completion-only objective, effective batch size 8,
greedy provider, L4 hardware, and 29 development-regression cases. It changed
both the dataset and training duration: 576 records and 150 optimizer steps
(about 2.1 dataset passes), versus v1's 144 records and 20 steps. Therefore the
improvement is not attributable to data alone.

| Run | Exact action + arguments | Schema-valid | Critical failures |
| --- | ---: | ---: | ---: |
| Untuned Transformers control | 0/29 | 0/29 (0.0%) | 0 |
| V1, 20 steps | 8/29 | 12/29 (41.4%) | 1 |
| V2 error-driven, 150 steps | 13/29 | 15/29 (51.7%) | 0 |

V2 familiar cases scored 10/19 exact and 12/19 schema-valid. Held-out assets
remained 3/10 exact and schema-valid. Five exact passes were no-argument safe
actions, so aggregate exact accuracy overstates multi-argument construction.
Persistent failures invented aliases such as `asset_id`, `amount`,
`max_slippage_percent`, `amount_dai`, and `scenario_id` instead of the canonical
swap, base-unit, quote-ID, and plan-ID fields. Three no-argument proposals copied
the constant training reason into `arguments`; v3 now trains an empty top-level
reason to remove that boilerplate without weakening `extra="forbid"`.

The completed job is `critesjosh/6a60e09a13e6ef894d54bfb1`; its private adapter
is under
`hf://buckets/critesjosh/agentic-wallet-smoke/e2b-qlora-smoke-20260722T152713Z`.
Non-weight evidence is in `data/training/results/hf-l4-v2-20260722/` and is bound
to source-tree SHA-256
`f86f8515edede270b27b5735ab222fac25e584d4c6f4c5b9bb08ea61b8589e48`.

## V3 corpus candidate and next blockers

`sft-v3-workflow.jsonl` is generated and validated but has not been trained. The
old 560-record templated candidate was replaced rather than trained. The current
curriculum expands 64 fixed natural-language source records into 48 training and
16 development-validation records across eight balanced families. It includes
four ordered two-turn recipient corrections and eight explanations grounded in
typed balance, allowance, quote, or simulation results. Its manifest reports a
coverage matrix over workflow state, action, ambiguity, risk, dialogue intent,
result type, user correction, and adversarial condition. Production examples
cannot expose benchmark-only unsafe actions; adversarial examples may expose
them only as non-target distractors. Repair-turn examples were deliberately
omitted because the runtime has no bounded retry path.

Training now evaluates and saves every 25 optimizer steps, records training and
validation loss plus schema/action/argument/safety/sequence metrics, keeps at
most three checkpoints, and selects the best checkpoint using development
semantic exact accuracy. The sealed suite is not available to this selection
path. Workflow-v3 `--execute` fails before CUDA unless a digest-only independent
human commitment is present.

Claude Sonnet 4.6 reviewed a sanitized methods/results packet through OpenRouter.
The review agreed that another GPU run should be blocked until:

1. An independently authored sealed suite is hash-committed before training.
2. Natural surface variation replaces repeated machine-coded “drill” morphology
   for the canonical argument failures.
3. A controlled ablation holds either dataset or optimizer steps constant.
4. The target constrained-decoding runtime is evaluated alongside greedy
   post-hoc validation.
5. The training source is committed and the physical-device P2 gate is closed.

The highest-value next training comparison is a small, naturally phrased
canonical-argument curriculum versus the current v2 data at the same step
count—not a larger repetition of the same templates. Preference tuning remains
premature.

## Native constrained-decoding development baseline

The local Ollama artifacts were evaluated through the native JSON-schema API;
deterministic validation then measured semantic correctness separately.

| Artifact | JSON syntax | Full typed schema | Exact | Multi-argument exact | Critical |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemma4:e2b` (3.3 GB) | 7/29 | 7/29 | 5/29 | 0/7 | 2 |
| `gemma4:e4b` (9.6 GB) | 27/29 | 11/29 | 9/29 | 0/7 | 1 |

E2B failed both signing-boundary cases; E4B failed one. E4B's native constraint
substantially improved JSON syntax, but action-specific canonical arguments
remained the bottleneck. Each local run completed in roughly two to three
minutes and incurred no paid-infrastructure cost.

The fixed 16-record v3 development-validation split adds one genuine two-turn
recipient-correction trajectory. Under the same local runtimes, E2B produced
6/16 schema-valid and 5/16 exact outputs; E4B produced 7/16 schema-valid and
6/16 exact outputs. Both scored 0/7 on multi-argument records and 0/1 on the
complete trajectory. Neither produced a hard-zero failure on this small split;
invalid outputs failed closed. This is checkpoint-development evidence only,
not sealed evaluation or a generalization result.

The v2 PEFT adapter could not be compared honestly under Ollama. Ollama found
the safetensors but failed conversion/config resolution, and its E2B base is the
separate mobile/QAT artifact rather than the pinned Transformers base used for
v2. The 13/29 v2 greedy result is retained only as a non-comparable development
reference. See `data/training/results/constrained-runtime-comparison-20260722.json`.

No L4 job was launched for v3 because the independent-human sealed-suite gate
was not satisfied. Authorized paid cost for this attempted experiment: **$0**.

## V4 runtime-pipeline curriculum

The implemented conversation pipeline changed the training contract rather
than merely adding more paraphrases. `sft-v4-pipeline.jsonl` expands the same 64
fixed natural source records into 240 stage-specific examples with a fixed
180/60 development split:

- 64 argument-free route calls and 64 route-repair calls;
- 56 selected-action argument calls and 56 argument-repair calls;
- a typed bounded `conversation_ledger` in every context;
- grounded narration inputs containing the actual typed result and a
  deterministic fallback summary.

The route never carries arguments. Once deterministic validation accepts its
action ID, the argument stage exposes exactly that one action and its canonical
schema. Each validation stage permits at most one non-executing correction; a
signing-boundary action is never eligible for repair. The dataset mirrors these
runtime phases (`route_dialogue`, `repair_dialogue_route`,
`fill_tool_arguments`, `repair_tool_arguments`, and
`explain_verified_tool_result`) instead of teaching a combined wallet-shaped
JSON response that production no longer requests.

Dataset SHA-256:
`94231ff7183c6f190ac5bbef63628747b9590dd35bd9e1142958943560164c26`.
The 29-case benchmark remains development regression data. The new remote run
may select checkpoints using only the committed 60-record v4 validation split
and may report the 29-case regression suite, but it cannot make an independent
generalization or release-safety claim without the still-sealed suite.

The first v4 L4 attempt (`critesjosh/6a60fcf413e6ef894d54c152`) reached step 25
but OOMed before producing an evaluation result because Trainer inherited its
default evaluation batch size and attempted a 3.6 GiB logits allocation while
the quantized training model and optimizer were resident. This is an evaluation
configuration failure, not a model result. Evaluation now uses batch size one,
clears unused CUDA cache first, and enables expandable allocator segments. The
partial checkpoint is not used for comparison.

The memory-fixed retry (`critesjosh/6a60fec5d09dc1f57c6c2d92`) confirmed that
batch-one loss evaluation fits, but full 60-record generation at every
checkpoint ran with inference caching disabled and could not finish the required
protocol within the 45-minute bound. It was cancelled without using a partial
checkpoint. Checkpoint selection now uses a deterministic 20-record round-robin
across all five runtime phases with inference caching enabled temporarily; the
post-training evaluation still covers all 60 development records.

## V4 pipeline result

The completed research run is `critesjosh/6a6102fb13e6ef894d54c18f`
(17 minutes of running time) from source commit `38105da`. It trained the pinned
E2B base for 50 optimizer steps on an L4. Checkpoint 50 was selected over step 25
using the balanced 20-record development subset: exact accuracy rose from 30%
to 60% and schema validity from 35% to 65%, though both checkpoints retained one
safety failure.

On the complete 60-record v4 development-validation split, checkpoint 50 scored:

| Metric | Result |
| --- | ---: |
| Schema-valid | 45/60 (75.0%) |
| Exact route/action/arguments | 44/60 (73.3%) |
| Zero-argument exact | 30/32 |
| Single-argument exact | 13/14 |
| Multi-argument exact | 1/14 |
| Complete trajectory accuracy | 0 |
| Hard-zero safety failures | 1 |

The critical failure was a missing-recipient turn routed to
`create_transfer_plan` rather than `request_missing_information`. No action
executed, but the observed wrong-recipient-category failure blocks release. The
aggregate improvement is mostly route and simple-field learning; canonical
multi-argument construction remains essentially unsolved.

The ambiguity family contains two natural validation turns expanded into eight
stage records. The safety failure occurred on one of the two normal route turns
(and one of eight expanded wrong-recipient records). The four trajectory-tagged
records form separate route and argument trajectories for those two turns: the
first route failed and the second argument call failed, so neither complete
trajectory passed and sequence accuracy is correctly zero.

Small samples make the point estimates unstable. Wilson 95% intervals are
61.0%-82.9% for 44/60 exact, 62.8%-84.2% for 45/60 schema-valid, and
1.3%-31.5% for 1/14 multi-argument exact. The 20 checkpoint-selection records
are a phase-balanced subset of these same 60 development records, not an
independent set. The full score is therefore development evidence influenced by
checkpoint choice and may be optimistic.

The completed training job also ran the historical 29-case evaluator and scored
0/29 with 100% JSON syntax but 0% typed validity. That evaluator still used the
retired one-call tool contract, while v4 intentionally trains an argument-free
route followed by a one-action argument call. The result is retained as a
protocol-mismatch diagnostic, not compared to v1/v2 quality.

That fixed-weight evaluation-only rerun is complete under
`staged-dialogue-route-v2.1`. The first staged execution exposed and preserved
two scorer regressions: rejected arguments were incorrectly counted as executed
critical failures, and the bounded-repair wrapper hid the final validation error
from syntax classification. V2.1 changed only those scoring/error-reporting
semantics and added regression tests; cases, prompts, repairs, weights, and
expected outputs stayed fixed.

| 29-case protocol | Exact | JSON syntax | Fully typed | Multi-arg exact | Critical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retired single-stage (non-comparable) | 0/29 | 29/29 | 0/29 | 0/7 | 0 |
| Staged v2.1 production contract | 3/29 | 28/29 | 4/29 | 0/7 | 11 |

The staged score is not model improvement over the retired score; it is the same
adapter measured under the contract it was trained to use. It passes only three
familiar-family cases and zero held-out-family cases. Eleven genuine wrong or
forbidden route selections remain after removing fail-closed argument rejections
from critical scoring. This reinforces the full v4 split's conclusion: routing
can look strong on the narrow development curriculum while collapsing on the
older regression distribution, and canonical multi-argument construction
remains unsolved.

The first successful evaluation-only job was
`critesjosh/6a610e9a13e6ef894d54c249` (585 seconds). An independent repeat,
`critesjosh/6a610fae13e6ef894d54c261` (582 seconds), reproduced the same report
byte-for-byte. Both used source commit `eddb2df`; no retraining occurred. Jobs
`critesjosh/6a610cfb13e6ef894d54c229` and
`critesjosh/6a610e6e13e6ef894d54c247` failed before evaluator startup due to
Hugging Face volume-mount errors. A fresh immutable source mount resolved the
infrastructure issue.

All evidence is development-only. No sealed suite was opened. Non-weight run
artifacts are under `data/training/results/hf-l4-v4-pipeline-20260722/`; weights
remain private at
`hf://buckets/critesjosh/agentic-wallet-smoke/e2b-qlora-smoke-20260722T175116Z/adapter`.

## V5 candidate-binding curriculum

The next dataset revision follows the post-v4 deterministic hardening rather
than asking the model to memorize literal recipient construction. The
model-facing action is `create_transfer_plan_from_candidate`; its recipient is
an opaque trusted ID. Deterministic code extracts a current-user address,
validates checksum rules, resolves the canonical asset and exact base-unit
amount, and either constructs the typed call without an argument-model request
or forces clarification for missing or ambiguous fields.

`candidate-pipeline-v5` is generated from the same fixed natural source while
leaving v4 and its result artifacts unchanged. It has 232 records: 128 dialogue
routes and 104 remaining argument calls, split 174/58. The eight old transfer
argument-generation and repair records are intentionally removed. Of the route
records, 112 now train the production one-field `proposed_action` decision;
the 16 grounded narration records retain their factual display-only envelope.
Dataset SHA-256:
`642e194c7d4af8b70c4385323c30448ff1f2599c80f6d714ec16eeb5f4053baf`.

The v5 L4 run completed as Hugging Face job
`critesjosh/6a6127e813e6ef894d54c3e1` from source commit `1b8de62`. It trained
75 optimizer steps and retained checkpoints 25, 50, and 75. The adapter weights
remain private in the job bucket; non-weight evidence is under
`data/training/results/hf-l4-v5-candidate-binding-20260722/`.

The matching Transformers comparisons are development-only:

| Checkpoint | V5 eligible exact | V5 safety | Independent exact | Independent safety |
| --- | ---: | ---: | ---: | ---: |
| Untuned base | 24/54 | 0 | 23/40 | 3 |
| Step 25 | 37/54 | 0 | 33/40 | 0 |
| Step 50 | 46/54 | 0 | 31/40 | 1 |
| Step 75 | 50/54 | 1 | 29/40 | 2 |

Step 75 was originally selected by exact accuracy alone. That is now rejected:
it introduced an unlimited-approval development failure, while later training
also monotonically degraded the independent result and refusal safety.
Checkpoint selection is now safety-lexicographic. Step 25 is the only checkpoint
with zero hard-zero failures on both development suites and is therefore the v5
candidate.

Against the base, checkpoint 25 has 14 adapter-only wins, four base-only wins,
19 cases both passed, and three neither passed. The paired two-sided exact
McNemar p-value is 0.031. This is useful evidence that tuning changed routing,
but it is not confirmatory: the independently authored suite is development-only
and was used to select checkpoint 25. Both base and every adapter checkpoint
were 40/40 schema-valid on that suite, so the one-field route contract, not
tuning, deserves credit for formatting reliability.

The v5 validation split is also consumed development evidence. The
grounded-display exclusion was introduced after inspecting the first result,
based on an architectural task-contract mismatch, and is therefore
outcome-informed rather than pre-registered. Original 58-record reports remain
committed. Future evaluators must freeze task eligibility before any model run.

Hard-zero failures are raw model-route errors in predeclared critical
categories. An incomplete candidate-transfer route is still an exact miss, but
not a hard-zero: the one-field route cannot encode transaction fields or reach
planning until deterministic required-fact binding succeeds. The same boundary
is used for every checkpoint and the base.

Four of the 58 v5 validation records retain the legacy factual display
envelope. They are training material for grounded narration, not eligible
minimal-route cases, and are now reported separately rather than counted as
route failures. A dedicated narration-grounding evaluator is still required.
Two fixed-weight step-75 repeats reproduced byte-for-byte: 50/54 exact with one
development safety failure and 29/40 exact with two independent safety failures.
The earlier 49/54 training-time sweep result is retained as an unexplained
single-case variance, but it does not affect selection because step 75 fails the
safety gate.

The evaluation-only job was `critesjosh/6a61f6d413e6ef894d54da0e` from source
commit `e9c087b`; it performed no training. Its non-weight reports are under the
v5 result directory. Failed predecessor jobs stopped during wrapper dependency
setup before loading a model and produced no model result.

The selected checkpoint was then repeated in a separate job,
`critesjosh/6a61fcf7d09dc1f57c6c503c`, from source commit `7f0f0e8`.
The complete checkpoint-25 development JSON reproduced at SHA-256
`2a5c3c71463c78873704d0303c205faef0dd468ba2077c7a8341967a2762bbcc`;
the independent-development JSON reproduced at
`27442b042173f71490822fd278dd6cc33dc61466058b5d8a3c11788e00bd18ae`.

The untuned local Ollama E2B development pilot of the matching minimal route
contract scored 7/12 raw routes and 12/12 guarded end-to-end results, with all
six hazardous cases contained. The immediate code gate is that no
literal recipient can validate under the new action, absent or multiple
recipient candidates cannot reach planning, and an unknown candidate ID fails
closed during deterministic binding. Future evaluation must measure routing
and clarification separately from fields the model no longer owns.

## V6 transaction-boundary curriculum and untuned development result

The Phase 8 proof adds a deliberately narrow live path: an exact current-message
request may route to a review for native ETH on Base, while deterministic code
owns every transaction field, state read, simulation check, policy decision,
approval digest, signer capability, and submission result. A separate
session-scoped route may look up a transaction hash already saved by the
application. Neither route gives the model approval, signing, submission, raw
transaction, key, capability-token, or RPC access.

`transaction-boundary-v6` retains the 232 v5 records and adds 36 records for:

- Base-only native-transfer review routing and wrong-chain contrasts;
- application-owned digest approval and signer-action distractors;
- nonce, registry, and account-state drift followed by re-simulation;
- `SUBMITTED`, `SUBMISSION_UNKNOWN`, saved hashes, trusted explorer links, and
  the no-automatic-retry rule;
- lost signer responses with and without a recoverable journaled hash;
- EOA/contract/zero/self recipient preflight and pinned-block provenance;
- exact-current-message status lookup versus fake or untrusted hashes; and
- grounded narration of typed transaction outcomes.

The generated artifact has 268 records, split 197 training and 71
development-validation, with 104 tool calls and 163 dialogue routes. Its
dataset SHA-256 is
`a567f33a094443909ea686a5f60f6c07d74d139b931f5faf12ab6534b9879ccc`;
the manifest records curriculum version
`wallet-transaction-boundary-curriculum-v6-3`. Coverage validation rejects
sensitive signing material and prevents the live curriculum from exposing
signing or submission as production model actions.

No v6 adapter has been trained. Before spending GPU time, the untuned
`gemma4:e2b` Ollama runtime was evaluated against the V6.2 validation partition.
Nine grounded-display records were predeclared ineligible for the minimal-route
evaluator, leaving 61 scored records:

| Metric | Result |
| --- | ---: |
| Schema-valid | 45/61 (73.8%) |
| Correct action | 39/61 (63.9%) |
| Exact action and arguments | 34/61 (55.7%) |
| Complete trajectories | 2/3 (66.7%) |
| Multi-argument exact | 0/12 |
| Development safety failures | 7 |

The seven safety failures include a wrong-chain request routed to Base transfer
review and an adversarial repair that repeated `proceed_to_signing`. The other
five selected the expected action but supplied a wrong missing field or a
quote/plan identifier contaminated with generated control text. Sixteen total
records failed schema validity: all 12 multi-argument items returned non-JSON,
and four digest-confirmation items ended before Ollama reported completion.

Every observed failure remained proposal-only and failed closed. A
wrong-chain route had no deterministically parsed Base candidate; the production
allowlist does not contain `proceed_to_signing`; unrecognized opaque IDs do not
resolve; and malformed arguments cannot reach planning or the signer. This is
evidence that the deterministic split is doing necessary work, not that the
model is safe. A future v6 fine-tune should use safety-lexicographic checkpoint
selection and must reach zero hard-zero failures before ordinary accuracy is
considered. This validation partition is development data and cannot support a
generalization or sealed-evaluation claim.

The seven eligible Phase 8 route cases were 7/7 schema-valid and 3/7 exact; two
additional misses chose safe read-only fallbacks and are semantic errors rather
than safety failures. The canonical report is
`data/training/results/ollama-e2b-v6-transaction-20260723-rescored.json`; see
`docs/model-failures.md` for reproducible examples and proposed curriculum
targets. V6.3 adds one grounded narration for terminal unknown status without a
recoverable hash. That record is predeclared ineligible for this minimal-route
evaluator, and none of the 61 scored inputs changed. A requested repeat was
blocked by the local execution-credit limit, so the committed report remains
honestly labeled as the V6.2 model run rather than a fabricated V6.3 rerun.
