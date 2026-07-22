# Agentic Wallet

**This is an experimental research prototype, not a production wallet.** It
explores whether small, private language models can run locally on mobile
devices and reliably help with narrow, high-stakes workflows when deterministic
software—not the model—retains control.

The first use case is an onchain wallet assistant built around Gemma 4 edge
models. The model proposes typed actions; deterministic code resolves live
facts, performs exact arithmetic, enforces policy, and eventually hands an exact
approved transaction to a separate wallet signer. The model never receives a
seed phrase or private key, and model output is never authorization.

The broader experiment is also about **fine-tuning edge models for specific use
cases**: measure an untuned mobile-sized model, build an error-driven benchmark,
fine-tune with QLoRA, convert the result to a mobile runtime, and compare safety,
quality, memory, latency, and thermals against the untuned baseline. The wallet
domain is the test bed; the model/harness split should generalize to other
private, specialized assistants.

See [plan.md](plan.md), especially the Consensus revisions block, for the full
research and safety design.

## Current status

The browser demo is fixture-backed and read-only. It demonstrates chat-to-tool
routing, strict schemas, fail-closed validation, and deterministic portfolio
reads, but it does **not** read a live chain or load a real account yet. The
unsigned transfer/swap, simulation, policy, and approval-integrity workflow is
implemented and tested in the Python core but is not yet connected end-to-end
to chat. There is no transaction submission or key custody.

The repository contains typed schemas, the workflow state machine, a read-only
harness, swappable inference providers, the behavioral benchmark, an Android
runtime spike, and a deterministic unsigned planning/simulation flow. There is
no signing, submission, or key custody.

## Layout

```text
src/agentic_wallet/
  schemas/        pydantic models (intent, tool-call, portfolio, plan,
                  simulation-result, policy, approval-envelope)
  state_machine.py  workflow states and validated transitions
  digest.py       canonical serialization + sha256 for the C1 approval digest
  registry.py     canonical id -> address (primary root of trust, plan.md P4)
  inference.py    InferenceProvider seam (local/remote now, on-device later)
  tool_contract.py shared typed action registry, prompts, and decoding schemas
  candidate_binding.py  trusted recipient IDs + required-fact clarification
  planning.py     deterministic unsigned transfer and pinned-route swap plans
  simulation.py   normalized before/after diff verification
  policy_engine.py deterministic plan and simulation policy enforcement
  approval_guard.py C1 approval freshness and mutation invalidation
  harness/        fixture-backed, network-free, read-only tools
  benchmark/      29 familiar/held-out eval cases with arguments and P6 blockers
  training/       deterministic SFT generation, leakage checks, prompt masking
  web/            FastAPI read-only demo + static page
fixtures/         sample watch-only portfolio
data/benchmark/   train and held-out eval families (distinct asset universes)
data/training/    generated SFT v1 data, manifest, and non-weight run results
schemas/          generated JSON Schema (from scripts/export_schemas.py)
tests/            state machine, digest, schemas, harness, benchmark, web
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,web]"
pytest                                  # run the suite
python scripts/export_schemas.py        # regenerate /schemas
uvicorn agentic_wallet.web.app:app --reload   # http://127.0.0.1:8000
```

The chat uses the deterministic keyword responder by default. A remote
llama.cpp deployment serving the target Gemma family can be selected without
shipping model weights in the app:

```bash
AGENTIC_WALLET_INFERENCE_PROVIDER=llama-cpp-http \
AGENTIC_WALLET_INFERENCE_BASE_URL=https://model.example \
  uvicorn agentic_wallet.web.app:app --reload
```

The UI discloses remote mode and requires consent per message before wallet
intent is sent. Ollama and llama.cpp are classified from their configured URL:
loopback is local, while any non-loopback host is remote and activates the same
consent gate. For local Transformers development instead, install the ML
extra and select it explicitly:

```bash
pip install -e ".[dev,web,ml]"
AGENTIC_WALLET_INFERENCE_PROVIDER=local-transformers \
  uvicorn agentic_wallet.web.app:app --reload
```

The model can only propose enumerated tools. The conversation pipeline now uses
three separated model stages: an argument-free dialogue route, an action-specific
argument envelope when a tool is selected, and a grounded explanation only
after deterministic execution. This keeps the route schema small and prevents
the model from inventing a result before the tool runs. `message` is always
display-only; only the typed proposal can enter validation.

Every stage is constrained where the runtime supports it and validated again in
deterministic code. A malformed route or argument envelope gets at most one
non-executing repair attempt; signing-boundary actions are never repairable.
Unknown assets, extra arguments, unavailable actions, repeated invalid output,
and invented narration facts fail closed. Conversation state is a bounded typed
ledger of resolved intent, corrections, verified facts, prior proposals, and
recent messages. It deliberately contains no approval field: history is context,
never authorization. Observed failures are recorded in
[`docs/model-failures.md`](docs/model-failures.md).

Candidate-bound transfers now remove recipient-address generation from the
v3 candidate-binding model contract. Deterministic code extracts a
checksum-valid address only from
the current user message (or a separately verified contact), assigns an opaque
`recipient:*` ID, and constructs the transfer fields without an argument-model
call. Missing or multiple recipients force clarification before planning.
Transaction-history addresses are never promoted to candidates automatically.
The resolved address still enters the existing unsigned planning, simulation,
policy, and digest-bound approval flow; this adds no signing authority and is
not yet exposed by the read-only web demo.

Ollama, llama.cpp, and OpenRouter request native schema-constrained decoding.
The direct Transformers provider currently has post-hoc whole-output validation
only; it is explicitly non-conformant with the native constrained-decoding
preference and must independently pass the measured structured-output gate.

`scripts/run_web_poc.sh` also enables an in-memory, loopback-only development
transcript at `http://127.0.0.1:8000/debug/transcripts`. It is bounded, sends
no-store headers, requires both a loopback client and loopback Host header,
never writes messages to disk, and clears on server restart.
Set `AGENTIC_WALLET_DEBUG_TRANSCRIPTS=0` to disable it.

The EIP-1559 signing-request types and optional `signer` dependencies are Phase
8 preparation authorized for the upcoming Ethereum MCP signer. They contain no
signing implementation or key material; the running web app still advertises
`signing: false` and rejects signing requests before inference.

The Android GGUF is an optional, separately downloaded model pack, never part
of the APK/AAB. See [docs/android-spike.md](docs/android-spike.md) for the
reproducible emulator measurements and the still-open real-device gate.
Use `scripts/android/run_benchmark.sh --resume`; it shuts down the dedicated
emulator automatically when the run exits.

## Fine-tuning feasibility

Three bounded E2B QLoRA experiments have run successfully on Hugging Face L4
hardware using completion-only masking and a pinned `google/gemma-4-E2B-it`
revision. V1 used 144 records and 20 steps; error-driven v2 used 576 records and
150 steps; staged-pipeline v4 used 240 records and 50 steps.

On a controlled same-checkpoint/same-runtime comparison, the untuned model
produced 0/29 schema-valid calls. V1 produced 12/29 schema-valid and 8/29 exact
calls with one critical failure. V2 produced 15/29 schema-valid and 13/29 exact
calls with no critical failure. V2 changed both data and step count, and its
scenarios were selected from failures on the same 29 cases, so those cases are
now development regression data—not an independent evaluation. This proves the
training path can change target behavior; it does **not** establish safety,
generalization, or release readiness. The committed
non-weight results are under `data/training/results/`; adapter weights remain in
a private Hugging Face bucket and are excluded from Git.

V4 reached 44/60 exact and 45/60 schema-valid on its development-validation
split, but only 1/14 multi-argument calls were exact, complete-trajectory
accuracy was zero, and one missing-recipient route tripped the wrong-recipient
hard-zero gate. It is therefore not release-ready. Its initial 0/29 historical
regression score used the retired single-stage evaluator and is recorded only as
a protocol-mismatch diagnostic; the benchmark runner now follows the staged
route-then-arguments contract.

With the same adapter and the corrected staged evaluator, v4 scored only 3/29
exact and 4/29 fully typed-valid, with 0/7 multi-argument cases and 11 genuine
critical route selections. The earlier 0/29 remains separately recorded under
the retired, non-comparable evaluator. Neither result is sealed evaluation, and
v4 is not a capability milestone or release candidate.

Generate and validate the data and inspect the run configuration without a GPU:

```bash
python scripts/generate_training_data.py
python scripts/generate_training_data.py --profile pipeline-v4
python scripts/generate_training_data.py --profile candidate-pipeline-v5
python scripts/train_qlora.py --dataset data/training/sft-v4-pipeline.jsonl
```

The v4 pipeline curriculum deterministically expands the fixed, naturally
phrased 64-record v3 source into 240 exact-runtime examples: 128 argument-free
routes (including route repairs) and 112 selected-action argument calls
(including argument repairs), split 180/60 for development. It includes typed
conversation ledgers and grounded typed results. This is development data, not
an independent safety evaluation. The sealed suite remains unavailable and must
not be opened or used for checkpoint selection.

The generated v5 candidate-binding curriculum preserves v4 as historical
evidence while adapting future routing data to the safer runtime. It contains
232 records (174 train, 58 development-validation). Candidate transfers appear
only as route decisions; the eight obsolete free-generated transfer-argument
and repair records are absent because deterministic code now owns those fields.
V5 has not been trained or evaluated yet.

The training path evaluates and checkpoints every 25 optimizer steps. The
training command requires explicit
`--execute --acknowledge-p2-gate`, CUDA, and BF16 support; it never pushes to
the Hub automatically. `scripts/run_hf_qlora_smoke.py` is the bounded remote
entry point used for the recorded run. See
[docs/fine-tuning.md](docs/fine-tuning.md).

The current native constrained-decoding development baselines are 5/29 exact
for local Ollama E2B and 9/29 for E4B. Both scored 0/7 on multi-argument cases
and retained signing-boundary failures. On the separate 16-record v3 validation
split, E2B scored 5 exact and E4B scored 6; both failed all seven multi-argument
records and the one complete two-turn trajectory. These are development
diagnostics, not generalization evidence. No v3 L4 job has run; the attempted
experiment cost $0 because the sealed-evaluation gate stopped it before
submission.

## Notebook

`gemma-4-E4B.ipynb` is a thin GPU driver (Colab). It imports this package for
schemas and benchmark data and keeps only model load, LoRA training, and
inference in cells. `nbstripout` is installed as a git filter, so notebook
outputs are stripped on commit but kept in the working copy.
