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

The model can only propose the enumerated read tools. Tool-specific arguments
are validated again before the deterministic harness executes them. Invalid
output, unknown assets, extra arguments, and unavailable actions fail closed.
Each model response is a structured dialogue turn with strictly separated
fields: `message` is display-only, `proposed_action` is the only field that may
enter the typed tool pipeline, and `suggested_actions` contains only canonical
IDs whose labels/prompts are owned by the server. A bounded conversation history
is context, never approval. When a read tool runs, the model receives its typed
result in a second call before writing the conversational explanation.

For target-model compatibility, the native dialogue decoding schema uses a flat
argument envelope rather than a nested per-action union. That schema constrains
the wire structure and known fields; deterministic code then validates the exact
required arguments for the selected action before any tool runs. This is a
deliberate fail-closed compromise documented with observed model failures in
[`docs/model-failures.md`](docs/model-failures.md).

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

Two bounded E2B QLoRA experiments have run successfully on Hugging Face L4
hardware using completion-only masking and a pinned `google/gemma-4-E2B-it`
revision. V1 used 144 records and 20 steps; error-driven v2 used 576 records and
150 steps.

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

Generate and validate the data and inspect the run configuration without a GPU:

```bash
python scripts/generate_training_data.py
python scripts/generate_training_data.py --profile workflow-v3
python scripts/train_qlora.py
```

Another GPU run is blocked on the independently authored suite in
`docs/sealed-eval-protocol.md`, a controlled data-versus-step ablation, and the
physical-device P2 measurements in `docs/android-spike.md`. The repetitive
560-record v3 draft has been replaced by a fixed, naturally phrased 64-record
curriculum with a generator-independent 48/16 development split. The training
path now evaluates and checkpoints every 25 optimizer steps, but fails closed
before CUDA until a digest-only independent-human sealed-suite commitment is
present. The training command requires explicit
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
