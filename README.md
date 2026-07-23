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

The browser defaults to a fixture-backed, read-only demo. It demonstrates
chat-to-tool routing, strict schemas, fail-closed validation, and deterministic
portfolio reads.

An explicitly enabled, **loopback-only Phase 8 proof of concept** now adds one
live workflow: a native ETH transfer on Base to an externally owned account.
Chat or Gemma may request a review, but deterministic code reads live RPC state,
constructs the exact EIP-1559 preimage, performs a pinned preflight, applies
policy, and renders the digest-bound review. Approval is a separate browser
action. A private stdio MCP process loads the key only from an approved OS
keyring, rechecks freshness, signs, and submits. The model receives no approval,
capability, raw transaction, RPC credential, or key.

This remains a research POC, not a production wallet. Live signing is disabled
unless the feature flag, Base RPC, high-entropy capability key, secure keyring,
and provisioned signer all pass readiness. The current development shell has no
approved keyring backend, so its real-funds path correctly remains disabled;
tests use fake RPC and signer implementations.

The loopback web process is inside this POC's trusted computing base because it
mints the signer capability after the browser approval request. The isolated
signer still revalidates the exact envelope and live state and is the only
process that can access the key, but it cannot independently prove that a human
clicked Approve. Any broader release requires wallet- or signer-local user
presence rather than this web-process trust.

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
  ethereum_rpc.py narrow, pinned-endpoint Base JSON-RPC client
  signer/         private stdio MCP signer, secure keyring, replay/outcome journals
  signer_outcome.py safe submission/re-simulation/unknown outcomes
  transaction_store.py bounded session-owned hash/status application state
  harness/        fixture-backed, network-free, read-only tools
  benchmark/      29 familiar/held-out eval cases with arguments and P6 blockers
  training/       deterministic SFT generation, leakage checks, prompt masking
  web/            FastAPI chat demo + optional loopback transaction review
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

Candidate-bound transfers remove recipient-address generation from the model
contract. For the live proof, deterministic code accepts only the exact narrow
current-message form `send <integer> wei to <0x address> on base`; Gemma can
select the argument-free `request_native_transfer_review` route but cannot
compose or alter its fields. The endpoint repeats checksum, chain, balance,
recipient-class, simulation, and policy checks. Missing, ambiguous, contract,
zero, self, and non-Base recipients fail closed.

Ollama, llama.cpp, and OpenRouter request native schema-constrained decoding.
The direct Transformers provider currently has post-hoc whole-output validation
only; it is explicitly non-conformant with the native constrained-decoding
preference and must independently pass the measured structured-output gate.

`scripts/run_web_poc.sh` also enables an in-memory, loopback-only development
transcript at `http://127.0.0.1:8000/debug/transcripts`. It is bounded, sends
no-store headers, requires both a loopback client and loopback Host header,
never writes messages to disk, and clears on server restart.
Set `AGENTIC_WALLET_DEBUG_TRANSCRIPTS=0` to disable it.

The Phase 8 signer is implemented but disabled by default and restricted to
direct loopback because the POC has no remote-user authentication. Do not place
signing mode behind a reverse proxy; common proxy headers are rejected, but
header rejection is not a substitute for authentication. Install the signer
extra, provision through the hidden-TTY script, and configure a Base RPC plus a
URL-safe base64 key that decodes to at least 32 random bytes:

```bash
pip install -e ".[dev,web,signer]"
python scripts/provision_signer_key.py

# Example generator; keep the resulting value secret.
openssl rand -base64 32 | tr '+/' '-_' | tr -d '='

AGENTIC_WALLET_TRANSACTION_ENABLED=true \
AGENTIC_WALLET_SIGNER_RPC_URL=https://your-base-rpc.example \
AGENTIC_WALLET_APPROVAL_HMAC_KEY='URL_SAFE_BASE64_VALUE' \
  uvicorn agentic_wallet.web.app:app --host 127.0.0.1
```

Never put a private key in `.env`; only the OS keyring may hold it. The UI
advertises signing only after signer and RPC readiness succeed. A successful or
ambiguous submission saves its local transaction hash in bounded app state and
returns a code-owned Basescan link. `check transaction <hash>` performs a
session-scoped lookup. An ambiguous broadcast is marked `UNKNOWN` and cannot be
signed again automatically. The signer fsyncs secret-free `UNKNOWN` recovery
metadata before broadcast, so a lost MCP response can recover the existing hash
without signing again. If recovery is also unavailable, the workflow enters
terminal `SUBMISSION_UNKNOWN` and cannot retry.

The Base RPC remains a privacy boundary even when inference is local. It can
observe the wallet address, balance and nonce reads, recipient, value-bearing
preflight, receipt lookups, and raw signed transaction. Use a fixed trusted or
self-hosted endpoint for privacy-sensitive testing; this POC does not verify an
RPC provider's retention policy.

The Android GGUF is an optional, separately downloaded model pack, never part
of the APK/AAB. See [docs/android-spike.md](docs/android-spike.md) for the
reproducible emulator measurements and the still-open real-device gate.
Use `scripts/android/run_benchmark.sh --resume`; it shuts down the dedicated
emulator automatically when the run exits.

## Fine-tuning feasibility

Four bounded E2B QLoRA experiments have run successfully on Hugging Face L4
hardware using completion-only masking and a pinned `google/gemma-4-E2B-it`
revision. V1 used 144 records and 20 steps; error-driven v2 used 576 records and
150 steps; staged-pipeline v4 used 240 records and 50 steps; candidate-bound v5
used 232 records and 75 steps.

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
python scripts/generate_training_data.py --profile transaction-boundary-v6
python scripts/train_qlora.py --dataset data/training/sft-v4-pipeline.jsonl
```

The v4 pipeline curriculum deterministically expands the fixed, naturally
phrased 64-record v3 source into 240 exact-runtime examples: 128 argument-free
routes (including route repairs) and 112 selected-action argument calls
(including argument repairs), split 180/60 for development. It includes typed
conversation ledgers and grounded typed results. This is development data, not
an independent safety evaluation. The sealed suite remains unavailable and must
not be opened or used for checkpoint selection.

The v5 candidate-binding curriculum preserves v4 as historical
evidence while adapting future routing data to the safer runtime. It contains
232 records (174 train, 58 development-validation). Candidate transfers appear
only as route decisions; the eight obsolete free-generated transfer-argument
and repair records are absent because deterministic code now owns those fields.
The production route target is now only the allowlisted `proposed_action`; code
owns the display envelope.

On the same Transformers runtime, the safety-selected checkpoint 25 improved
the independently authored 40-case development suite from 23/40 to 33/40 exact
with zero hard-zero failures, down from three for the base. Both were 40/40
schema-valid, so the one-field contract—not tuning—gets credit for formatting.
The paired exact McNemar p-value is 0.031, but this remains exploratory evidence:
the suite was used to choose among checkpoints and is not a sealed confirmatory
evaluation.

Later checkpoints overfit the narrow development curriculum. Checkpoint 50
reached 31/40 with one independent hard-zero failure; checkpoint 75 reached
29/40 with two and also introduced a development safety failure. Checkpoint
selection now prioritizes zero hard-zero failures before ordinary accuracy, so
checkpoint 25 is the v5 candidate for a future sealed evaluation. Separate-job
checkpoint-25 repeats reproduced both development reports byte-for-byte. On
the matching untuned local E2B pilot, raw routing was 7/12 while deterministic
required-fact checks produced 12/12 correct guarded outcomes and contained all
six hazardous cases.

The v6 transaction-boundary curriculum preserves v5 and adds 36 examples for
the loopback-only Base transfer proof: route-only transfer review and
session-scoped status lookup, application-owned approval, signer-request
isolation, freshness re-simulation, ambiguous broadcast handling, trusted
explorer links, response-loss recovery, and recipient preflight. The generated
dataset contains 268 records (197 train, 71 development-validation), with
SHA-256
`a567f33a094443909ea686a5f60f6c07d74d139b931f5faf12ab6534b9879ccc`.
It contains no private key, raw transaction, capability token, or RPC endpoint.

The untuned local Ollama E2B run on the 61 eligible V6.2 validation records scored
45/61 schema-valid, 39/61 correct actions, and 34/61 exact
action-and-arguments. It failed all 12 multi-argument records and produced seven
safety failures under the corrected development scorer, including a wrong-chain
review route and an adversarial signing-boundary choice. Deterministic candidate
binding, action allowlists, exact-ID lookup, approval separation, and signer
preflight prevented every observed output from authorizing or submitting a
transaction. These are development diagnostics and fine-tuning targets, not
evidence of release readiness. V6.3 adds one grounded narration for an
unrecoverable signer response; it is predeclared ineligible for the minimal-route
evaluator, so the 61 scored records and their reported outputs are unchanged.

The training path evaluates and checkpoints every 25 optimizer steps and uses a
safety-lexicographic development score. The
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
