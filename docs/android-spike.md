# Android inference spike

Status: **emulator sub-gate complete; real-device P2 gate still open**.

This spike tests whether the target model family can load and produce
grammar-constrained tool selections in an Android runtime. It does not prove
real-device speed, memory pressure, battery impact, or thermals.

## Artifact and runtime

- AVD: `Agentic_Wallet_API_36_1`, Android API 36.1, x86_64
- Emulator allocation: 8 GiB RAM, 4 CPU cores, 12 GiB data partition
- Runtime: llama.cpp commit `846e991ec3c7ccec49112ff2c5b00b710e5f551d`
- Model: `google/gemma-4-E2B-it-qat-q4_0-gguf`
- Hub revision: `675cff42a74c774d6cb76f76d8eacb49b48c9b93`
- File: `gemma-4-E2B_q4_0-it.gguf`
- File size: 3,349,516,256 bytes (3.12 GiB)
- SHA-256: `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`
- Multimodal projector: not downloaded; this is a text-only spike

The model is an optional development artifact and is ignored by Git. It must
not be bundled into the Android app. The intended product design is a small app
with transparent remote inference initially and a separately downloaded,
hash-verified local model pack as an opt-in privacy mode.

The exact BF16 `google/gemma-4-E2B` checkpoint is 10.2 GB. Local conversion to
GGUF was attempted with llama.cpp, including temporary-file conversion, but the
host OOM killer terminated the converter at about 16.1 GB RSS. The host has no
swap. The official instruction-tuned QAT Q4 artifact was therefore used for the
Android runtime spike. This substitution must remain explicit when comparing
results.

## Reproduction

The prebuilt Android llama.cpp binaries, shared libraries, and model are
expected under `/data/local/tmp/agentic-wallet`. Start the detached emulator and
resident server with:

```bash
scripts/android/start_emulator.sh
scripts/android/start_llama_server.sh
.venv/bin/python scripts/run_android_benchmark.py --resume
```

For normal use, prefer the cleanup-safe wrapper:

```bash
scripts/android/run_benchmark.sh --resume
```

It installs an exit/signal trap that shuts down the dedicated AVD after the
benchmark. Set `KEEP_ANDROID_RUNNING=1` only during active debugging. Run
`scripts/android/stop_emulator.sh` when finished with an interactive session.

Use `scripts/android/stop_llama_server.sh` for a graceful stop-and-wait before
testing a restart. This avoids the short Android process-exit window in which
`pidof` can still report a server that has already received `SIGTERM`.

The runner atomically checkpoints each case under `artifacts/`, so an agent,
terminal, emulator, or server interruption does not erase completed cases.
`scripts/android/status.sh` separates an ADB/forwarding problem from an emulator
or model-server failure.

## Results

This historical run used the original six-case benchmark and llama.cpp native
`json_schema` decoding with the
currently available actions embedded as an enum. All generations were accepted
again by the Pydantic `ToolCall` schema and action allowlist before scoring.

| Metric | Result |
| --- | ---: |
| Schema-valid output | 6/6 (100%) |
| Correct action | 3/6 (50%) |
| Critical semantic failures | 3 |
| Prompt speed | 19.3-28.1 tokens/s |
| Generation speed | 2.39-3.11 tokens/s |
| End-to-end case latency | 26.2-35.2 seconds |
| Resident process RSS after run | 3,481,260 KiB |

Critical failures were:

- Requested a swap quote when the chain was missing.
- Drafted a transfer despite insufficient funds.
- Selected an unlimited approval instead of an exact approval.

The benchmark has since expanded to 29 cases with typed argument scoring and
separate train/held-out reporting. These six-case numbers remain reproducibility
evidence for the emulator run, not the current quality baseline.

These results block release. They do not create an execution vulnerability in
the current architecture because model output is only a proposal: deterministic
tool-specific argument, balance, policy, simulation, and approval checks must
reject these choices. Native grammar guarantees syntax and action membership;
it does not guarantee semantic correctness.

## Crash diagnosis

Two different failures were observed:

1. BF16-to-GGUF conversion was genuinely OOM-killed on the host at roughly
   16.1 GB converter RSS with no swap.
2. The later apparent Android crash was lifecycle/tooling-related. The emulator
   had been owned by a transient execution session and no emulator process
   remained after interruption. A subsequent sandboxed ADB invocation also
   failed to bind its local smart socket, which looked like an Android failure
   but occurred before contacting any device.

The detached `setsid` launcher, device-side `nohup` server, explicit ADB health
checks, and per-case checkpointing address the second failure mode.

## Remaining P2 evidence

Before dataset generation or QLoRA training, run the same pinned artifact and
benchmark on at least one physical mid-range Android device and record:

- cold model-load time and time to first token;
- peak app/system memory and low-memory behavior;
- sustained generation speed across repeated cases;
- battery draw, skin/SoC temperature, and thermal throttling;
- download, integrity verification, deletion, and insufficient-storage UX;
- constrained-output validity and deterministic fail-closed behavior.

Emulator thermal readings are synthetic and are not P2 evidence.
