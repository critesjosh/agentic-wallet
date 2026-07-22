# Local Gemma 4 E2B development

The project targets Gemma 4 E2B because the effective model size is intended
for edge deployment. Local development currently supports three distinct paths:

- Ollama for the fastest host-side web and benchmark loop (`gemma4:e2b`).
- Hugging Face Transformers for checkpoint and future QLoRA experiments.
- Android llama.cpp for the historical GGUF runtime spike.

Install the normal web environment with `pip install -e ".[dev,web]"`. The
Transformers path additionally requires `pip install -e ".[ml]"` and loads
weights lazily so core imports do not require PyTorch.

Run the current local baseline with:

```bash
.venv/bin/python scripts/run_local_benchmark.py \
  --provider ollama --model-id gemma4:e2b
```

## Untuned baseline (2026-07-22)

The command above was run against the 22-case train/held-out suite using local
Ollama `gemma4:e2b`:

- exact action-and-argument passes: 5/22;
- schema-valid outputs: 6/22 (27.3%);
- train family: 4/14 exact, 5/14 schema-valid;
- held-out eval family: 1/8 exact, 1/8 schema-valid;
- hard-zero critical failures: 1 (`proceed_to_signing` selected at the signing
  boundary case);
- release-ready: false.

The 16 malformed outputs were rejected before any tool execution. These results
are the honest untuned baseline, not a release claim; see
[`model-failures.md`](model-failures.md) for fine-tuning targets.

The direct Transformers provider uses 4-bit NF4 by default. It performs strict
whole-output JSON validation but does not currently have native constrained
decoding, so its structured-output validity must be measured rather than
assumed. Do not begin QLoRA until the stock checkpoint-to-mobile-runtime path
and the expanded held-out benchmark are accepted.

Model files, adapters, runtime artifacts, credentials, and local transcripts
are deliberately excluded from Git. See `docs/android-spike.md` for the pinned
historical Android artifact and reproducible emulator evidence.
