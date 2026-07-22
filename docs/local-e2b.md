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

## Untuned 22-case baseline (2026-07-22, historical)

The command above was run against the then-current 22-case train/held-out suite using local
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

The frozen benchmark now contains 29 cases, including at least two cases for
every hard-zero category. Run the command again for the current baseline; never
use either benchmark family as SFT training text.

## Untuned 29-case baseline (2026-07-22, current)

- exact action-and-argument passes: 5/29;
- schema-valid outputs: 7/29 (24.1%);
- familiar family: 4/19 exact, 6/19 schema-valid;
- held-out family: 1/10 exact, 1/10 schema-valid;
- hard-zero critical failures: 2 (both selected `proceed_to_signing` instead of
  respecting approval invalidation or requesting exact confirmation);
- release-ready: false.

All 22 malformed outputs failed closed before tool execution. This result is the
comparison baseline for a future adapter evaluated through the same provider
contract.

The direct Transformers provider uses 4-bit NF4 by default. It performs strict
whole-output JSON validation but does not currently have native constrained
decoding, so its structured-output validity must be measured rather than
assumed. QLoRA data and script plumbing may be dry-run now, but dataset-scale
training, merging, and conversion remain gated on physical-device P2 evidence.

Model files, adapters, runtime artifacts, credentials, and local transcripts
are deliberately excluded from Git. See `docs/android-spike.md` for the pinned
historical Android artifact and reproducible emulator evidence.
