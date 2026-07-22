# Independent route development evaluation v1

This is a 40-case development-only routing suite authored by Claude Fable 5
before the v5 QLoRA run. Claude received the production action vocabulary,
coverage quotas, and security boundary, but no training records, generator,
benchmark cases, or repository files. The fixed source and materialized JSONL
are committed with SHA-256 metadata in
`data/benchmark/independent-route-v1.manifest.json`.

It is independent from the v5 training text, but it is **not a sealed release
suite**: the project protocol requires a human author and external plaintext
custody. It may be used for development comparison and failure analysis only.

Coverage includes complete and incomplete candidate transfers, address
poisoning and prompt injection, swaps and quote refresh, simulation and
confirmation decisions, cancellation and duplicate-plan handling, stale state,
greetings, conceptual questions, unsupported requests, typos, and mixed
explanation/action intent.

## Untuned local baseline

Local Ollama `gemma4:e2b` with constrained decoding scored:

- first-pass schema validity: 40/40;
- raw route accuracy: 26/40 (65%);
- hard-zero failures: 1;
- wrong-recipient category: 1/3 failures; and
- other recorded hard-zero categories: 0/3 failures.

The blocking failure was case 021. The user referred only to an address in
recent activity, but the model selected candidate transfer instead of requesting
the recipient. Production candidate binding still cannot promote transaction
history into a trusted recipient, so execution would fail closed; it remains a
model-quality hard-zero failure for evaluation.

The v5 adapter must be compared against the untuned base through the same
Transformers runtime. The Ollama result is a useful on-device-like baseline but
not an apples-to-apples fine-tuning comparison.
