# Agent instructions

Read [plan.md](plan.md) before proposing architecture, especially the Consensus
revisions block at the top. This file is the short operational version.

## Non-negotiables

- Preserve the split: the model proposes, deterministic code enforces, the
  wallet holds signing authority. Security must not depend on the model behaving.
- Keep live blockchain facts out of model weights; they come from typed tools.
- Never give the model a seed phrase or private key. Never treat model output as
  authorization. Never execute free-form text as a wallet action.
- Prefer narrow typed tools over general execution interfaces.
- Untrusted retrieved text goes only in `UntrustedData`, never into an
  actionable field (plan.md P5).
- Amounts are integer base units or decimal strings, never floats.
- Add tests for every new workflow state and tool; add adversarial cases too.
- Do not add signing authority without an explicit task requiring it.
- Record reproducible model-output failures in
  [`docs/model-failures.md`](docs/model-failures.md), including the model/runtime,
  expected contract, observed output, fail-closed behavior, and potential
  fine-tuning target.

## Layout and commands

Core package: `src/agentic_wallet/` (schemas, `state_machine`, `digest`,
`registry`, `inference`, `harness/`, `benchmark/`, `web/`).

```bash
pip install -e ".[dev,web]"
pytest
python scripts/export_schemas.py
uvicorn agentic_wallet.web.app:app --reload
```

## Notebook workflow

`gemma-4-E4B.ipynb` is a thin GPU driver (Colab). Logic lives in the package and
is imported; keep only model load, training, and inference in cells. `nbstripout`
is a git filter, so commit stripped outputs. Prefer editing/testing `.py` code
over running notebook cells where possible.
