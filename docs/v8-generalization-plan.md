# V8 plan: closing the generalization gap

## The problem, measured

The V7 disjoint evaluation showed the step-75 adapter memorizes training-set
surface forms rather than generalizing:

| Checkpoint | In-distribution | Disjoint (novel wording) | Gap |
| --- | ---: | ---: | ---: |
| 25 (~1 epoch) | 64.1% | 55.0% | 9.1 |
| 50 (~2 epochs) | 84.4% | 77.5% | 6.9 |
| 75 (~3 epochs) | 92.2% | 62.5% | 29.7 |

Disjoint accuracy peaks at ~2 epochs and falls 15 points by 3 epochs while the
in-distribution score keeps rising. Two root causes: too few, too uniform
training utterances (about 12 near-duplicate templates per intent), and one
epoch too many.

## What the research says

- **Diversity is the primary lever.** Instruction expansion, roughly 5-10x
  varied paraphrases per seed with different phrasing, register, and structure,
  forces generalization instead of memorization. Diversity, alongside quality
  and difficulty, is the metric that most improves robustness.
- **Fewer epochs for small data.** Single-epoch LoRA is often the practical
  optimum on limited data; extra epochs exacerbate overfitting. Our disjoint
  peak at ~2 epochs is consistent.
- **Select on out-of-distribution, not validation loss.** OOD success can
  diverge from val loss, so the held-out disjoint suite, not in-distribution
  accuracy or loss, must choose the checkpoint. We now have that suite.
- **Continued pre-training does not help here.** Additional pre-training did not
  improve instruction-tuning task performance in the literature, and our problem
  is surface-form generalization, not missing domain knowledge. Not pursued.
- **Synthetic data has failure modes.** Iterated self-generated data can cause
  model collapse and reduced diversity, so paraphrases must come from a stronger
  model, be quality-filtered and deduplicated, and be diversity-measured.

## Plan, in priority order

### 1. Re-select checkpoint 50 now (zero cost) — done

Step 50 generalizes 15 points better than step 75 with the same zero safety
failures. Treat it as the current V7 candidate. Stop quoting the 92.2%
in-distribution number as a capability; the honest figure is ~77% at step 50.

Recorded in `docs/fine-tuning.md` under "Selection: the V7 release candidate is
checkpoint-50". No source config pinned a release checkpoint (the adapter is
passed per run via `AGENTIC_WALLET_ADAPTER`), so the selection lives in the
decision record rather than in code.

### 2. V8 dataset: diversity augmentation (primary fix)

Keep V7's proven split of duties: a model authors only the natural utterance and
scenario, deterministic code derives every gold action and argument. Change the
volume and spread of the authored surface forms.

- For each intent and each adversarial family, generate 5-10x paraphrase
  variants: vary wording, register (terse, formal, casual), sentence structure,
  and synonyms. Grow the account and refusal families from ~12 templates to
  50-100 distinct utterances; broaden the read and transfer families similarly.
- Randomize identifiers per example (addresses, hashes, amounts) so the model
  cannot anchor on specific tokens.
- Author through Claude Code only, never OpenRouter or Fable, per the standing
  rule. Quality-filter and deduplicate; measure diversity (distinct-n and
  embedding spread) and reject low-diversity batches to avoid collapse.
- Keep the disjoint suite strictly held out. It is an evaluation set and must
  never seed training data, or it stops measuring generalization.

Landed (first increment). The plain-context account, read, and refusal cluster
is authored and generated: `src/agentic_wallet/training/account_curriculum_v8.py`
produces `data/training/sft-v8-account-diversity.jsonl` (profile
`account-diversity-v8`), growing V7's 12 fixed additions to 90 phrasing-diverse
examples across `get_account`, `get_portfolio`, `get_balance`, `get_allowances`,
`get_registry`, `show_help`, `reject_state_changing`, key-disclosure refusals,
untrusted-note refusals, and out-of-scope conversation. The frozen v6 base is
inherited byte-for-byte. Identifiers are randomized from a fixed seed, so the
committed dataset digest is reproducible. Diversity is measured per family
(`training/diversity.py`, distinct-n and pairwise spread) and recorded in the
manifest; `assert_diverse` rejects a low-diversity or near-duplicate batch, and a
disjointness gate fails closed if any authored utterance or identifier appears in
the held-out suite. Embedding-spread measurement is a documented future
extension; the fail-closed gate stays lexical so generation is offline and
deterministic.

Next increment. Grow the transfer-review and transaction-status families, which
route from a parsed candidate in context rather than plain IDLE text, and extend
the same paraphrase-and-randomize treatment to the swap/approval argument-filling
families in the v6 base.

### 3. Training regularization

- Target ~2 epochs, not 3. Recompute the step budget for the larger dataset
  (steps per epoch scale with example count).
- Add LoRA dropout (0.05-0.1) and modest weight decay; consider a slightly lower
  learning rate.
- Checkpoint every ~10-15 steps for finer selection.

Landed (plumbing). The regularization knobs are now first-class and recorded in
the run plan and training metadata: `train_qlora.py` takes `--weight-decay`,
`--lora-dropout`, and `--save-total-limit`; `run_hf_qlora_smoke.py` and
`launch_hf_training.py` forward `--checkpoint-steps` (driving matched eval/save
steps) plus learning rate, weight decay, dropout, and save-total-limit as
optional env overrides. Every override defaults to the V7 value, so omitting them
reproduces the V7 launch exactly. `run_hf_disjoint_checkpoint_eval.py` now
discovers `checkpoint-N` directories from the adapter instead of assuming
25/50/75, so the finer curve is read from whatever the run wrote.

Step budget for V8 (261 train examples, effective batch 8) is ~33 steps/epoch.
The recommended first run: `--max-steps 66` (~2 epochs), `--checkpoint-steps 11`
(six checkpoints from ~1 to ~2 epochs), `--save-total-limit 8` (retain the whole
curve), `--lora-dropout 0.1`, `--weight-decay 0.01`, learning rate held at 2e-4
so diversity and epoch count are the isolated variables this run tests.

### 4. Selection protocol change

Every training run ends with the per-checkpoint disjoint curve
(`run_hf_disjoint_checkpoint_eval.py`). The release checkpoint is the disjoint
maximum with zero hard-zero failures, not the in-distribution or loss optimum.

### 5. Not doing

- **Continued pre-training on domain text.** Wrong tool for surface-form
  generalization; the research and our diagnosis agree it would not help.
- **Inference-time skills for the fine-tuned model.** Measured three times to
  degrade the adapter. Revisit only for a general, untuned on-device deployment
  with Gallery-style gated injection, which is a different product.

## Success criteria

Retrain V8, run the disjoint curve, and require: disjoint accuracy materially
above V7's 77.5%, the in-distribution-minus-disjoint gap under about 10 points at
the selected checkpoint, and zero hard-zero failures on the disjoint suite. A
true generalization claim, versus this development evidence, still requires the
sealed-evaluation protocol and is out of scope for routine iteration.
