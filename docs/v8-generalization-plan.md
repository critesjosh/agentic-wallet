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

### 1. Re-select checkpoint 50 now (zero cost)

Step 50 generalizes 15 points better than step 75 with the same zero safety
failures. Treat it as the current V7 candidate. Stop quoting the 92.2%
in-distribution number as a capability; the honest figure is ~77% at step 50.

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

### 3. Training regularization

- Target ~2 epochs, not 3. Recompute the step budget for the larger dataset
  (steps per epoch scale with example count).
- Add LoRA dropout (0.05-0.1) and modest weight decay; consider a slightly lower
  learning rate.
- Checkpoint every ~10-15 steps for finer selection.

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
