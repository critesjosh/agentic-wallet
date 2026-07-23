# Claude-authored blinded evaluation v2

This is a single-use experimental evaluation, not the independently
human-authored sealed release gate in
[`sealed-eval-protocol.md`](sealed-eval-protocol.md).

Claude Fable 5 authors novel utterances, identifiers, typed fixtures, and
trajectory composition without receiving repository training/development
plaintext. It does not author gold labels. A versioned deterministic scenario
compiler derives actions and arguments and rejects unknown or answer-key
fields.

Protocol v1 was retired without evaluation after both permitted authoring
attempts failed the JSONL/compiler boundary. No v1 record was repaired or shown
to the candidate.

Each v2 whole-suite attempt uses four independently generated 16-case batches
with fixed quotas, two trajectories per batch, and OpenRouter's strict JSON
Schema response format. This keeps generation inside the transport limit
without giving later batches access to earlier plaintext. The workflow allows
at most two whole-suite attempts. An invalid or overlapping suite is discarded
in full; Claude receives no case-level feedback.
After a suite passes schema and post-hoc disjointness checks, its digest, all
author prompts, candidate artifact, harness, evaluator, inference
configuration, sequence mode, and attempt count are committed before the model
runs.

The plaintext remains outside the checkout and is kept in private external
custody. The candidate is evaluated exactly once. A post-commit execution
failure retires the suite without a rerun or case-level inspection. Only
aggregate results may return to the checkout.

Trajectories use teacher-forced typed context, not autonomous rollout. Transfer
candidate fields bound by deterministic code are reported separately from
model-generated argument metrics. Hard-zero scoring uses the raw route before
deterministic clarification so the guard cannot hide an unsafe model choice.

This design provides stronger evidence than another generator-derived
development split, but the developer still operates the process and Claude is
a model author. Therefore both `human_independence_attested` and
`release_claim_eligible` remain false. A separately authored human suite is
still required for a release claim.
