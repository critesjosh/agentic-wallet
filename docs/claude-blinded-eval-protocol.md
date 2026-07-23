# Claude-authored blinded evaluation v7

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

Protocol v2 was retired before authoring because no available Claude endpoint
could simultaneously honor strict JSON Schema output and OpenRouter's
zero-data-retention routing filter.

Protocol v3 was retired before authoring because its raw schema used
constraints unsupported by Claude structured outputs and an open-ended nested
context object.

Protocol v4 was retired before authoring because OpenRouter's compatible
provider route rejected the nested record schema even after unsupported
constraints were removed.

Protocol v5 was retired before authoring because the compatible Azure
structured-output endpoint does not accept an explicit `temperature` parameter
when OpenRouter is required to preserve every supplied parameter.

Protocol v6 was retired after its first whole-suite attempt when a fixture-heavy
16-case response was truncated by the provider. No partial shard is reused.

Each v7 whole-suite attempt uses eight independently generated eight-case batches
with fixed quotas, one trajectory per batch, and OpenRouter's strict JSON
Schema response format. The constrained outer response is an array of eight
strings. Each string encodes one complete source record; deterministic code
requires strict JSON, exact top-level fields, a known scenario type, and an
object context before subjecting it to the unchanged scenario compiler.
This keeps generation inside the transport limit
without giving later batches access to earlier plaintext. The workflow allows
at most two whole-suite attempts. An invalid or overlapping suite is discarded
in full; Claude receives no case-level feedback.

The author prompts contain only public project contracts and fictional
synthetic fixtures—never user, wallet, key, account, or private repository
data. The compatible structured-output route is not constrained to
zero-data-retention providers; this exception applies only to synthetic suite
authoring and does not change the product's inference privacy boundary.
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
