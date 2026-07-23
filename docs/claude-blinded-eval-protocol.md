# Claude-authored blinded evaluation v9

This is a single-use experimental evaluation, not the independently
human-authored sealed release gate in
[`sealed-eval-protocol.md`](sealed-eval-protocol.md).

Claude Sonnet, invoked only through the directly authenticated Claude Code CLI,
authors novel utterances, identifiers, typed fixtures, and
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

Protocol v7 was retired before authoring when the user prohibited calling
Claude through OpenRouter. Direct Claude Code Fable 5 was unavailable because
the account had reached its monthly Fable spend limit.

Protocol v8 was retired before commitment or candidate evaluation. Its first
attempt failed deterministic compilation, and an audit stopped the second
attempt after finding that the commitment path did not independently enforce
the exact shard shape and that the one-shot ledger path was caller-selectable.
No v8 plaintext is reused.

Each v9 whole-suite attempt uses Claude Sonnet through the directly
authenticated Claude Code CLI: eight independently generated eight-case batches
with fixed quotas, one trajectory per batch, and the CLI's strict JSON Schema
response format. Tools are disabled, safe mode is enabled, and session
persistence is off. The constrained outer response is an array of eight
strings. Each string encodes one complete source record; deterministic code
requires strict JSON, exact top-level fields, a known scenario type, and an
object context before subjecting it to the unchanged scenario compiler.
This keeps generation inside the transport limit
without giving later batches access to earlier plaintext. The workflow allows
at most two whole-suite attempts. Each shard has a distinct fixed ID prefix,
scenario quota, and one four-turn trajectory. The materializer and commitment
path both require all eight original shard files in canonical order, recompile
them, and compare their exact canonical suite bytes. An invalid or overlapping
suite is discarded in full; Claude receives no case-level feedback.

The author prompts contain only public project contracts and fictional
synthetic fixtures—never user, wallet, key, account, or private repository
data. Claude runs in an isolated external directory and cannot read the
training checkout.
After a suite passes schema and post-hoc disjointness checks, its digest, all
eight source-shard digests, all author prompts, author request and commitment
scripts, candidate artifact, complete evaluator behavior, exact inference
environment, sequence mode, and attempt count are committed and pushed before
the model runs. The commitment script refuses drift in that tracked scope and
binds the recorded harness revision to the actual Git HEAD.

The plaintext remains outside the checkout and is kept in private external
custody. The candidate is evaluated exactly once on the frozen Hugging Face L4
configuration. The ledger and aggregate paths are derived from the suite digest
under one frozen persistent bucket mount; callers cannot select alternate
paths. The durable `claimed` state is already irreversible, so host loss or
SIGKILL cannot reopen the suite. Normal success refines it to `completed`; a
handled failure refines it to `retired`. Aggregate publication is atomic and
create-once. Only aggregate results may return to the checkout.

At execution time the evaluator re-hashes the live canonical suite, selected
adapter, evaluator script, and complete harness behavior against the committed
values before or immediately after the irreversible claim as appropriate.
Author sources and prompts are not needed by candidate inference; their digests
establish the provenance of the already-bound canonical suite. The GPU job is
launched without SSH or an exposed port, and the suite mount is used only by
the evaluator process.

If Claude Code reaches its limit before a complete attempt is produced, that
whole attempt is discarded. A Terra-authored replacement must first be frozen
as a new protocol revision with explicit author provenance and fresh prefixes;
Claude and Terra shards are never silently mixed under the v9 identity.

Trajectories use teacher-forced typed context, not autonomous rollout. Transfer
candidate fields bound by deterministic code are reported separately from
model-generated argument metrics. Hard-zero scoring uses the raw route before
deterministic clarification so the guard cannot hide an unsafe model choice.

This design provides stronger evidence than another generator-derived
development split, but the developer still operates the process and Claude is
a model author. Therefore both `human_independence_attested` and
`release_claim_eligible` remain false. A separately authored human suite is
still required for a release claim. Sixty-four cases also have limited power
for small accuracy differences. External custody is a procedural boundary:
the custodian can technically inspect plaintext, and pretraining contamination
outside the repository cannot be ruled out. These results support an
experimental direction, not a population-level or release-safety claim.
