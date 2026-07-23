# Terra blinded-author procedure v1

This procedure is used only after the directly authenticated Claude Code
interface reports a provider spend limit.

For each of the eight frozen shard prompts, the root agent spawns one
`gpt-5.6-terra` subagent with `fork_turns="none"`. The task message contains
only `terra-blinded-author-shared-v1.md`, that shard's prompt, the absolute
external output path, and these controls:

- Do not read, search, list, or otherwise inspect the repository. This is an
  instruction boundary, not a technically restricted tool sandbox.
- Do not call another model or subagent.
- Generate exactly the requested eight JSONL records.
- Write plaintext only to the supplied path under `/tmp`.
- Return only aggregate success or failure, never case content.

The root requests `fork_turns="none"` and the `gpt-5.6-terra` model explicitly;
these are operator-recorded invocation settings rather than a cryptographic
attestation. At most three author subagents run concurrently. Later subagents receive no
earlier output. The root agent does not read case plaintext. Deterministic code
requires all eight original source files, validates every fixed quota and
prefix, compiles gold labels, audits disjointness, and discards the whole
attempt on any failure.

At most two whole-suite Terra attempts are allowed. Claude and Terra shards are
never mixed. The commitment records the Terra model identity, this procedure
digest, all prompt and source digests, and
`human_independence_attested=false`.
