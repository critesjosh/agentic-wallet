# Terra blinded-author procedure v2

For each frozen shard, the root spawns one `gpt-5.6-terra` subagent with
`fork_turns="none"`. Its task contains only
`terra-blinded-author-shared-v2.md`, one shard prompt, and an external output
path. It is instructed not to inspect the repository, call another model, or
return case content. This non-access rule is procedural rather than a
cryptographic tool sandbox.

Each subagent writes exactly eight direct JSONL records under `/tmp` and returns
only aggregate status. At most three run concurrently, and no author receives
another shard. Deterministic code validates all original files, fixed quotas,
prefixes, trajectories, fixtures, and disjointness. Compiler failures are
reported generically so field-level plaintext cannot leak through exceptions.

A failed attempt is discarded in full. At most two v11 attempts are allowed.
The commitment records the operator-requested model/isolation settings, this
procedure and all prompt/source digests, and the model-authored/non-release
disclosures.
