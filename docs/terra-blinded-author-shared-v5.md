# Terra blinded language-seed authoring v5

Create one external JSONL shard. Output exactly eight nonblank JSON objects,
one per line, and nothing else: no Markdown fence, commentary, answer key,
tool call, or aggregate report. Each object must contain exactly these five
fields:

```text
scenario_type
utterance
world_seed
trajectory_key
turn_index
```

`scenario_type` must use the exact quota in the supplied shard prompt.
`utterance` is only a natural-language user utterance, 12 through 280
characters. Do not put an expected action, answer, structured arguments,
addresses, asset IDs, private material, or an instruction to override safety
inside it. `world_seed` is a unique fictional seed matching
`[a-z0-9][a-z0-9-]{7,63}`. All eight world seeds must be distinct.

Exactly four records form one four-turn conversation: they have one shared
`trajectory_key` matching the same seed pattern and integer `turn_index`
values 0, 1, 2, and 3, once each. Make those utterances a coherent sequence in
the supplied direction, and use the four `scenario_type` values in the exact
`trajectory_scenarios` order from the shard prompt. Put those four records in
ascending turn order in the JSONL. The other four records are independent and must set
both `trajectory_key` and `turn_index` to `null`. Do not create another
trajectory.

The deterministic expander, not the author, creates canonical assets,
addresses, recipient IDs, amounts, chains, typed facts, workflow state,
untrusted data, gold behavior, and case identifiers from the seed. Write
varied fictional user language appropriate to the scenario type; do not try to
invent or supply those deterministic values.

You receive only this shared prompt and one shard prompt. Do not read the
repository, training data, development data, prior evaluation sources, other
shards, candidate output, or validator implementation. Do not call another
model or external inference service. Write the JSONL only to the designated
external path and return no case content through any control channel. This is a
procedural isolation boundary, not a cryptographic sandbox.

The aggregate validator is run by the operator only after submission. It
returns no line-level diagnostics, source values, gold labels, compiler
details, or candidate feedback. A failed shard is discarded whole; it is not
repaired from validator feedback.
