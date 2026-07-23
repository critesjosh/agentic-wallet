# Terra blinded-author procedure v5

V14 replaces author-written benchmark fixtures with eight language seeds per
shard. The external author receives only
[`terra-blinded-author-shared-v5.md`](terra-blinded-author-shared-v5.md), its
one v5 shard prompt, and a designated path outside the checkout. It has no
repository, training, development, prior-suite, candidate, or validator access.
It writes exactly one eight-line JSONL shard to that external path and returns
only a completion status; its JSONL is never echoed through the control plane.

Each seed has exactly `scenario_type`, `utterance`, `world_seed`,
`trajectory_key`, and `turn_index`. The shard prompt supplies the frozen fresh
`tb12...` prefix and quota. It contains exactly four independent seeds and one
four-turn trajectory whose scenario sequence is frozen by the shard prompt.
Authors do not write trusted facts, identifiers, action arguments, or labels.

After all submissions, the operator invokes the frozen seed expander and its
aggregate validator. The response is only aggregate validity and, on success,
the expanded case count; it provides no source values, line number, error code,
compiler exception, label, or candidate result. An invalid shard invalidates
the entire suite attempt and is not repaired using validator feedback.

The root independently materializes the eight shards in fixed compiler-prefix
order, checks the exact 64-case and quota shape, audits disjointness, commits
the canonical expanded bytes, then performs one frozen candidate evaluation.
No candidate evaluation occurs before successful commitment. The resulting
evidence remains model-authored, operator-mediated, and ineligible for a
human-independent or release-safety claim.
