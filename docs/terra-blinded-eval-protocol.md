# Terra-authored blinded evaluation v10

Claude-authored v9 was retired without commitment or candidate evaluation after
Claude Code reached its monthly spend limit during its first whole-suite
attempt. No v9 shard is reused.

Version 10 retains the v9 deterministic compiler, exact 64-case shape,
recursive disjointness audit, create-once commitment, frozen candidate/runtime,
commitment-keyed persistent ledger, claim-before-plaintext ordering, and
aggregate-only single evaluation. The author changes to eight isolated
`gpt-5.6-terra` subagents using
[`terra-blinded-author-procedure-v1.md`](terra-blinded-author-procedure-v1.md).
Fresh `tb8...` prefixes prevent accidental reuse of v9 records.

The root requests no forked conversation and provides no repository plaintext.
Subagents are instructed not to read the checkout; current collaboration tools
do not provide a cryptographic sandbox attestation, so this remains an
operator-recorded procedural boundary. Each subagent writes one external
eight-case shard and returns only an aggregate status. The materializer and
commitment path require all eight shards in canonical order and reproduce the
exact suite bytes.

This remains model-authored, operator-mediated experimental evidence.
`human_independence_attested` and `release_claim_eligible` are both false. The
statistical, compiler-oracle, pretraining-contamination, and custodian-access
limitations documented for v9 still apply.
