# Terra-authored blinded evaluation v12

V11 was retired before commitment or candidate evaluation after both permitted
whole-suite attempts failed aggregate deterministic validation. No source is
reused.

V12 uses fresh `tb10...` prefixes and a frozen aggregate-only validator that
each isolated author may run against its own shard for at most three total
generations. Authors still do not supply gold labels, see training/development
data, see other shards, or receive candidate feedback. The validity bit does
come from the complete deterministic scenario compiler, including gold
derivation and candidate-binding feasibility. Bounded author retries therefore
condition the source distribution on that compiler oracle; this is an explicit
methodological limitation, not independent raw authorship. Root still performs
the final strict compilation, disjointness audit, digest commitment, and
one-shot candidate evaluation.

The evidence remains model-authored, operator-mediated, non-human-independent,
and ineligible for a release claim.
