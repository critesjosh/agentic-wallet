# Terra blinded-author procedure v4

V12 attempt 1 was retired after one shard remained invalid through three
aggregate-only generations. No source was committed or evaluated.

V13 retains isolated `gpt-5.6-terra` authors and the frozen public compiler
validator. The validator now returns only a line number and a small,
predeclared, value-free structural error code. It never returns source values,
training/development data, gold labels, or candidate-model feedback.

Each author may replace its whole shard and validate up to three total
generations. Root receives only final aggregate status. Final shards are still
independently recompiled, quota checked, disjointness audited, committed, and
used for one candidate run. Validator-conditioned authorship remains an
explicit compiler-oracle selection limitation.
