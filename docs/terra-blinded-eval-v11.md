# Terra-authored blinded evaluation v11

Version 10 attempt 1 was retired before commitment or candidate evaluation when
a source fixture failed deterministic compilation. The compiler error also
revealed field-level values in local diagnostics. No v10 record is reused.

Version 11 uses fresh `tb9...` prefixes. The author prompt requires `0x` plus 40
lowercase hex characters for every address and
`recipient:[a-z0-9-]+` for verified recipient IDs. Deterministic source
validation enforces the address rule on every address-valued field and every
`0x` address token; candidate compilation enforces the recipient rule.
Materialization failures have no retained compiler exception chain and expose
only an aggregate message. All other v10 one-shot, disjointness, custody,
frozen-candidate, aggregate-only, and non-release controls remain unchanged.
