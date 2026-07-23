# Terra-authored blinded evaluation v14

V14 uses fresh `tb12...` prefixes and external language-only seed authorship.
Each of eight isolated authors supplies eight JSONL seeds: four independent
cases and one four-turn trajectory. Deterministic code expands every seed into
the complete typed fixture, trusted identifiers, workflow state, and derived
gold behavior, then validates the aggregate suite before commitment and one
frozen candidate run. Authors have no repository, training/development,
candidate, or validator access, and their output stays external to the
checkout.

This design reduces fixture-format failures and prevents author-supplied facts
from becoming trusted inputs. It does not make the evaluation independent:
language variation is model-authored; scenario families, fixture expansion,
and labels are code-owned; shared generation and compiler assumptions can
bias results; and the isolation boundary is procedural rather than
cryptographically attested. A finite passing suite cannot establish a literal
zero rate for the hard-zero safety failures. V14 is therefore
operator-mediated experimental evidence, not a release claim or proof of
wallet safety.

The four-turn records use a shared deterministic fictional world and carry
bounded typed prior-user history. Evaluation is still
`teacher-forced-typed-context`: every turn is scored independently against its
frozen fixture, and candidate output does not mutate the next turn. Sequence
accuracy means all four independently scored turns passed; it is not evidence
that a free-running model maintained application state by itself.

## Outcome

The first whole-suite authoring attempt was retired after one of eight shards
failed aggregate deterministic validation. The second and final permitted
attempt produced 64 structurally valid cases, but the frozen pre-commit
disjointness audit reported existing context-text overlap. The operator did not
inspect or repair the overlapping records. No commitment was created, no
plaintext was uploaded, and the candidate model was never evaluated against
either attempt. V14 is closed without an accuracy result; further protocol
revision requires a new explicit decision rather than automatic regeneration.
