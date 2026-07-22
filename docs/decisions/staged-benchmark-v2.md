# Staged benchmark protocol v2

Status: frozen before the v4 adapter evaluation-only rerun.

The historical 29-case suite remains immutable development-regression data.
Only its inference protocol changes, because the production model contract was
split into route and argument stages before v4 training.

For every case, `staged-dialogue-route-v2` performs:

1. Build a typed `ConversationLedger` from the case workflow state and chain.
   It has no approval field.
2. Ask for an argument-free `DialogueRoute` using the case's complete
   state-scoped action list and no suggestion chips.
3. Apply the production retry rule: at most one repair for returned output that
   fails validation; no retry for transport failures or non-repairable signing
   and unlimited-approval selections.
4. If the validated route selects an action, ask for `ToolCall` arguments with
   exactly that one selected action exposed. Apply the same bounded repair rule.
5. Score the final validated action and arguments against the existing expected
   values. Preserve the existing safe-refusal exception and hard-zero scoring.

The cases, expected outputs, hard-zero categories, base model, v4 adapter, and
greedy decoding settings do not change. A test asserts the route call receives
the full action set and the argument call receives only the selected action.

The rerun is a protocol-sensitivity measurement on fixed weights, not a model
improvement. Reports must show the retired single-stage 0/29 result separately,
identify it as non-comparable, and label the corrected score as offline
development regression only. The sealed suite remains unused.

The v4 20-record checkpoint-selection subset is intentionally drawn from the
same 60-record development-validation partition. Therefore the 60-record score
is checkpoint-development evidence and may be optimistic; it is not an
independent test. The corrected 29 cases were also used during earlier dataset
design and are development regression data. Neither supports a release,
generalization, execution-safety, or model-selection claim.
