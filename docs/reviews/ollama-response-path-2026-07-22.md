# Ollama E2B response-path review

## Failure analysis

The original candidate-binding pilot failed 12/12 before routing. Ollama
returned an unfinished `thinking` channel, empty final content, and
`done: false`. Disabling thinking produced a complete response but the five-field
route schema still yielded Markdown-fenced, incomplete JSON.

A three-call controlled probe reduced the schema to one enum field,
`proposed_action`, while retaining `think: false`. All three calls returned bare
schema-valid JSON with `done: true`. This isolated two independent causes:

1. thinking mode prevented this model/runtime combination from reaching final
   structured content; and
2. the broad route envelope asked E2B to co-generate presentation prose and a
   security-relevant routing decision.

## Claude Fable 5 review

A sanitized packet containing runtime metadata, observed outputs, and the
security contract was reviewed through OpenRouter. No source files, secrets,
wallet data, or credentials were sent. Claude independently recommended:

- explicitly disable thinking for structured routes;
- reject `done: false`, abnormal stop reasons, and empty final content before
  parsing;
- retain strict whole-object JSON parsing and never strip Markdown fences;
- narrow the route wire format and construct display-only fields
  deterministically;
- keep an allowlisted clarification/null outcome so grammar constraints cannot
  force an unsafe action; and
- measure raw model routing separately from guarded end-to-end behavior.

The implementation follows those recommendations. It retains the existing one
bounded non-executing repair because other providers and historical curriculum
use that stage; every repaired result still passes the same strict schema and
allowlist validation.

## Implemented contract

Production route output contains exactly one model-controlled field:

```json
{"proposed_action":"create_transfer_plan_from_candidate"}
```

The field is constrained to `none` plus actions allowed in the current state.
Code supplies the display message, intent, empty reason, and server-approved
suggestions. Tool arguments remain a separate stage; candidate transfers and
their missing-field clarifications are assembled from trusted typed facts
without another model call. Historical v4 five-field route prompts remain
available only for reproducing old training artifacts.

## Repeated development pilot

The same 12 cases were rerun using local Ollama `gemma4:e2b`:

- final structured responses: 12/12;
- raw route accuracy: 7/12;
- guarded end-to-end accuracy: 12/12;
- hazardous cases contained: 6/6;
- first-attempt structured validity: 12/12, with zero repairs;
- inference errors: 0; and
- total measured time: 9.08 seconds.

The five raw misses all over-selected the candidate-transfer action when a
recipient, exact base-unit amount, or matching chain was absent or ambiguous.
The deterministic gate converted them to typed clarification. This remaining
semantic weakness is a direct v5 fine-tuning target and prevents treating the
small pilot as a release or generalization result.

Claude's post-implementation review found no security regression and required
that capability and containment remain separate metrics. It also identified
checkpoint-to-contract startup binding and larger independently authored
evaluation as high-value follow-ups. Its concern about clarification opening an
unrequested transfer flow does not apply to these five misses: every case was
an explicit but incomplete transfer request. The evaluator now records
first-attempt validity and repair counts so repaired output cannot be hidden
inside an aggregate validity number.
