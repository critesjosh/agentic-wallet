# Conversation pipeline review — 2026-07-22

## Scope

The implementation was reviewed against the Consensus revisions and the five
non-training priorities: constrained decoding, two-stage route/argument calls,
bounded validation repair, typed conversation memory, and grounded post-tool
narration. The external reviewers received a sanitized architecture, aggregate
test, dataset, and model-failure summary only. They did not receive repository
files, `.env`, credentials, wallet data, or model weights.

## Claude Sonnet 4.6

Claude judged the change ready for a bounded research run, not release. Its
main requests were to diagnose the live E2B route-schema failure, verify that
the expanded train/validation records remain separated at their source-record
boundary, exercise the deterministic narration fallback, and label every
result as development/regression evidence while the sealed suite is absent.

## GLM 5.2

GLM likewise judged it ready only for a bounded research run. It emphasized
asserting that native schemas are actually attached to requests, testing that
signing and unlimited-approval choices receive no repair attempt, testing
negative narration claims, and retaining the wallet as the sole signer.

## Resolution

- Provider-conformance tests compare the exact route schema attached to Ollama,
  llama.cpp, and OpenRouter requests.
- The local Ollama E2B output was captured: it returned the same Markdown-fenced
  partial route on the initial call and its single repair. This is recorded in
  `docs/model-failures.md`; both outputs failed closed.
- Tests prove signing-boundary routes are not retried, transport errors are not
  retried, malformed safe routes get only one retry, and invented narration
  facts fall back to deterministic text.
- The v4 dataset test verifies that user requests in its source-preserving train
  and validation partitions are disjoint.
- Checkpoint semantic evaluation uses a deterministic 20-record round-robin
  across all five runtime phases rather than an order-biased prefix. The final
  adapter is still evaluated against the complete 60-record v4 validation set.
- The sealed suite remains unavailable and will not be used or represented as
  completed. The next run is research-only development evidence.

No reviewer finding justified weakening validation, increasing retry count, or
moving approval/signing authority into model-controlled state.

## Post-training methods review

Claude and GLM 5.2 reviewed a second sanitized packet containing only aggregate
v4 results and the proposed staged benchmark correction. Both agreed that
rerunning fixed adapter weights through a corrected production-matching
evaluator is legitimate protocol-sensitivity testing, provided both protocols
remain visible and the rerun is not described as model improvement.

Both emphasized that 1/14 multi-argument accuracy and the missing-recipient
hard-zero failure are core capability and safety gaps. They also requested
explicit small-sample uncertainty, an explanation of zero trajectory accuracy,
and disclosure that checkpoint selection uses part of the same 60-record
development partition. These limitations are now recorded in the fine-tuning
report and the frozen `staged-dialogue-route-v2.1` protocol. No release,
generalization, execution-safety, or model-selection claim is made.

The final fixed-weight staged-v2.1 rerun scored 3/29 exact with 11 critical
routes. This resolves the reviewers' methodological question but confirms their
substantive concern: the adapter has neither robust routing generalization nor
multi-argument capability. The hard-zero release failure is already decisive
under the project plan.

## Claude Fable 5 follow-up

Claude Fable 5 reviewed a third sanitized packet after the v4 development run
and the v2.1 scorer correction. It received the architecture and aggregate
metrics only, not source files, credentials, wallet data, or model weights.

The review agreed that the authority split is sound, but identified argument
composition as the immediate bottleneck: v4 scored 30/32 on zero-argument and
13/14 on single-argument records, but only 1/14 on multi-argument records. Its
highest-priority recommendations were deterministic rather than weight-based:

1. bind recipients and assets to trusted candidate IDs instead of accepting
   model-generated literal addresses;
2. force clarification before argument generation whenever required facts are
   absent;
3. compare one-pass joint argument generation with constrained per-field
   extraction;
4. keep unit conversion, checksum validation, name resolution, and other
   canonicalization in deterministic code; and
5. carry trajectory state as typed facts rather than asking the model to
   reconstruct it from prose.

The review also highlighted indirect prompt injection through attacker-chosen
token names, ENS labels, or memos; address-poisoning in candidate lists;
omission of required warnings from otherwise factually grounded narration;
fact staleness between reads and approval; and the need to report fail-closed
rejection rate separately from exact accuracy.

These findings strengthen the existing release gates. Before transaction
workflows can be treated as usable, the project needs deterministic recipient
binding, a required-field clarification gate, typed snapshot/freshness binding,
materially better multi-argument and trajectory results, and a once-only sealed
evaluation. The model-independent confirmation view must continue to render
decoded transaction and simulation facts rather than model narration.

The next bounded experiments are:

- candidate-ID binding: no free-generated recipient address and no reachable
  missing-recipient transaction proposal;
- joint versus per-field extraction on at least 50 multi-argument development
  cases, targeting at least 80% exact with zero validated dangerous arguments;
- adversarial typed-fact tests covering embedded instructions, confusables,
  provenance, and narration omissions, with zero route or narration compromise;
- trajectory evaluation with deterministic state carriage and forced
  clarification, targeting at least 70% complete trajectories and zero
  critical failures; and
- an independently authored, single-use sealed suite of at least 100 cases,
  targeting at least 90% schema validity and zero hard-zero failures before any
  transaction-readiness claim.

## Candidate-binding implementation review

Claude Fable 5 reviewed a sanitized description of the resulting candidate
implementation. It identified three pre-commit issues: positional IDs could be
rebound across turns, the historical literal-recipient action needed an
explicit production rejection rather than a documentation convention, and bare
human-unit amounts could be confused with base units.

All three were addressed. Recipient IDs now commit to the current request and
address, with a cross-turn rebinding test. A production action-set validator and
web-dispatch test reject the legacy action. The amount parser accepts only an
integer explicitly labeled as base units or wei; bare human-unit and ambiguous
amounts force clarification. Conflicting explicit chains also force
clarification rather than being silently replaced by the wallet chain.

The later recommendations remain open: contact-name confusable defenses,
single-use candidate snapshot expiry, multi-recipient UX, and telemetry for
clarification and legacy-action rejection. Replay protection for approved
transactions remains owned by the existing digest, nonce, expiry, and state
anchor checks rather than the model-facing candidate ID.
