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
- Checkpoint semantic evaluation now covers the complete 60-record v4
  validation partition by default rather than an order-biased first 32 records.
- The sealed suite remains unavailable and will not be used or represented as
  completed. The next run is research-only development evidence.

No reviewer finding justified weakening validation, increasing retry count, or
moving approval/signing authority into model-controlled state.
