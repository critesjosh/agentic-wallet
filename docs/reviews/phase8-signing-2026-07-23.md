# Phase 8 signing and transaction-boundary review

Date: 2026-07-23

## Review path

Claude Code was invoked locally with a read-only security-review prompt. It
returned no review because the configured Claude account had reached its
monthly spend limit. No Claude model was called through OpenRouter.

Under the pre-agreed fallback, a fresh Terra subagent inspected the uncommitted
Phase 8 diff, Consensus revisions, v6 curriculum/evaluator/result, and relevant
tests. It made no file changes and reported 84 focused tests passing.

## Findings and disposition

### Release gates remain closed

The reviewer correctly classified the v6 result as non-release evidence:
seven development safety failures, 0/12 multi-argument exact results, and no
valid independently sealed C3 evaluation. The implementation and project page
remain explicitly labelled as a direct-loopback, single-user Base-native
proof—not a safe-wallet or release-ready signing feature.

Disposition: documented in `README.md`, `docs/fine-tuning.md`, and
`docs/model-failures.md`. No sealed-suite revision or release claim was made.

### Browser approval is application-attested

The isolated signer verifies an exact, expiring, single-use HMAC capability, but
the local web process mints it after recording the button click. The signer
cannot independently observe user presence, so a compromised web process is
inside this POC's trust boundary.

Disposition: the signing ADR and README now state this directly. Production
requires hardware-, wallet-, or OS-owned confirmation that the application
cannot forge. The current mechanism remains useful for proving that model text,
chat history, remote requests, envelope mutation, and replay cannot authorize a
different transaction.

### Receipt lookup needed a fresh chain check

The saved-hash status path validated receipt shape and hash but did not recheck
the configured RPC chain immediately before lookup.

Disposition: fixed. `TransactionController.transaction_status()` now requires
the expected chain and compares it with the saved record before accepting a
receipt. An adversarial chain-switch test confirms the saved state remains
unchanged.

### Loopback is not proxy authentication

Client and Host loopback checks are valid only for the documented direct server
launch. A reverse proxy can otherwise make a remote client appear local.

Disposition: common forwarding headers now disable the transaction surface,
with an HTTP test. Documentation prohibits proxy deployment in signing mode and
states that header rejection is defense-in-depth, not authentication.

### RPC metadata is externally visible

The configured RPC observes wallet and transaction metadata even when model
inference is local.

Disposition: the ADR and README now enumerate the disclosure, require a fixed
HTTPS endpoint outside loopback, recommend trusted/self-hosted infrastructure,
and avoid making provider-retention claims.

### Cross-session transaction-hash collision

During the local follow-up audit, the application store was found to return an
existing record solely by transaction hash. A duplicate hash submitted from a
different session could therefore return the first session's record.

Disposition: fixed. Duplicate insertion is idempotent only when the session,
workflow, plan/envelope digests, chain, sender, and signing hash all match.
Cross-session or cross-workflow rebinding fails closed and has a focused test.

### Canonical snapshot and lost-response recovery

The completion audit also tightened two reachable post-review cases. Recipient
code and `eth_call` now use the captured block hash through EIP-1898 with
canonical ancestry required, and that hash is part of the approval envelope.
The signer now journals safe local hashes before broadcast. If the stdio result
is lost, deterministic application code queries the journal for the same
envelope and persists the recovered outcome without minting another capability
or signing again.

### Application-index failure after signing

The final re-review found that a validated terminal signer result could be
hidden behind a generic error if the bounded in-memory transaction index failed.
This was not a replay path—the workflow was already terminal and the signer
journal retained the outcome—but it deprived the user of a known hash.

Disposition: fixed. The response now returns the validated transaction hash,
code-generated explorer link, terminal status, and
`storage_error_code: APP_STATE_RECORD_FAILED`, with `app_state_saved: false`.
The UI tells the user to retain the hash because later in-app lookup may be
unavailable. Parameterized tests cover both `SUBMITTED` and `UNKNOWN` outcomes
and prove neither can be resubmitted.

## Assessment

The final independent Terra re-review approved the code for publication as the
clearly labelled bounded POC with no remaining release-blocking security
finding. It is not
approved for real-funds encouragement or a production-wallet claim. The
cryptographic and deterministic model boundary is substantive; the remaining
release blockers are model quality, independent evaluation, real-device
evidence, durable/authenticated product state, RPC privacy choices, and
signer-owned user presence.

After all review dispositions and follow-up hardening, the complete local suite
collects and passes 395 tests.
