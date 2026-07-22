# ADR 0003: No autonomous signing

Status: accepted

The wallet, not the model or agent harness, owns signing authority. A future
signer may accept only a single-use capability bound to the exact C1 approval
digest, account, chain, transaction preimage, state anchor, nonce, and expiry.
Any mutation or drift forces re-simulation and fresh user approval.

The current web application has no key custody, signer, submission, or receipt
monitor. Signing schemas are Phase 8 preparation, not an enabled capability.
