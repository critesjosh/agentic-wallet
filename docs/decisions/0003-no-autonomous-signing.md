# ADR 0003: No autonomous signing

Status: accepted

The model has no signing authority. The Phase 8 signer holds the cryptographic
key and accepts only a single-use application assertion bound to the exact C1
approval digest, account, chain, transaction preimage, state anchor, nonce, and
expiry. Any mutation or drift forces re-simulation and fresh browser approval.

The signer is a private stdio MCP child process, never a model tool. It loads a
key only from an approved OS keyring and returns safe metadata without raw
signed bytes. The optional Base-native-transfer POC is disabled by default and
loopback-only until real remote-user authentication exists. It does not grant
autonomous or delegated authority.

This local POC trusts the web process to attest that the separate approval
button was used; the signer cannot independently observe user presence. A
production wallet must replace that shared-application assertion with
signer- or wallet-owned confirmation that the application cannot forge.
