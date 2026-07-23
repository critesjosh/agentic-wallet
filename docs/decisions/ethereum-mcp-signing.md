# Ethereum MCP signing boundary

Status: accepted for the first Phase 8 proof of concept.

## Scope

The first live path supports only a native ETH EIP-1559 transfer on Base to an
externally owned account. ERC-20 transfers, approvals, swaps, contract
recipients, zero/self transfers, arbitrary calldata, automatic repricing,
replacement transactions, other chains, and delegated authority remain
disabled.

The model may propose a transfer intent. Deterministic application code resolves
trusted facts, assembles the complete transaction preimage, simulates and checks
it, evaluates policy, and renders the immutable approval envelope. The model
never receives an approval, signing, submission, raw-transaction, RPC, or key
tool.

## Approval and signing sequence

1. A pinned RPC client reads the configured chain, signer address, pending
   nonce, balance, fee data, gas estimate, and relevant account state.
2. Deterministic code creates the complete EIP-1559 preimage, requires an
   empty-code recipient at the captured canonical block hash, and runs an
   EIP-1898 `eth_call` preflight pinned to that hash with canonical ancestry
   required. For this narrow native transfer it constructs and checks the
   expected value-plus-maximum-fee delta; this is not represented as a general
   trace/state-diff simulator. The envelope binds the plan, preimage, normalized
   check, policy, registry digest, snapshot block hash, relevant-state anchor,
   nonce, and expiry.
3. The web UI renders those immutable fields. Chat text, model output,
   suggestions, and transcript history cannot approve them.
4. A dedicated approval endpoint accepts only the exact displayed digest and
   moves `AWAITING_CONFIRMATION` to `READY_TO_SIGN`.
5. A separate submit operation re-reads relevant account state. Sender balance
   or pending-nonce drift clears approval and forces re-simulation. The captured
   block hash remains approval-bound snapshot provenance but does not invalidate an
   otherwise unchanged account merely because a new block arrived; the signer
   independently repeats the exact call and gas checks against current state.
6. Only deterministic application code invokes the private stdio MCP signer.
   It passes the immutable envelope and a short-lived application assertion
   created only after the web workflow records explicit approval.
7. The signer independently verifies the capability, envelope, expiry, chain,
   signer, pending nonce, relevant-state anchor, simulation, policy, and exact
   transaction binding. It reads the private key from an OS secure-store
   backend, signs that preimage, verifies the recovered sender, broadcasts only
   the resulting raw transaction through its configured RPC endpoint, and
   requires the returned hash to equal the locally computed hash.
8. Before broadcast, the signer fsyncs a secret-free `UNKNOWN` outcome with the
   locally computed hashes. A separate internal lookup recovers that exact
   outcome if the stdio response is lost; recovery never signs again.
9. The MCP result contains only a typed outcome, transaction/signing hashes
   when known, sender, and envelope digest—never the private key or raw signed
   transaction. Freshness rejection returns to `SIMULATING`. Post-sign broadcast
   ambiguity retains the local hash as `UNKNOWN` and has no retry edge.

## Key custody

The MCP signer is a dedicated child process using stdio, not HTTP or SSE. It is
not registered in the model tool vocabulary. Provisioning reads a key
interactively from a TTY and stores it in an OS keyring under fixed
application/account identifiers. Provisioning and signing refuse fail, null,
plaintext, or otherwise unavailable keyring backends; there is no file or
environment-variable private-key fallback.

The current development shell reports `keyring.backends.fail.Keyring`, so real
key provisioning and live signing must remain disabled here until an OS secure
store is available. Tests use injected fake stores and never persist a real
key.

The web transaction endpoints are loopback-only. Cookie sessions and CSRF
tokens prevent cross-site action but are not user authentication; remote live
signing remains disabled until a real authentication boundary is designed. The
direct server rejects common proxy-forwarding headers, and signing mode must not
be placed behind a reverse proxy. This is defense-in-depth, not proxy-aware
authentication: an intermediary that strips those headers is outside the POC's
threat model. The
application advertises signing only after the secure backend, provisioned signer
address, and pinned Base RPC pass readiness. The capability HMAC key is supplied
as URL-safe base64 decoding to at least 32 random bytes; it is not a wallet key.
Each signer process atomically claims a hash-only capability record in an
owner-only XDG state directory before key access. Incomplete or corrupt claim
records remain fail-closed tombstones, preventing concurrent replay.

## Trust boundary and user-presence limitation

The signer independently proves that a capability was minted by this configured
application for the exact envelope; it cannot independently observe the browser
click. The local web process holds the HMAC key and is therefore trusted to mint
the assertion only after its in-memory workflow records the matching digest.
The split prevents model text, chat history, a remote client, or a replayed MCP
call from authorizing a different transaction. It does **not** protect against a
compromised local web process.

Accordingly, this is a single-user, direct-loopback boundary proof, not a
production wallet authorization design. A production signer must require a
signer- or wallet-owned user-presence signal—such as hardware-wallet
confirmation or OS-mediated authentication—that the application process cannot
forge. The HMAC assertion must not be described as independent proof of user
presence.

## RPC privacy boundary

Local model inference does not make live transactions local. The configured RPC
provider observes the signer address, chain and account-state reads, recipient,
amount-bearing preflight, gas estimation, receipt lookups, and the raw signed
transaction at broadcast. Provider logs can therefore correlate wallet activity
and intent. Operators choose the fixed endpoint; per-request/model-selected
endpoints and URL-embedded credentials are rejected, and non-loopback endpoints
must use HTTPS.

This POC does not verify provider retention or zero-logging claims. Privacy-
sensitive testing should use a self-hosted or otherwise trusted Base endpoint
and should treat its retention, jurisdiction, and access controls as separate
deployment decisions.

## Submission state

Submitted hashes are kept in a bounded, lock-protected application transaction
store, separate from chat history and approval state. Records contain only safe
metadata: session/workflow ID, plan and envelope digests, chain, sender,
transaction/signing hashes, status, timestamps, deterministic error code, and a
chain-aware explorer URL. They never retain a raw transaction, signature,
capability, approval object, RPC credential, or key.

The signer also keeps a separate owner-only XDG outcome journal keyed by the
envelope digest. It fsyncs a safe `UNKNOWN` record containing only the local
transaction/signing hashes before broadcast, then atomically promotes it to
`SUBMITTED` only after the RPC returns the matching hash. If the MCP response is
lost or malformed, the web process performs a lookup without minting another
capability or signing again. If both response and lookup are unavailable, the
workflow becomes terminal `SUBMISSION_UNKNOWN` without a hash and must not be
retried.

If the validated signer outcome is known but the bounded application index
fails, the terminal response still returns the code-generated explorer link and
hash with `app_state_saved: false`; the UI tells the user to retain the hash
because session lookup may be unavailable.

The signer separately keeps an owner-only XDG outcome journal keyed by envelope
digest. It stores only the same safe hashes and typed outcome, is durably
written as `UNKNOWN` before broadcast, and may be promoted to `SUBMITTED` only
when the returned RPC hash matches the locally computed hash. Deterministic
application code may query it only to recover the exact envelope already in
`SUBMITTING`; it is not a model-facing tool.

Explorer URLs come only from code-owned chain metadata and a validated
transaction hash. Unknown chains fail closed rather than accepting a model- or
client-provided URL. The application revalidates the configured chain before
trusting a refreshed receipt status.

## Failure behavior

Expiry, nonce or relevant-state drift, policy/registry/preimage mutation,
signer mismatch, insecure key storage, RPC chain mismatch, signing failure, RPC
error, or returned-hash mismatch prevents a successful submission. The signer
never silently changes nonce, fees, gas, recipient, value, calldata, or chain.
Ambiguous post-broadcast timeouts or returned-hash mismatches preserve the
locally computed hash as `UNKNOWN`, return its trusted Basescan link, and do not
trigger automatic re-signing. A session-scoped `get_transaction_status` route
can look up only an exact current-message hash already owned by that browser
session.
