# Candidate-bound transfer proposals

Status: implemented for the next transaction-pipeline iteration; not exposed by
the read-only web demo.

The v4 model failed joint multi-argument construction and sometimes selected a
transfer when the recipient was missing. The replacement production action is
`create_transfer_plan_from_candidate`. It never accepts a literal address. Its
model-facing recipient field is an opaque `recipient:*` ID, and deterministic
code owns the mapping to a checksum-validated EVM address.

Any route exposing this action declares
`wallet-tool-call-v3-candidate-binding`. Historical actions continue to render
the v2 prompt, preserving the frozen staged-v2.1 benchmark protocol.

Only an address explicitly present in the current user message or supplied by
a separately verified contact source can become a candidate. Retrieved text,
token names, memos, and transaction history are not candidate sources. Duplicate
addresses are deduplicated; zero or multiple candidates force clarification.
IDs contain a short commitment to the current request and address rather than a
mutable positional index. Reusing an ID with a changed turn or address therefore
fails closed during binding; unknown or ambiguous IDs do as well.

For a complete transfer request, deterministic code also resolves one canonical
asset ID, an exact integer base-unit amount, and the wallet chain. It constructs
the selected action without an argument-generation model call. The bound result
then enters the existing unsigned planner, simulation, policy, and digest-bound
approval flow. The wallet remains the sole signing authority.

Bare human-unit amounts such as `5 USDC` are not interpreted as five base units.
Until deterministic decimal scaling is added, only an explicitly labeled
integer base-unit amount is accepted; other forms force clarification. An
explicit chain that conflicts with the wallet chain is likewise never silently
rewritten.

The literal-recipient `create_transfer_plan` action remains in the registry only
so historical v1-v4 datasets and immutable regression artifacts can still be
loaded and interpreted. The production action-set validator explicitly rejects
that legacy action (as well as benchmark-only unlimited approval and signing
actions), and the web dispatcher rejects it as unavailable. Production state
allowlists may expose only the candidate-bound action. V5 training data removes
the obsolete transfer argument and repair targets rather than rewriting v4
history.
