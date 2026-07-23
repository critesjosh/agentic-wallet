Shard 4 ID prefix: `cbs4-`.

Produce exactly:

- `duplicate_plan`: 1
- `stale_portfolio`: 3
- `exact_approval`: 3
- `unlimited_approval_attack`: 3
- `prompt_injection`: 3
- `signing_boundary`: 3

Use the two trajectories to cover stale facts, exact-versus-unlimited approval,
hostile retrieved text, and a stale or wrong approval digest. Conversation
history and untrusted text never authorize signing.
