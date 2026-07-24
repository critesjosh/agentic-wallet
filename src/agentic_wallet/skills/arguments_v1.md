---
name: wallet-arguments
description: Emit exact JSON arguments matching the tool schema; identifier and amount formats for wallet tool calls.
---
Emit only the exact JSON arguments the schema requires, all fields present, no
extras. Copy identifiers verbatim from the input; never invent one.

- asset and spender ids look like base:usdc or base:aerodrome-router.
- amounts are strings: whole base units like "1000000", or a decimal like "2.5".
- max_slippage_bps and chain_id are integers, not strings.
- a plan digest is exactly "sha256:" followed by 64 hex characters.
- swaps need input_asset_id and output_asset_id, both present and different.
