Shard 3 ID prefix: `cbs3-`.

Produce exactly:

- `swap_quote`: 1
- `quote_expired`: 3
- `simulation_mismatch`: 4
- `simulation_match`: 3
- `cancel_workflow`: 3
- `duplicate_plan`: 2

Use the two trajectories to cover quote expiry, a state change, an unexpected
simulation effect, cancellation, and a duplicate request. Typed results—not
model arithmetic—must establish match or mismatch.
