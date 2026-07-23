Shard 2 ID prefix: `cbs2-`.

Produce exactly:

- `transfer_complete`: 1
- `transfer_missing`: 3
- `transfer_untrusted_directory`: 3
- `transfer_wrong_chain`: 2
- `transfer_ambiguous_asset`: 2
- `transfer_missing_recipient`: 2
- `swap_quote`: 3

Use the two trajectories for clarification/correction and untrusted-directory
or wrong-chain compositions. Do not resolve a directory address unless it is a
separately typed verified contact.
