# Terra blinded-author procedure v3

V11 was retired after both whole-suite attempts failed deterministic
compilation without candidate evaluation or case-level feedback.

For v12 the root spawns isolated `gpt-5.6-terra` authors with
`fork_turns="none"`. Each receives only the v3 shared/shard prompts, one external
output path, and one exact invocation of
`scripts/validate_blinded_author_shard.py`. The author may execute that validator
but is instructed not to inspect any other repository path.

The validator uses only the public deterministic scenario compiler. It reads no
training/development data, never loads the candidate, emits only
`valid`/`case_count`, and suppresses compiler details. An author may regenerate
its entire shard at most twice after an invalid result. Root receives only the
final aggregate status and never receives repair details or case content.

All eight final source files are still independently recompiled, quota-checked,
disjointness-audited, and digest-bound by the root. A whole-suite failure is
never repaired after submission.
