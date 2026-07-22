# Sealed evaluation protocol

Status: **blocking another quality-training GPU run; no independently human-authored
sealed suite has been supplied**.

The 29-case benchmark is development regression data because v2 scenarios were
chosen after inspecting its failures. It remains useful for debugging, but it
cannot support a generalization claim.

## Independence and custody

- A person who has not read the SFT generator or its prompts authors at least 20
  cases using a different vocabulary, composition protocol, assets, recipients,
  spenders, amounts, and conversation phrasing.
- Store plaintext outside the training checkout. Commit only a SHA-256 digest,
  case count, rubric version, author role, and creation timestamp before
  training starts.
- Training, prompt construction, and checkpoint selection must not read the
  plaintext suite.
- Evaluate a candidate once. Only aggregate metrics and hard-zero status may be
  used for a release gate. If individual outputs are opened for debugging, that
  suite version is retired and replaced before the next claim.

## Minimum coverage

- Five multi-argument construction cases with unseen values and compositions.
- Four multi-turn trajectories covering corrections, cancellation, quote
  expiry, duplicate requests, and state changes between turns.
- Two digest-bound confirmation/signing-boundary cases.
- Two prompt-injection cases with distinct untrusted sources.
- Two simulation mismatches, including an unexpected asset debit.
- One exact-versus-unlimited approval case.
- One stale-portfolio refresh case.
- Grounded explanations for balances, allowances, quotes, and simulations.
- Natural greetings, vague speech, typos, mixed explanation/action intent, and
  unsupported requests with safe alternatives.

## Reporting

Run the same checkpoint with and without the target runtime's constrained
decoding. Report schema validity, action and argument accuracy, zero-argument
and multi-argument results separately, sequence/state accuracy, unsupported
fact claims, and every hard-zero category. Any hard-zero failure blocks release.

The suite does not replace physical-device P2 measurements: cold load, memory,
latency, sustained throughput, thermals, battery, crashes, and constrained
decoding on the real target remain separate gates.

Use [sealed-eval-author-packet.md](sealed-eval-author-packet.md) for the
standalone human handoff. `scripts/commit_sealed_suite.py` refuses plaintext
inside the checkout and records only digest metadata. Workflow-v3 training also
refuses execution until `data/benchmark/sealed-suite-v1.commitment.json` has a
valid pre-training digest and independence attestation. After development-only
checkpoint selection, `scripts/evaluate_sealed.py` verifies that commitment and
emits aggregate metrics without persisting case-level outputs.
