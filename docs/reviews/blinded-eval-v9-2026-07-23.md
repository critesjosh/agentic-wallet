# Blinded evaluation v9 review

Claude Sonnet reviewed a design-only summary through the directly authenticated
Claude Code CLI with session persistence disabled. It received no repository
training/development plaintext and did not inspect sealed cases. No Claude model
was called through OpenRouter.

The review identified three methodological limits:

- 64 cases have wide uncertainty for small capability differences.
- Deterministic compiler behavior defines the scoring oracle, so compiler
  correctness must be tested independently.
- Repository disjointness cannot prove absence from unknown pretraining data.

It also asked whether committed artifacts are revalidated at execution time and
whether the custodian could inspect plaintext after claiming. The evaluator
does re-hash the live suite, selected adapter, evaluator, and complete harness
against the commitment. The suite digest already binds the deterministically
compiled author sources. Custodian access cannot be eliminated by a local
script, so v9 explicitly remains model-authored, operator-mediated evidence
with `release_claim_eligible=false`.

The review did not identify a reason to expose case-level outputs, weaken the
irreversible claim, or reuse a failed suite.
