# Blinded evaluation v14 review

## Review routes

- Claude Code 2.1.217 was invoked locally with Claude Sonnet and instructed to
  review only the repository protocol. It was explicitly prohibited from
  reading external blinded plaintext. The request reached Claude Code but
  stopped at the account's monthly spend limit before producing findings.
  No Claude model was called through OpenRouter.
- A fresh `gpt-5.6-terra` subagent performed three read-only passes as the
  user-authorized fallback. It did not author evaluation cases or inspect
  external blinded plaintext.

## Findings addressed

- Added the actual v14 seed compiler to the frozen harness digest and protected
  tracked paths.
- Made the four turns in each trajectory share one deterministic fictional
  world, use canonical turn order, and carry bounded teacher-forced prior
  messages and proposals.
- Enforced an exact, versioned scenario sequence for every trajectory instead
  of relying on prompt-only coherence.
- Removed code-generated refusal and expected-decision cues from adversarial
  user requests.
- Strictly validate every supplied conversation ledger before inference and
  fail closed on extra authorization-shaped fields.
- Construct seeded ledgers through the same strict schema and leave ambiguous
  or conflicting intent fields unresolved.
- Reject whitespace-only utterances, global world-seed collisions, and overlap
  between an independent world seed and the trajectory's persistent world key.
- Aligned frozen metadata with the actual no-repository, no-author-validator
  procedure.

## Interpretation

V14 is a seed-expanded, model-authored, operator-mediated evaluation. Its
multi-turn metric uses teacher-forced typed context: all four turns must pass,
but candidate output does not mutate the next case. It is useful experimental
evidence, not independent-human evidence, a release gate, or proof that
hard-zero failures have zero probability.
