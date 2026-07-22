# ADR 0002: Model proposals plus deterministic enforcement

Status: accepted

The model interprets requests and proposes one narrow typed tool call. A shared,
versioned tool contract constrains names and arguments where the runtime permits.
Deterministic code independently validates registry identities, arithmetic,
workflow state, simulation output, policy, and approval freshness.

Model output is never authorization and free-form text is never executed. Live
facts do not come from model weights. Untrusted retrieved text stays inside the
typed `untrusted_data` field and cannot populate actionable fields.
