# ADR 0001: Local-first model inference

Status: accepted

The product target is on-device model inference so wallet context and user
intent do not normally go to a hosted model. Live chain state, quotes, prices,
and simulation still come from typed external providers; "local-first" applies
to inference, not to inherently remote blockchain facts.

The web POC may use a server-hosted target checkpoint, but it must disclose the
actual endpoint location and require per-message consent when that endpoint is
not loopback. Remote and on-device artifacts remain separate conformance targets.
