"""Deterministic policy decision over a proposed plan."""

from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class PolicyResult(StrictModel):
    allowed: bool
    violations: list[str] = Field(default_factory=list)
    requires_elevated_confirmation: bool = False
