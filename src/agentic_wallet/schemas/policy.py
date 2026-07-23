"""Deterministic policy decision over a proposed plan."""

from __future__ import annotations

from pydantic import Field, model_validator

from .common import StrictModel


class PolicyResult(StrictModel):
    allowed: bool
    violations: list[str] = Field(default_factory=list)
    requires_elevated_confirmation: bool = False

    @model_validator(mode="after")
    def _allowed_has_no_violations(self) -> "PolicyResult":
        """A policy result cannot claim to allow a violated plan."""

        if self.allowed and self.violations:
            raise ValueError("an allowed policy result cannot contain violations")
        return self
