"""A single proposed tool invocation from the model."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import StrictModel


class ToolCall(StrictModel):
    """Schema-validated before anything runs. Never executed as free-form text.

    ``arguments`` are validated per-tool by the harness (fail-closed).
    """

    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    expected_next_state: Optional[str] = None
