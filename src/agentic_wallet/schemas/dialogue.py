"""Structured separation between display-only conversation and tool proposals."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import StrictModel
from .tool_call import ToolCall

DialogueIntent = Literal[
    "conversation", "offer_action", "propose_tool", "clarify", "refuse"
]


class ModelDialogueTurn(StrictModel):
    """Model output. ``message`` is display-only and never parsed as a command."""

    message: str = Field(min_length=1, max_length=2_000)
    intent: DialogueIntent
    proposed_action: ToolCall | None = None
    suggested_actions: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _consistent_and_unique(self) -> "ModelDialogueTurn":
        if len(set(self.suggested_actions)) != len(self.suggested_actions):
            raise ValueError("suggested actions must be unique")
        if self.intent == "propose_tool" and self.proposed_action is None:
            raise ValueError("propose_tool intent requires proposed_action")
        if self.proposed_action is not None and self.intent != "propose_tool":
            raise ValueError("a proposed action requires propose_tool intent")
        return self


class SuggestedAction(StrictModel):
    """Server-owned presentation for a safe suggestion chip."""

    action: str
    label: str
    prompt: str


class DialogueWireTurn(StrictModel):
    """Flat constrained-decoding shape normalized into ``ModelDialogueTurn``."""

    message: str = Field(min_length=1, max_length=2_000)
    intent: DialogueIntent
    proposed_action: str
    arguments: dict = Field(default_factory=dict)
    reason: str = ""
    suggested_actions: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _unique_suggestions(self) -> "DialogueWireTurn":
        if len(set(self.suggested_actions)) != len(self.suggested_actions):
            raise ValueError("suggested actions must be unique")
        return self


class DialogueRoute(StrictModel):
    """First-stage conversational routing without any tool arguments."""

    message: str = Field(min_length=1, max_length=2_000)
    intent: DialogueIntent
    proposed_action: str | None = None
    reason: str = ""
    suggested_actions: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _consistent_route(self) -> "DialogueRoute":
        if len(set(self.suggested_actions)) != len(self.suggested_actions):
            raise ValueError("suggested actions must be unique")
        if self.intent == "propose_tool" and self.proposed_action is None:
            raise ValueError("propose_tool intent requires proposed_action")
        if self.proposed_action is not None and self.intent != "propose_tool":
            raise ValueError("a proposed action requires propose_tool intent")
        return self
