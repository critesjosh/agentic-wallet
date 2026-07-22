"""The swappable inference seam (plan.md web-first sequencing).

Remote HTTP now (the untuned target model served remotely), on-device later.
Every provider must return schema-valid, fail-closed output: invalid output or
an action not available in the current state is rejected, never executed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas.tool_call import ToolCall
from .schemas.dialogue import ModelDialogueTurn


class InferenceError(RuntimeError):
    """Raised when a provider returns unusable or disallowed output."""


class InferenceProvider(ABC):
    name: str = "abstract"
    native_constrained_decoding: bool = False

    @abstractmethod
    def propose_tool_call(self, context: dict, available_actions: list[str]) -> ToolCall:
        """Return a validated ToolCall for the given context and state."""

    def _validate(self, raw: dict, available_actions: list[str]) -> ToolCall:
        from .tool_contract import validate_tool_arguments

        try:
            tc = ToolCall.model_validate(raw)
        except Exception as exc:  # fail-closed on any schema violation
            raise InferenceError(f"invalid tool-call schema: {exc}") from exc
        if tc.action not in available_actions:
            raise InferenceError(
                f"action {tc.action!r} not available in this state"
            )
        validate_tool_arguments(tc.action, tc.arguments)
        return tc

    def propose_dialogue_turn(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> ModelDialogueTurn:
        """Compatibility path; concrete model providers return dialogue JSON."""

        if not available_actions:
            return ModelDialogueTurn(
                message="Here is the verified tool result.",
                intent="conversation",
            )
        call = self.propose_tool_call(context, available_actions)
        return ModelDialogueTurn(
            message="I will use a validated read-only tool for that request.",
            intent="propose_tool",
            proposed_action=call,
        )


class ScriptedProvider(InferenceProvider):
    """Deterministic provider for tests and benchmarks. Maps a scenario key to a
    raw tool-call dict, standing in for the target model so the harness and
    benchmark run without a live model.
    """

    name = "scripted"

    def __init__(self, script: dict[str, dict]) -> None:
        self._script = script

    def propose_tool_call(self, context: dict, available_actions: list[str]) -> ToolCall:
        key = context.get("scenario_id")
        if key not in self._script:
            raise InferenceError(f"no scripted response for {key!r}")
        return self._validate(self._script[key], available_actions)


class RemoteHTTPProvider(InferenceProvider):
    """Points at the untuned target model (Gemma 4 E2B/E4B) served remotely,
    e.g. Ollama or an OpenAI-compatible endpoint. The network call is left
    unimplemented in the scaffold; wire it to the demo endpoint, keeping the
    same schema and grammar enforcement the on-device provider will use so the
    two pass a shared conformance suite (plan.md web-first sequencing).
    """

    name = "remote-http"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def propose_tool_call(self, context: dict, available_actions: list[str]) -> ToolCall:
        raise NotImplementedError("wire to the remote target-model endpoint")
