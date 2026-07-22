"""The swappable inference seam (plan.md web-first sequencing).

Remote HTTP now (the untuned target model served remotely), on-device later.
Every provider must return schema-valid, fail-closed output: invalid output or
an action not available in the current state is rejected, never executed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas.tool_call import ToolCall
from .schemas.dialogue import DialogueRoute, ModelDialogueTurn

_NON_REPAIRABLE_ACTIONS = frozenset(
    {"proceed_to_signing", "create_unlimited_approval_plan"}
)


class InferenceError(RuntimeError):
    """Raised when a provider returns unusable or disallowed output."""


class ProposalValidationError(InferenceError):
    """A structured completion was returned but failed the wallet contract."""


class InferenceProvider(ABC):
    name: str = "abstract"
    native_constrained_decoding: bool = False
    last_raw_output: dict | None = None
    last_attempt_count: int = 0

    @abstractmethod
    def propose_tool_call(self, context: dict, available_actions: list[str]) -> ToolCall:
        """Return a validated ToolCall for the given context and state."""

    def _validate(self, raw: dict, available_actions: list[str]) -> ToolCall:
        from .tool_contract import validate_tool_arguments

        try:
            tc = ToolCall.model_validate(raw)
        except Exception as exc:  # fail-closed on any schema violation
            raise ProposalValidationError(f"invalid tool-call schema: {exc}") from exc
        if tc.action not in available_actions:
            raise ProposalValidationError(
                f"action {tc.action!r} not available in this state"
            )
        try:
            validate_tool_arguments(tc.action, tc.arguments)
        except InferenceError as exc:
            raise ProposalValidationError(str(exc)) from exc
        return tc

    def propose_tool_call_with_repair(
        self, context: dict, selected_action: str
    ) -> ToolCall:
        """Try once, then make at most one non-executing schema repair call."""

        self.last_attempt_count = 1
        try:
            return self.propose_tool_call(context, [selected_action])
        except ProposalValidationError as first_error:
            if selected_action in _NON_REPAIRABLE_ACTIONS:
                raise
            repair_context = {
                **context,
                "phase": "repair_tool_arguments",
                "selected_action": selected_action,
                "previous_output": self.last_raw_output,
                "validation_error": str(first_error)[:2_000],
                "repair_attempt": 1,
            }
            self.last_attempt_count = 2
            try:
                return self.propose_tool_call(repair_context, [selected_action])
            except InferenceError as second_error:
                raise InferenceError(
                    "tool proposal failed after one bounded repair attempt"
                ) from second_error

    def propose_dialogue_route(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> DialogueRoute:
        """Compatibility route for providers that still emit a combined turn."""

        turn = self.propose_dialogue_turn(
            context, available_actions, suggested_action_ids
        )
        return DialogueRoute(
            message=turn.message,
            intent=turn.intent,
            proposed_action=(
                turn.proposed_action.action if turn.proposed_action else None
            ),
            reason=(turn.proposed_action.reason if turn.proposed_action else ""),
            suggested_actions=turn.suggested_actions,
        )

    def propose_dialogue_route_with_repair(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> DialogueRoute:
        """Try routing once, then make one schema-only repair when safe."""

        self.last_attempt_count = 1
        try:
            return self.propose_dialogue_route(
                context, available_actions, suggested_action_ids
            )
        except ProposalValidationError as first_error:
            raw_action = (
                self.last_raw_output.get("proposed_action")
                if isinstance(self.last_raw_output, dict)
                else None
            )
            if raw_action in _NON_REPAIRABLE_ACTIONS:
                raise
            repair_context = {
                **context,
                "phase": "repair_dialogue_route",
                "previous_output": self.last_raw_output,
                "validation_error": str(first_error)[:2_000],
                "repair_attempt": 1,
            }
            self.last_attempt_count = 2
            try:
                return self.propose_dialogue_route(
                    repair_context, available_actions, suggested_action_ids
                )
            except InferenceError as second_error:
                raise InferenceError(
                    "dialogue route failed after one bounded repair attempt"
                ) from second_error

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
