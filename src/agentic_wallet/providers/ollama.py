"""Schema-constrained inference through a local Ollama daemon."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..inference import InferenceError, InferenceProvider, ProposalValidationError
from ..schemas.tool_call import ToolCall
from ..schemas.dialogue import DialogueRoute, ModelDialogueTurn
from ..tool_contract import (
    dialogue_turn_json_schema,
    dialogue_turn_messages,
    tool_call_json_schema,
    tool_call_messages,
    dialogue_route_json_schema,
    dialogue_route_messages,
    validate_dialogue_route_decision,
    validate_dialogue_turn,
)

OllamaTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class OllamaIncompleteResponseError(ProposalValidationError):
    """Ollama closed or truncated a response before a normal final message."""


def _transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except HTTPError as exc:
        raise InferenceError(f"Ollama request failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise InferenceError(f"Ollama request failed: {type(exc).__name__}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError("Ollama returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise InferenceError("Ollama response must be a JSON object")
    return value


ollama_tool_call_schema = tool_call_json_schema


class OllamaProvider(InferenceProvider):
    """Use local Ollama structured outputs; the result remains only a proposal."""

    name = "ollama"
    native_constrained_decoding = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        keep_alive: str = "5m",
        transport: OllamaTransport | None = None,
        think: bool = False,
        skill: str | None = None,
        argument_skill: str | None = None,
    ) -> None:
        if not model.strip():
            raise InferenceError("an Ollama model is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        # Optional inference-time guidance, off unless a run opts in. Prepended
        # as a system message; never alters the frozen contract text. ``skill``
        # applies to routing; ``argument_skill`` to argument filling.
        self.skill = skill
        self.argument_skill = argument_skill
        # Reasoning is off by default. It was originally disabled because
        # thinking mode returned incomplete responses, and the done/done_reason
        # gates below still reject those. Enabling it is an evaluated choice:
        # hidden reasoning tokens are a large latency cost on the mobile target,
        # and the SFT corpus contains no reasoning traces.
        self.think = think
        self._transport = transport or _transport
        self.last_response_metadata: dict[str, Any] = {}

    @staticmethod
    def _asset_ids(context: dict) -> list[str] | None:
        value = context.get("canonical_asset_ids")
        return (
            value
            if isinstance(value, list) and all(isinstance(item, str) for item in value)
            else None
        )

    def _complete(
        self,
        schema: dict,
        messages: list[dict[str, str]],
        skill: str | None = None,
    ) -> dict:
        if skill:
            messages = [{"role": "system", "content": skill}, *messages]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "think": self.think,
            "options": {"temperature": 0, "seed": 0},
            "keep_alive": self.keep_alive,
        }
        response = self._transport(
            f"{self.base_url}/api/chat", payload, self.timeout
        )
        self.last_response_metadata = {
            key: response.get(key)
            for key in (
                "model",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "eval_count",
            )
        }
        if response.get("done") is not True:
            self.last_raw_output = {"incomplete_response": True}
            raise OllamaIncompleteResponseError(
                "Ollama returned an incomplete response (done was not true)"
            )
        if response.get("done_reason") != "stop":
            self.last_raw_output = {"incomplete_response": True}
            raise OllamaIncompleteResponseError(
                "Ollama response did not end with the normal stop reason"
            )
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise InferenceError("Ollama response has no message content") from exc
        if not isinstance(content, str):
            raise InferenceError("Ollama message content must be a string")
        if not content:
            self.last_raw_output = {"incomplete_response": True}
            raise OllamaIncompleteResponseError(
                "Ollama returned empty message content"
            )
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "Ollama completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "Ollama completion must be a JSON object"
            )
        self.last_raw_output = raw
        return raw

    def propose_dialogue_route(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> DialogueRoute:
        raw = self._complete(
            dialogue_route_json_schema(available_actions, suggested_action_ids),
            dialogue_route_messages(
                context, available_actions, suggested_action_ids
            ),
            skill=self.skill,
        )
        return validate_dialogue_route_decision(
            raw, available_actions, suggested_action_ids
        )

    def propose_tool_call(
        self, context: dict, available_actions: list[str]
    ) -> ToolCall:
        schema = tool_call_json_schema(available_actions, self._asset_ids(context))
        raw = self._complete(
            schema,
            tool_call_messages(context, available_actions),
            skill=self.argument_skill,
        )
        return self._validate(raw, available_actions)

    def propose_dialogue_turn(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> ModelDialogueTurn:
        schema = dialogue_turn_json_schema(
            available_actions, suggested_action_ids, self._asset_ids(context)
        )
        raw = self._complete(
            schema,
            dialogue_turn_messages(
                context, available_actions, suggested_action_ids
            ),
            skill=self.skill,
        )
        return validate_dialogue_turn(
            raw, available_actions, suggested_action_ids
        )
