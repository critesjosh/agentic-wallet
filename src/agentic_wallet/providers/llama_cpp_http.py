"""Schema-constrained llama.cpp HTTP inference.

The provider is usable against a remote llama.cpp deployment or an Android
device reached through ``adb forward``.  It deliberately uses the native
``/completion`` endpoint because that endpoint accepts a per-request JSON
schema, making the set of currently available actions part of decoding rather
than only a post-generation check.
"""

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
    dialogue_turn_prompt,
    dialogue_route_json_schema,
    dialogue_route_prompt,
    tool_call_json_schema,
    tool_call_prompt,
    validate_dialogue_turn,
    validate_dialogue_route_decision,
)

JSONTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_transport(
    url: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise InferenceError(
            f"llama.cpp request failed: {type(exc).__name__}"
        ) from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError("llama.cpp returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise InferenceError("llama.cpp response must be a JSON object")
    return value


class LlamaCppHTTPProvider(InferenceProvider):
    """Call a persistent llama.cpp server with fail-closed constrained output."""

    name = "llama-cpp-http"
    native_constrained_decoding = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18080",
        *,
        max_new_tokens: int = 128,
        timeout: float = 180.0,
        transport: JSONTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self._transport = transport or _default_transport
        self.last_response_metadata: dict[str, Any] = {}

    @staticmethod
    def _build_prompt(context: dict, available_actions: list[str]) -> str:
        return tool_call_prompt(context, available_actions)

    def propose_tool_call(
        self, context: dict, available_actions: list[str]
    ) -> ToolCall:
        asset_ids = context.get("canonical_asset_ids")
        schema = tool_call_json_schema(
            available_actions,
            asset_ids if isinstance(asset_ids, list) else None,
        )
        payload = {
            "prompt": self._build_prompt(context, available_actions),
            "json_schema": schema,
            "n_predict": self.max_new_tokens,
            "temperature": 0.0,
            "seed": 0,
            "stream": False,
            "cache_prompt": True,
        }
        response = self._transport(
            f"{self.base_url}/completion", payload, self.timeout
        )
        content = response.get("content")
        if not isinstance(content, str):
            raise InferenceError("llama.cpp response has no string content")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "llama.cpp completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "llama.cpp completion must be a JSON object"
            )
        self.last_raw_output = raw
        self.last_response_metadata = {
            "timings": response.get("timings", {}),
            "tokens_evaluated": response.get("tokens_evaluated"),
            "truncated": response.get("truncated"),
            "stop_type": response.get("stop_type"),
        }
        return self._validate(raw, available_actions)

    def propose_dialogue_route(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> DialogueRoute:
        payload = {
            "prompt": dialogue_route_prompt(
                context, available_actions, suggested_action_ids
            ),
            "json_schema": dialogue_route_json_schema(
                available_actions, suggested_action_ids
            ),
            "n_predict": self.max_new_tokens,
            "temperature": 0.0,
            "seed": 0,
            "stream": False,
            "cache_prompt": True,
        }
        response = self._transport(
            f"{self.base_url}/completion", payload, self.timeout
        )
        content = response.get("content")
        if not isinstance(content, str):
            raise InferenceError("llama.cpp response has no string content")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "llama.cpp completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "llama.cpp completion must be a JSON object"
            )
        self.last_raw_output = raw
        return validate_dialogue_route_decision(
            raw, available_actions, suggested_action_ids
        )

    def propose_dialogue_turn(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> ModelDialogueTurn:
        asset_ids = context.get("canonical_asset_ids")
        schema = dialogue_turn_json_schema(
            available_actions,
            suggested_action_ids,
            asset_ids if isinstance(asset_ids, list) else None,
        )
        payload = {
            "prompt": dialogue_turn_prompt(
                context, available_actions, suggested_action_ids
            ),
            "json_schema": schema,
            "n_predict": self.max_new_tokens,
            "temperature": 0.0,
            "seed": 0,
            "stream": False,
            "cache_prompt": True,
        }
        response = self._transport(
            f"{self.base_url}/completion", payload, self.timeout
        )
        content = response.get("content")
        if not isinstance(content, str):
            raise InferenceError("llama.cpp response has no string content")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "llama.cpp completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "llama.cpp completion must be a JSON object"
            )
        self.last_raw_output = raw
        self.last_response_metadata = {
            "timings": response.get("timings", {}),
            "tokens_evaluated": response.get("tokens_evaluated"),
            "truncated": response.get("truncated"),
            "stop_type": response.get("stop_type"),
        }
        return validate_dialogue_turn(
            raw, available_actions, suggested_action_ids
        )
