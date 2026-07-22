"""OpenRouter-backed off-device inference for the web POC."""

from __future__ import annotations

import json
import warnings
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
    dialogue_route_json_schema,
    dialogue_route_messages,
    tool_call_json_schema,
    tool_call_messages,
    validate_dialogue_turn,
    validate_dialogue_route_decision,
)

OpenRouterTransport = Callable[
    [str, dict[str, Any], dict[str, str], float], dict[str, Any]
]


def _transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except HTTPError as exc:
        # Never include request headers or the API key in errors/logs.
        raise InferenceError(f"OpenRouter request failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise InferenceError(f"OpenRouter request failed: {type(exc).__name__}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError("OpenRouter returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise InferenceError("OpenRouter response must be a JSON object")
    return value


openrouter_tool_call_schema = tool_call_json_schema


class OpenRouterProvider(InferenceProvider):
    """Remote structured-output provider; model output remains only a proposal."""

    name = "openrouter"
    native_constrained_decoding = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_tokens: int = 128,
        data_collection: str = "deny",
        zero_data_retention: bool = False,
        transport: OpenRouterTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise InferenceError("OPENROUTER_API_KEY is required")
        if "gemma-4-e2b" not in model.lower() and "gemma-4-e4b" not in model.lower():
            warnings.warn(
                "The web demo is intended to use the target Gemma 4 E2B/E4B family; "
                f"configured model {model!r} is for review/debugging only.",
                RuntimeWarning,
                stacklevel=2,
            )
        if data_collection not in {"allow", "deny"}:
            raise ValueError("data_collection must be 'allow' or 'deny'")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.data_collection = data_collection
        self.zero_data_retention = zero_data_retention
        self._transport = transport or _transport
        self.last_response_metadata: dict[str, Any] = {}

    def propose_tool_call(
        self, context: dict, available_actions: list[str]
    ) -> ToolCall:
        asset_ids = context.get("canonical_asset_ids")
        schema = tool_call_json_schema(
            available_actions,
            asset_ids if isinstance(asset_ids, list) else None,
        )
        messages = tool_call_messages(context, available_actions)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "wallet_tool_call",
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {
                "require_parameters": True,
                "data_collection": self.data_collection,
                "zdr": self.zero_data_retention,
            },
        }
        response = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/agentic-wallet",
                "X-Title": "Agentic Wallet POC",
            },
            self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError("OpenRouter response has no completion content") from exc
        if not isinstance(content, str):
            raise InferenceError("OpenRouter completion content must be a string")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "OpenRouter completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "OpenRouter completion must be a JSON object"
            )
        self.last_raw_output = raw
        self.last_response_metadata = {
            "id": response.get("id"),
            "model": response.get("model", self.model),
            "provider": response.get("provider"),
            "usage": response.get("usage", {}),
        }
        return self._validate(raw, available_actions)

    def propose_dialogue_route(
        self,
        context: dict,
        available_actions: list[str],
        suggested_action_ids: list[str],
    ) -> DialogueRoute:
        schema = dialogue_route_json_schema(
            available_actions, suggested_action_ids
        )
        payload = {
            "model": self.model,
            "messages": dialogue_route_messages(
                context, available_actions, suggested_action_ids
            ),
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "wallet_dialogue_route",
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {
                "require_parameters": True,
                "data_collection": self.data_collection,
                "zdr": self.zero_data_retention,
            },
        }
        response = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/agentic-wallet",
                "X-Title": "Agentic Wallet POC",
            },
            self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError("OpenRouter response has no completion content") from exc
        if not isinstance(content, str):
            raise InferenceError("OpenRouter completion content must be a string")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "OpenRouter completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "OpenRouter completion must be a JSON object"
            )
        self.last_raw_output = raw
        self.last_response_metadata = {
            "id": response.get("id"),
            "model": response.get("model", self.model),
            "provider": response.get("provider"),
            "usage": response.get("usage", {}),
        }
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
            "model": self.model,
            "messages": dialogue_turn_messages(
                context, available_actions, suggested_action_ids
            ),
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "wallet_dialogue_turn",
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {
                "require_parameters": True,
                "data_collection": self.data_collection,
                "zdr": self.zero_data_retention,
            },
        }
        response = self._transport(
            f"{self.base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/agentic-wallet",
                "X-Title": "Agentic Wallet POC",
            },
            self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError("OpenRouter response has no completion content") from exc
        if not isinstance(content, str):
            raise InferenceError("OpenRouter completion content must be a string")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            self.last_raw_output = {"unparsed_output": content[:2_000]}
            raise ProposalValidationError(
                "OpenRouter completion was not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            self.last_raw_output = {"unparsed_output": raw}
            raise ProposalValidationError(
                "OpenRouter completion must be a JSON object"
            )
        self.last_raw_output = raw
        self.last_response_metadata = {
            "id": response.get("id"),
            "model": response.get("model", self.model),
            "provider": response.get("provider"),
            "usage": response.get("usage", {}),
        }
        return validate_dialogue_turn(
            raw, available_actions, suggested_action_ids
        )
