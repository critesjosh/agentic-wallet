"""Deterministic read-only chat responder.

Chat is the only user interface to the wallet. This responder stands in for the
fine-tuned target model behind ``InferenceProvider``: when the model is wired
(``RemoteHTTPProvider`` -> Gemma), it replaces this keyword logic while the
harness, tools, and state machine stay identical. It is read-only and never
signs, submits, or drafts an executable plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..harness import HarnessError, MockReadOnlyHarness
from ..inference import InferenceError, InferenceProvider
from ..registry import BASE_REGISTRY, Registry, RegistryError
from ..schemas.conversation import ConversationLedger
from ..schemas.dialogue import SuggestedAction
from ..schemas.tool_call import ToolCall
from ..state_machine import StateMachine
from ..tool_contract import (
    BalanceArguments,
    validate_production_actions,
    validate_tool_arguments,
)
from .narration import render_verified_result, validate_grounded_message

_ASSET_ALIASES = {
    "eth": "base:native",
    "native": "base:native",
    "usdc": "base:usdc",
    "weth": "base:weth",
}

_HELP = (
    "I'm a read-only wallet assistant. Try: \"show my portfolio\", "
    "\"what's my USDC balance\", \"show allowances\", or \"show the registry\". "
    "State-changing actions (swaps, transfers) are not wired yet: they require "
    "the model to draft a plan, a simulation, and your explicit approval."
)

_STATE_CHANGING = re.compile(
    r"\b(?:swap|swaps|send|sending|transfer|transfers|approve|approval|buy|buying|"
    r"sell|selling|bridge|bridging|sign|signed|signing|submit|submitted|submitting)\b",
    re.IGNORECASE,
)
_CONCEPTUAL = re.compile(r"\b(?:what|why|how|explain|learn|mean|work)\b", re.IGNORECASE)

_MODEL_ACTIONS = [
    "get_portfolio",
    "get_balance",
    "get_allowances",
    "get_registry",
    "show_help",
    "reject_state_changing",
]

_SUGGESTIONS = {
    "get_portfolio": SuggestedAction(
        action="get_portfolio", label="Show portfolio", prompt="Show my portfolio"
    ),
    "get_balance": SuggestedAction(
        action="get_balance", label="Check a balance", prompt="What balances can you check?"
    ),
    "get_allowances": SuggestedAction(
        action="get_allowances", label="Check allowances", prompt="Show my allowances"
    ),
    "get_registry": SuggestedAction(
        action="get_registry", label="Show trusted assets", prompt="Show the trusted asset registry"
    ),
}
@dataclass
class _Session:
    sm: StateMachine
    ledger: ConversationLedger


class DemoChatAgent:
    def __init__(
        self,
        harness: MockReadOnlyHarness,
        registry: Registry = BASE_REGISTRY,
        provider: InferenceProvider | None = None,
    ) -> None:
        validate_production_actions(_MODEL_ACTIONS)
        self.harness = harness
        self.registry = registry
        self.provider = provider
        self._sessions: dict[str, _Session] = {}

    def _session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            sm = StateMachine()
            session = _Session(
                sm=sm,
                ledger=ConversationLedger(
                    workflow_state=sm.state.value,
                    chain_id=self.harness.get_portfolio().chain_id,
                ),
            )
            self._sessions[session_id] = session
        return session

    @staticmethod
    def _suggestions(action_ids: list[str]) -> list[dict]:
        return [
            _SUGGESTIONS[action_id].model_dump()
            for action_id in action_ids[:3]
            if action_id in _SUGGESTIONS
        ]

    def _response(
        self,
        session: _Session,
        reply: str,
        data: Any = None,
        suggested_action_ids: list[str] | None = None,
    ) -> dict:
        return {
            "reply": reply,
            "state": session.sm.state.value,
            "data": data,
            "suggested_actions": self._suggestions(suggested_action_ids or []),
        }

    @staticmethod
    def _record_history(session: _Session, user_message: str, reply: str) -> None:
        session.ledger.workflow_state = session.sm.state.value
        session.ledger.record_message("user", user_message)
        session.ledger.record_message("assistant", reply)

    def debug_ledger(self, session_id: str) -> dict:
        """Return a copy for local tests/debugging; never an authorization object."""

        return self._session(session_id).ledger.model_dump()

    def _execute_model_call(self, session: _Session, call: ToolCall) -> dict:
        try:
            if call.action == "get_portfolio":
                validate_tool_arguments(call.action, call.arguments)
                portfolio = self.harness.get_portfolio()
                data = {"type": "portfolio", "portfolio": portfolio.model_dump()}
            elif call.action == "get_balance":
                arguments = validate_tool_arguments(call.action, call.arguments)
                assert isinstance(arguments, BalanceArguments)
                amount = (
                    self.harness.get_native_balance()
                    if arguments.asset_id == "base:native"
                    else self.harness.get_token_balance(arguments.asset_id)
                )
                data = {
                    "type": "balance",
                    "asset_id": arguments.asset_id,
                    "amount": amount.model_dump(),
                }
            elif call.action == "get_allowances":
                validate_tool_arguments(call.action, call.arguments)
                portfolio = self.harness.get_portfolio()
                allowances = [allowance.model_dump() for allowance in portfolio.allowances]
                data = {"type": "allowances", "allowances": allowances}
            elif call.action == "get_registry":
                validate_tool_arguments(call.action, call.arguments)
                data = {
                    "type": "registry",
                    "version_digest": self.registry.version_digest(),
                    "entries": [entry.__dict__ for entry in self.registry.entries()],
                }
            elif call.action == "show_help":
                validate_tool_arguments(call.action, call.arguments)
                return self._response(session, _HELP)
            elif call.action == "reject_state_changing":
                validate_tool_arguments(call.action, call.arguments)
                return self._response(
                    session,
                    "State-changing actions are not enabled in this read-only demo. "
                    "No transaction was drafted, signed, or submitted.",
                )
            else:
                return self._response(session, "The proposed action was rejected.")
            session.ledger.record_validated_arguments(call.arguments)
            session.ledger.record_proposal(call.action, call.arguments)
            session.ledger.record_verified_fact(str(data["type"]), data)
            return self._response(session, render_verified_result(data), data)
        except (HarnessError, InferenceError, RegistryError, ValueError) as exc:
            return self._response(session, f"The proposed read action was rejected: {exc}")

    def _respond_with_model(self, session: _Session, message: str) -> dict:
        context = {
            "user_request": message,
            "conversation_ledger": session.ledger.model_dump(),
            "canonical_asset_ids": [entry.asset_id for entry in self.registry.entries()],
            "read_only": True,
        }
        try:
            route = self.provider.propose_dialogue_route_with_repair(
                context, _MODEL_ACTIONS, list(_SUGGESTIONS)
            )
        except InferenceError:
            return self._response(
                session,
                "I couldn't produce a valid conversational response, so no wallet tool was run.",
                suggested_action_ids=["get_portfolio", "get_balance"],
            )
        if route.proposed_action is None:
            return self._response(
                session,
                route.message,
                suggested_action_ids=route.suggested_actions,
            )

        argument_context = {
            **context,
            "phase": "fill_tool_arguments",
            "selected_action": route.proposed_action,
            "route_reason": route.reason,
        }
        try:
            call = self.provider.propose_tool_call_with_repair(
                argument_context, route.proposed_action
            )
        except InferenceError:
            return self._response(
                session,
                "I couldn't produce valid typed arguments after one repair attempt, "
                "so no wallet tool was run.",
                suggested_action_ids=["get_portfolio", "get_balance"],
            )

        tool_response = self._execute_model_call(session, call)
        if tool_response["data"] is None:
            return tool_response

        explanation_context = {
            "phase": "explain_verified_tool_result",
            "user_request": message,
            "conversation_ledger": session.ledger.model_dump(),
            "proposed_action": call.model_dump(),
            "verified_tool_result": tool_response["data"],
            "deterministic_summary": tool_response["reply"],
            "read_only": True,
        }
        try:
            explanation = self.provider.propose_dialogue_route_with_repair(
                explanation_context, [], list(_SUGGESTIONS)
            )
            message = validate_grounded_message(
                explanation.message,
                tool_response["data"],
                tool_response["reply"],
            )
        except InferenceError:
            return tool_response
        return self._response(
            session,
            message,
            tool_response["data"],
            explanation.suggested_actions,
        )

    def _respond_with_keywords(self, session: _Session, message: str) -> dict:
        text = message.strip().lower()
        data: Optional[dict] = None
        suggestions: list[str] = []

        if not text or text in {"hi", "hello", "help", "?"}:
            reply = _HELP
            suggestions = ["get_portfolio", "get_balance", "get_allowances"]
        elif any(w in text for w in ("portfolio", "holdings", "positions")):
            p = self.harness.get_portfolio()
            data = {"type": "portfolio", "portfolio": p.model_dump()}
            reply = (
                f"Watch-only portfolio for {p.address} on chain {p.chain_id} "
                f"(block {p.as_of_block})."
            )
        elif "allowance" in text:
            p = self.harness.get_portfolio()
            data = {"type": "allowances", "allowances": [a.model_dump() for a in p.allowances]}
            reply = "Current token allowances:" if p.allowances else "No allowances set."
        elif "registry" in text or "address" in text:
            data = {
                "type": "registry",
                "version_digest": self.registry.version_digest(),
                "entries": [e.__dict__ for e in self.registry.entries()],
            }
            reply = "Canonical registry (the trusted id -> address mapping):"
        elif "balance" in text:
            asset = next((v for k, v in _ASSET_ALIASES.items() if k in text), None)
            if asset is None:
                reply = "Which asset? I can read base:native (ETH), base:usdc, or base:weth."
            else:
                try:
                    amt = (
                        self.harness.get_native_balance()
                        if asset == "base:native"
                        else self.harness.get_token_balance(asset)
                    )
                    data = {"type": "balance", "asset_id": asset, "amount": amt.model_dump()}
                    reply = (
                        f"{asset} balance: {amt.base_units} base units "
                        f"(decimals {amt.decimals})."
                    )
                except (RegistryError, HarnessError) as exc:
                    reply = f"Could not read {asset}: {exc}"
        elif _STATE_CHANGING.search(text):
            if _CONCEPTUAL.search(text):
                reply = (
                    "I can explain that workflow, but this demo cannot execute it. "
                    "A state-changing request would require a typed plan, simulation, "
                    "policy checks, exact approval, and a separate wallet signer."
                )
            else:
                reply = (
                    "State-changing actions are not enabled in this read-only demo. "
                    "They require a typed plan, simulation, policy checks, and your "
                    "explicit approval; none of those automatically sign or submit."
                )
            suggestions = ["get_portfolio", "get_allowances"]
        else:
            reply = "I didn't catch a supported request. " + _HELP
            suggestions = ["get_portfolio", "get_balance", "get_allowances"]

        if data is not None:
            session.ledger.record_verified_fact(str(data["type"]), data)
        return self._response(session, reply, data, suggestions)

    def respond(self, session_id: str, message: str) -> dict:
        session = self._session(session_id)
        if self.provider is not None:
            response = self._respond_with_model(session, message)
        else:
            response = self._respond_with_keywords(session, message)
        self._record_history(session, message, response["reply"])
        return response
