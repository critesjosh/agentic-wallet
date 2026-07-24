"""Deterministic chat responder with an optional review-only transfer request.

Chat is the only user interface to the wallet. This responder stands in for the
fine-tuned target model behind ``InferenceProvider``.  It never approves,
signs, or submits; when live transactions are explicitly enabled, one narrow
deterministic command may only request that the server build a review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..chain_metadata import (
    ChainMetadataError,
    explorer_address_url,
    get_chain_metadata,
)
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
    "I'm a wallet assistant. Try: \"what's my address\", \"show my portfolio\", "
    "\"what's my USDC balance\", \"show allowances\", or \"show the registry\". "
    "Swaps and approvals are not wired up. Native transfers are available only "
    "when this deployment explicitly enables them, and they always require a "
    "simulation and your separate approval of the exact digest."
)

_STATE_CHANGING = re.compile(
    r"\b(?:swap|swaps|send|sending|transfer|transfers|approve|approval|buy|buying|"
    r"sell|selling|bridge|bridging|sign|signed|signing|submit|submitted|submitting)\b",
    re.IGNORECASE,
)
_CONCEPTUAL = re.compile(r"\b(?:what|why|how|explain|learn|mean|work)\b", re.IGNORECASE)
_NATIVE_TRANSFER_COMMAND = re.compile(
    r"^\s*send\s+(?P<amount>\d{1,30}(?:\.\d{1,30})?)\s+(?P<unit>wei|eth)\s+to\s+"
    r"(?P<recipient>0x[0-9a-fA-F]{40})\s+on\s+(?P<chain>base\s+sepolia|base)\s*$",
    re.IGNORECASE,
)

# Code-owned phrase to chain mapping. A model or untrusted string can never
# introduce a chain here; an unrecognized phrase simply fails to match.
_CHAIN_PHRASES = {"base": 8453, "base sepolia": 84532}
_TRANSACTION_STATUS_COMMAND = re.compile(
    r"^\s*(?:check|show|look\s+up)\s+(?:transaction|tx)\s+"
    r"(?P<transaction_hash>0x[0-9a-fA-F]{64})\s*$",
    re.IGNORECASE,
)

_BASE_MODEL_ACTIONS = [
    "get_account",
    "get_portfolio",
    "get_balance",
    "get_allowances",
    "get_registry",
    "show_help",
    "reject_state_changing",
]

_TRANSFER_MODEL_ACTIONS = ["request_native_transfer_review", "get_transaction_status"]

_SUGGESTIONS = {
    "get_account": SuggestedAction(
        action="get_account", label="Show my account", prompt="What is my address?"
    ),
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
@dataclass(frozen=True)
class _TransferCommand:
    """Outcome of parsing one exact current-message transfer command.

    ``candidate`` is populated only when every field is proven; otherwise
    ``error`` explains which safety fact could not be established. A message
    that is not a transfer command at all yields neither.
    """

    candidate: dict[str, str | int] | None = None
    error: str | None = None


def _decimal_string_to_base_units(amount: str, decimals: int) -> str:
    """Convert a decimal string to integer base units using exact integer math.

    Floats are never involved, so no representable-value drift can change the
    amount a user is asked to approve.
    """

    whole, _, fraction = amount.partition(".")
    if len(fraction) > decimals:
        raise ValueError(
            f"amount has more precision than the asset's {decimals} decimals"
        )
    return str(int(f"{whole or '0'}{fraction.ljust(decimals, '0')}"))


def parse_native_transfer_command(
    message: str, *, live_chain_id: int, native_decimals: int
) -> _TransferCommand:
    """Extract an exact transfer candidate from the current user message only.

    Nothing here consults conversation history, model output, or retrieved
    text: a transfer can be described only by the message being sent now.
    """

    match = _NATIVE_TRANSFER_COMMAND.fullmatch(message)
    if match is None:
        return _TransferCommand()
    chain_id = _CHAIN_PHRASES[" ".join(match.group("chain").lower().split())]
    if chain_id != live_chain_id:
        return _TransferCommand(
            error=(
                f"This wallet is configured for "
                f"{get_chain_metadata(live_chain_id).name} (chain {live_chain_id}), "
                f"so it cannot open a review on "
                f"{get_chain_metadata(chain_id).name}."
            )
        )
    amount, unit = match.group("amount"), match.group("unit").lower()
    if unit == "wei":
        if "." in amount:
            return _TransferCommand(error="Amounts in wei must be whole numbers.")
        base_units = str(int(amount))
    else:
        try:
            base_units = _decimal_string_to_base_units(amount, native_decimals)
        except ValueError as exc:
            return _TransferCommand(error=f"{exc}.".capitalize())
    if int(base_units) <= 0:
        return _TransferCommand(error="The transfer amount must be greater than zero.")
    return _TransferCommand(
        candidate={
            "chain_id": chain_id,
            "amount_base_units": base_units,
            "recipient": match.group("recipient"),
        }
    )


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
        transfer_requests_enabled: bool = False,
        live_chain_id: int = 8453,
    ) -> None:
        self.harness = harness
        self.registry = registry
        self.provider = provider
        # This only permits a deterministic user-supplied command to request a
        # review.  It never enables approval, signing, or submission in chat.
        self.transfer_requests_enabled = transfer_requests_enabled
        # Fail fast if the deployment names a chain the signing allowlist and
        # registry cannot both prove.
        get_chain_metadata(live_chain_id)
        self.registry.native_asset(live_chain_id)
        self.live_chain_id = live_chain_id
        # Set per request by the application from the isolated signer. None
        # means no real account is loaded, so the account view says so.
        self.signer_address: str | None = None
        validate_production_actions([*_BASE_MODEL_ACTIONS, *_TRANSFER_MODEL_ACTIONS])
        self._sessions: dict[str, _Session] = {}

    @property
    def model_actions(self) -> list[str]:
        """Derive the offered actions so they cannot drift from the flag.

        This is computed rather than stored because the flag is set by the
        application after construction, once it knows whether a complete
        transaction controller exists.
        """

        return [
            *_BASE_MODEL_ACTIONS,
            *(_TRANSFER_MODEL_ACTIONS if self.transfer_requests_enabled else []),
        ]

    @property
    def _native_decimals(self) -> int:
        return self.registry.native_asset(self.live_chain_id).decimals

    def _transfer_usage(self) -> str:
        chain = get_chain_metadata(self.live_chain_id).name.lower()
        return (
            "I need an exact current-message transfer candidate before I can "
            f"open a review. Use: send <amount> eth to <0x address> on {chain}. "
            "Whole amounts in wei also work."
        )

    def _parse_transfer(self, message: str) -> _TransferCommand:
        if not self.transfer_requests_enabled:
            return _TransferCommand()
        return parse_native_transfer_command(
            message,
            live_chain_id=self.live_chain_id,
            native_decimals=self._native_decimals,
        )

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
        transaction_request: dict[str, str | int] | None = None,
        transaction_status_request: dict[str, str] | None = None,
    ) -> dict:
        return {
            "reply": reply,
            "state": session.sm.state.value,
            "data": data,
            "suggested_actions": self._suggestions(suggested_action_ids or []),
            "transaction_request": transaction_request,
            "transaction_status_request": transaction_status_request,
        }

    @staticmethod
    def _record_history(session: _Session, user_message: str, reply: str) -> None:
        session.ledger.workflow_state = session.sm.state.value
        session.ledger.record_message("user", user_message)
        session.ledger.record_message("assistant", reply)

    def _transfer_review_response(
        self, session: _Session, candidate: dict[str, str | int]
    ) -> dict:
        """Return only a review request from exact current-message fields."""

        return self._response(
            session,
            "I can create a simulated native-transfer review. It is not approved "
            "and nothing will be signed or sent until you separately approve the "
            "exact digest shown in the review card.",
            transaction_request=candidate,
        )

    def _transaction_status_response(
        self, session: _Session, status_match: re.Match[str]
    ) -> dict:
        return self._response(
            session,
            "I’ll look up that exact transaction hash in this browser session’s "
            "saved transaction state.",
            transaction_status_request={
                "transaction_hash": status_match.group("transaction_hash").lower()
            },
        )

    def debug_ledger(self, session_id: str) -> dict:
        """Return a copy for local tests/debugging; never an authorization object."""

        return self._session(session_id).ledger.model_dump()

    def _account_view(self) -> dict:
        """Build the account identity, distinguishing a real signer from fixture data.

        The demo portfolio is a synthetic fixture. Presenting its placeholder
        address as the user's account, next to a working explorer link, would
        invite someone to fund an address nobody controls, so a fixture is
        always labelled as one and never gets a link.
        """

        portfolio = self.harness.get_portfolio()
        if self.signer_address is not None:
            # A provisioned signer is the real account; its chain is the one
            # this deployment signs on, not the fixture snapshot's chain.
            return {
                "type": "account",
                "account": self._describe_account(
                    address=self.signer_address,
                    chain_id=self.live_chain_id,
                    source="signer",
                    as_of_block=None,
                    stale=False,
                ),
            }
        return {
            "type": "account",
            "account": self._describe_account(
                address=portfolio.address,
                chain_id=portfolio.chain_id,
                source="fixture",
                as_of_block=portfolio.as_of_block,
                stale=portfolio.stale,
            ),
        }

    @staticmethod
    def _describe_account(
        *,
        address: str,
        chain_id: int,
        source: str,
        as_of_block: int | None,
        stale: bool,
    ) -> dict[str, Any]:
        real = source == "signer"
        account: dict[str, Any] = {
            "address": address,
            "chain_id": chain_id,
            "source": source,
            "watch_only": not real,
            "as_of_block": as_of_block,
            "stale": stale,
        }
        try:
            account["chain_name"] = get_chain_metadata(chain_id).name
        except ChainMetadataError:
            # An unrecognized chain gets no trusted name rather than an
            # unverified one that would look equally trustworthy.
            account["chain_name"] = f"chain {chain_id}"
        account["explorer_url"] = None
        if real:
            try:
                account["explorer_url"] = explorer_address_url(chain_id, address)
            except ChainMetadataError:
                account["explorer_url"] = None
        return account

    def _balance_amount(self, asset_id: str):
        """Read one canonical balance, binding native assets to their chain.

        ``base:native`` and ``base:sepolia-native`` are both native, so the
        native path must be selected by registry provenance rather than a
        literal ID, and only when the snapshot covers that asset's chain.
        """

        entry = self.registry.resolve(asset_id)
        if not entry.is_native:
            return self.harness.get_token_balance(asset_id)
        snapshot_chain = self.harness.get_portfolio().chain_id
        if entry.chain_id != snapshot_chain:
            raise HarnessError(
                f"the portfolio snapshot covers chain {snapshot_chain}, "
                f"not chain {entry.chain_id}"
            )
        return self.harness.get_native_balance()

    def _execute_model_call(self, session: _Session, call: ToolCall) -> dict:
        try:
            if call.action == "get_account":
                validate_tool_arguments(call.action, call.arguments)
                data = self._account_view()
            elif call.action == "get_portfolio":
                validate_tool_arguments(call.action, call.arguments)
                portfolio = self.harness.get_portfolio()
                data = {"type": "portfolio", "portfolio": portfolio.model_dump()}
            elif call.action == "get_balance":
                arguments = validate_tool_arguments(call.action, call.arguments)
                assert isinstance(arguments, BalanceArguments)
                amount = self._balance_amount(arguments.asset_id)
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
                    (
                        "That state-changing request is not supported. Only an exact "
                        "native transfer command can open a review, and no "
                        "transaction was drafted, signed, or submitted."
                        if self.transfer_requests_enabled
                        else
                        "State-changing actions are not enabled in this read-only "
                        "demo. No transaction was drafted, signed, or submitted."
                    ),
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
        transfer = self._parse_transfer(message)
        status_match = _TRANSACTION_STATUS_COMMAND.fullmatch(message)
        context = {
            "user_request": message,
            "conversation_ledger": session.ledger.model_dump(),
            "canonical_asset_ids": [entry.asset_id for entry in self.registry.entries()],
            "transaction_review_enabled": self.transfer_requests_enabled,
            "read_only": not self.transfer_requests_enabled,
            "parsed_native_transfer_candidate": (
                {**transfer.candidate, "provenance": "exact_current_user_message"}
                if transfer.candidate is not None
                else None
            ),
            "parsed_transaction_status_candidate": (
                {
                    "transaction_hash": status_match.group(
                        "transaction_hash"
                    ).lower(),
                    "provenance": "exact_current_user_message",
                }
                if status_match is not None and self.transfer_requests_enabled
                else None
            ),
        }
        try:
            route = self.provider.propose_dialogue_route_with_repair(
                context, self.model_actions, list(_SUGGESTIONS)
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
        if route.proposed_action == "request_native_transfer_review":
            if transfer.candidate is None:
                return self._response(session, transfer.error or self._transfer_usage())
            return self._transfer_review_response(session, transfer.candidate)
        if route.proposed_action == "get_transaction_status":
            if status_match is None or not self.transfer_requests_enabled:
                return self._response(
                    session,
                    "I need an exact transaction hash in the current message. "
                    "Use: check transaction <0x transaction hash>.",
                )
            return self._transaction_status_response(session, status_match)

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
        elif "registry" in text or "trusted asset" in text:
            data = {
                "type": "registry",
                "version_digest": self.registry.version_digest(),
                "entries": [e.__dict__ for e in self.registry.entries()],
            }
            reply = "Canonical registry (the trusted id -> address mapping):"
        elif any(w in text for w in ("address", "account", "who am i")):
            # Checked after the registry so "show the registry addresses" still
            # resolves to the registry it explicitly names.
            data = self._account_view()
            reply = render_verified_result(data)
            suggestions = ["get_portfolio", "get_balance"]
        elif "balance" in text:
            asset = next((v for k, v in _ASSET_ALIASES.items() if k in text), None)
            if asset is None:
                reply = "Which asset? I can read base:native (ETH), base:usdc, or base:weth."
            else:
                try:
                    amt = self._balance_amount(asset)
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
        transfer = self._parse_transfer(message)
        status_match = _TRANSACTION_STATUS_COMMAND.fullmatch(message)
        if self.provider is not None:
            response = self._respond_with_model(session, message)
        elif transfer.candidate is not None:
            # In deterministic fallback mode, exact current-message extraction
            # replaces model routing. The proposal endpoint revalidates all
            # fields and this response still cannot approve or submit.
            response = self._transfer_review_response(session, transfer.candidate)
        elif transfer.error is not None:
            response = self._response(session, transfer.error)
        elif status_match is not None and self.transfer_requests_enabled:
            response = self._transaction_status_response(session, status_match)
        else:
            response = self._respond_with_keywords(session, message)
        self._record_history(session, message, response["reply"])
        return response
