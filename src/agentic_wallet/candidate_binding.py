"""Deterministic binding for model-selected transaction candidates.

The model may select an opaque candidate ID, but it never creates or copies a
recipient address into an actionable field. Only the current user message or a
separately verified contact source may create a trusted candidate.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from eth_utils import is_checksum_address
from pydantic import Field, field_validator

from .inference import InferenceError
from .schemas.common import EvmAddress, StrictModel
from .schemas.tool_call import ToolCall

_ADDRESS_RE = re.compile(r"(?<![0-9a-fA-F])0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_BASE_UNITS_RE = re.compile(
    r"\b(0|[1-9]\d*)\s*(?:base[- ]?units?|wei)\b", re.IGNORECASE
)
_ALTERNATE_BASE_UNITS_RE = re.compile(
    r"\b(0|[1-9]\d*)\s+or\s+(0|[1-9]\d*)\s*"
    r"(?:base[- ]?units?|wei)\b",
    re.IGNORECASE,
)
_CHAIN_ID_RE = re.compile(r"\bchain(?:\s+id)?\s*[:#]?\s*(\d+)\b", re.IGNORECASE)
_CHAIN_NAMES = {
    "ethereum": 1,
    "polygon": 137,
    "optimism": 10,
    "arbitrum": 42161,
    "base": 8453,
}
CANDIDATE_TRANSFER_ACTION = "create_transfer_plan_from_candidate"


class TrustedRecipientCandidate(StrictModel):
    recipient_id: str = Field(pattern=r"^recipient:[a-z0-9-]+$")
    address: EvmAddress
    provenance: Literal["current-user-input", "verified-contact"]

    @field_validator("address")
    @classmethod
    def _valid_checksum_when_mixed_case(cls, value: str) -> str:
        hex_part = value[2:]
        has_lower = any(char.isalpha() and char.islower() for char in hex_part)
        has_upper = any(char.isalpha() and char.isupper() for char in hex_part)
        if has_lower and has_upper and not is_checksum_address(value):
            raise ValueError("mixed-case recipient has an invalid EIP-55 checksum")
        return value.lower()


class RequiredFactsMissing(InferenceError):
    """The deterministic fact set cannot safely construct the selected action."""

    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        super().__init__(f"required trusted facts are missing: {', '.join(fields)}")


class BoundTransferArguments(StrictModel):
    chain_id: int = Field(gt=0)
    asset_id: str = Field(pattern=r"^[a-z0-9]+:[a-z0-9\-]+$")
    amount_base_units: str = Field(pattern=r"^(0|[1-9]\d*)$")
    recipient: EvmAddress
    recipient_id: str = Field(pattern=r"^recipient:[a-z0-9-]+$")


def _current_user_candidate_id(user_request: str, address: str) -> str:
    commitment = hashlib.sha256(
        f"{user_request}\0{address.lower()}".encode()
    ).hexdigest()[:16]
    return f"recipient:user-{commitment}"


def extract_current_user_recipient_candidates(
    user_request: str,
) -> list[TrustedRecipientCandidate]:
    """Extract only explicit addresses from the current trusted user message."""

    candidates: list[TrustedRecipientCandidate] = []
    seen: set[str] = set()
    for match in _ADDRESS_RE.finditer(user_request):
        address = match.group(0)
        lowered = address.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        try:
            candidates.append(
                TrustedRecipientCandidate(
                    recipient_id=_current_user_candidate_id(
                        user_request, lowered
                    ),
                    address=address,
                    provenance="current-user-input",
                )
            )
        except ValueError:
            # Invalid mixed-case/checksum inputs are not trusted candidates.
            continue
    return candidates


def prepare_inference_context(context: dict[str, Any]) -> dict[str, Any]:
    """Attach bounded typed candidates without reading untrusted context fields."""

    prepared = dict(context)
    request = prepared.get("user_request")
    current = (
        extract_current_user_recipient_candidates(request)
        if isinstance(request, str)
        else []
    )
    verified_raw = prepared.pop("verified_recipient_candidates", [])
    verified: list[TrustedRecipientCandidate] = []
    if isinstance(verified_raw, list):
        for value in verified_raw[:8]:
            try:
                candidate = TrustedRecipientCandidate.model_validate(value)
            except ValueError as exc:
                raise InferenceError(
                    f"invalid verified recipient candidate: {exc}"
                ) from exc
            if candidate.provenance != "verified-contact":
                raise InferenceError(
                    "externally supplied recipient candidates must be verified contacts"
                )
            verified.append(candidate)
    candidates = current + [
        item
        for item in verified
        if item.address not in {candidate.address for candidate in current}
    ]
    prepared["trusted_recipient_candidates"] = [
        candidate.model_dump() for candidate in candidates[:8]
    ]
    return prepared


def _validated_context_candidates(
    context: dict[str, Any]
) -> list[TrustedRecipientCandidate]:
    values = context.get("trusted_recipient_candidates", [])
    if not isinstance(values, list):
        return []
    candidates: list[TrustedRecipientCandidate] = []
    for value in values:
        try:
            candidates.append(TrustedRecipientCandidate.model_validate(value))
        except ValueError as exc:
            raise InferenceError(f"invalid trusted recipient candidate: {exc}") from exc
    request = context.get("user_request")
    for candidate in candidates:
        if candidate.provenance != "current-user-input":
            continue
        if not isinstance(request, str) or candidate.recipient_id != (
            _current_user_candidate_id(request, candidate.address)
        ):
            raise InferenceError("current-user recipient commitment mismatch")
    return candidates


def recipient_candidate_ids(context: dict[str, Any]) -> list[str]:
    candidates = _validated_context_candidates(context)
    return [candidate.recipient_id for candidate in candidates]


def _single_asset_id(context: dict[str, Any], request: str) -> str | None:
    values = context.get("canonical_asset_ids", [])
    if not isinstance(values, list):
        return None
    lowered_request = request.casefold()
    matches = []
    for value in values:
        if not isinstance(value, str):
            continue
        symbol = value.rsplit(":", 1)[-1]
        if value.casefold() in lowered_request or re.search(
            rf"\b{re.escape(symbol.casefold())}\b", lowered_request
        ):
            matches.append(value)
    return matches[0] if len(set(matches)) == 1 else None


def _request_matches_chain(request: str, chain_id: int) -> bool:
    explicit_ids = {int(value) for value in _CHAIN_ID_RE.findall(request)}
    lowered = request.casefold()
    named_ids: set[int] = set()
    for name, value in _CHAIN_NAMES.items():
        if name == "base":
            # "base units" is an amount denomination, not chain selection.
            pattern = (
                r"\b(?:on|over|via)\s+(?:the\s+)?base\b"
                r"|\bbase\s+(?:chain|network)\b"
            )
        else:
            pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, lowered):
            named_ids.add(value)
    requested = explicit_ids | named_ids
    return not requested or requested == {chain_id}


def _base_unit_amounts(request: str) -> list[str]:
    values = list(_BASE_UNITS_RE.findall(request))
    for first, second in _ALTERNATE_BASE_UNITS_RE.findall(request):
        values.extend([first, second])
    return list(dict.fromkeys(values))


def deterministic_candidate_tool_call(
    action: str, context: dict[str, Any]
) -> ToolCall | None:
    """Construct candidate-bound transfer arguments or force clarification.

    Other actions continue through their normal selected-action model call.
    """

    candidate_context = "trusted_recipient_candidates" in context
    if action not in {CANDIDATE_TRANSFER_ACTION, "request_missing_information"}:
        return None
    if action == "request_missing_information" and not candidate_context:
        return None
    request = context.get("user_request")
    request = request if isinstance(request, str) else ""
    candidates = recipient_candidate_ids(context)
    amounts = _base_unit_amounts(request)
    asset_id = _single_asset_id(context, request)
    ledger = context.get("conversation_ledger", {})
    chain_id = context.get("chain_id")
    if not isinstance(chain_id, int) and isinstance(ledger, dict):
        chain_id = ledger.get("chain_id")

    missing: list[str] = []
    if len(candidates) != 1:
        missing.append("recipient")
    if asset_id is None:
        missing.append("asset_id")
    if len(amounts) != 1:
        missing.append("amount_base_units")
    if not isinstance(chain_id, int) or chain_id <= 0:
        missing.append("chain_id")
    elif not _request_matches_chain(request, chain_id):
        missing.append("chain_id")
    if missing:
        if action == "request_missing_information":
            return ToolCall(
                action=action,
                arguments={"missing_fields": missing},
                reason="Deterministically derived from missing trusted facts.",
            )
        raise RequiredFactsMissing(missing)

    if action == "request_missing_information":
        return None

    return ToolCall(
        action=action,
        arguments={
            "chain_id": chain_id,
            "asset_id": asset_id,
            "amount_base_units": amounts[0],
            "recipient_id": candidates[0],
        },
        reason="Deterministically bound from trusted typed facts.",
    )


def bind_transfer_candidate(
    call: ToolCall, context: dict[str, Any]
) -> BoundTransferArguments:
    """Resolve an opaque model-facing recipient ID to its trusted address."""

    if call.action != CANDIDATE_TRANSFER_ACTION:
        raise InferenceError("candidate binding only accepts candidate transfer calls")
    recipient_id = call.arguments.get("recipient_id")
    candidates = _validated_context_candidates(context)
    matches = [item for item in candidates if item.recipient_id == recipient_id]
    if len(matches) != 1:
        raise InferenceError("recipient candidate is missing or ambiguous")
    return BoundTransferArguments(
        chain_id=call.arguments.get("chain_id"),
        asset_id=call.arguments.get("asset_id"),
        amount_base_units=call.arguments.get("amount_base_units"),
        recipient=matches[0].address,
        recipient_id=matches[0].recipient_id,
    )
