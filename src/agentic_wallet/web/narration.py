"""Deterministic summaries and grounded optional model narration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..inference import InferenceError

_EXECUTION_CLAIM = re.compile(
    r"\b(?:i|we|the wallet)\s+(?:have\s+)?(?:signed|submitted|sent|executed|moved)\b|"
    r"\btransaction\s+(?:was|is|has been)\s+(?:signed|submitted|sent|executed)\b",
    re.IGNORECASE,
)
_CANONICAL_ID = re.compile(r"(?<![a-z0-9])[a-z0-9]+:[a-z0-9-]+", re.IGNORECASE)
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
_NUMBER = re.compile(r"(?<![a-z0-9-])\d+(?:\.\d+)?(?![a-z0-9-])", re.IGNORECASE)


@dataclass(frozen=True)
class DeterministicNarration:
    message: str
    data: dict[str, Any]


def render_verified_result(data: dict[str, Any]) -> str:
    """Render trusted tool output without model participation."""

    kind = data.get("type")
    if kind == "portfolio":
        portfolio = data["portfolio"]
        return (
            f"Watch-only portfolio for {portfolio['address']} on chain "
            f"{portfolio['chain_id']} (block {portfolio['as_of_block']})."
        )
    if kind == "balance":
        amount = data["amount"]
        return (
            f"{data['asset_id']} balance: {amount['base_units']} base units "
            f"(decimals {amount['decimals']})."
        )
    if kind == "allowances":
        return (
            "Current token allowances:"
            if data["allowances"]
            else "No allowances set."
        )
    if kind == "registry":
        return "Canonical registry (the trusted id to address mapping):"
    raise InferenceError(f"unsupported verified result type: {kind!r}")


def validate_grounded_message(
    message: str, verified_result: dict[str, Any], deterministic_summary: str
) -> str:
    """Reject invented facts or claims that the wallet executed an action."""

    if _EXECUTION_CLAIM.search(message):
        raise InferenceError("model narration claims wallet execution")
    evidence = (
        json.dumps(verified_result, sort_keys=True, separators=(",", ":"))
        + " "
        + deterministic_summary
    ).casefold()
    supported = {
        *[value.casefold() for value in _CANONICAL_ID.findall(evidence)],
        *[value.casefold() for value in _ADDRESS.findall(evidence)],
        *_NUMBER.findall(evidence),
    }
    mentioned = {
        *[value.casefold() for value in _CANONICAL_ID.findall(message)],
        *[value.casefold() for value in _ADDRESS.findall(message)],
        *_NUMBER.findall(message.casefold()),
    }
    unsupported = sorted(mentioned - supported)
    if unsupported:
        raise InferenceError(
            f"model narration contains unsupported typed facts: {unsupported}"
        )
    return message
