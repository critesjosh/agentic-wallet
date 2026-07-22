"""Deterministic unsigned transfer and one-route swap planning.

The model may select these narrow operations, but it never supplies addresses
or calldata. Canonical IDs are resolved through the pinned registry and all
arithmetic is integer base-unit arithmetic.
"""

from __future__ import annotations

import re
from typing import Any

from eth_utils import is_checksum_address

from .digest import canonical_digest
from .harness import HarnessError, MockReadOnlyHarness
from .registry import RegistryError
from .schemas.common import Amount
from .schemas.quote import SwapQuote
from .schemas.transaction_plan import AssetDelta, TransactionPlan

APPROVED_ROUTER_ID = "base:fixture-swap-router"
_ERC20_TRANSFER_SELECTOR = "a9059cbb"
_MOCK_SWAP_SELECTOR = "5a19b9c3"
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


class PlanningError(RuntimeError):
    """Raised when deterministic planning rejects an unsafe request."""


def _address(value: str) -> str:
    if not _ADDRESS_RE.fullmatch(value):
        raise PlanningError("recipient must be a 20-byte EVM address")
    hex_part = value[2:]
    has_lower = any(character.isalpha() and character.islower() for character in hex_part)
    has_upper = any(character.isalpha() and character.isupper() for character in hex_part)
    if has_lower and has_upper and not is_checksum_address(value):
        raise PlanningError("mixed-case recipient has an invalid EIP-55 checksum")
    return value.lower()


def _require_amount(amount: Amount, decimals: int, *, name: str) -> int:
    if amount.decimals != decimals:
        raise PlanningError(f"{name} decimals do not match the registry")
    value = int(amount.base_units)
    if value <= 0:
        raise PlanningError(f"{name} must be greater than zero")
    return value


def _require_balance(balance: Amount, needed: int, *, asset_id: str) -> None:
    if int(balance.base_units) < needed:
        raise PlanningError(f"insufficient balance for {asset_id}")


def _erc20_transfer_calldata(recipient: str, amount: int) -> str:
    encoded_recipient = recipient[2:].rjust(64, "0")
    encoded_amount = f"{amount:064x}"
    return f"0x{_ERC20_TRANSFER_SELECTOR}{encoded_recipient}{encoded_amount}"


def _plan_id(payload: dict[str, Any]) -> str:
    return canonical_digest(payload)


def build_transfer_plan(
    harness: MockReadOnlyHarness,
    *,
    asset_id: str,
    amount: Amount,
    recipient: str,
    gas_reserve: Amount,
) -> TransactionPlan:
    """Build an unsigned native/ERC-20 transfer from trusted state."""

    recipient = _address(recipient)
    asset = harness.registry.resolve(asset_id)
    if asset.chain_id != harness.portfolio.chain_id:
        raise PlanningError("asset chain does not match the wallet chain")
    amount_value = _require_amount(amount, asset.decimals, name="amount")
    gas_value = _require_amount(gas_reserve, 18, name="gas reserve")
    sender = _address(harness.portfolio.address)
    native_balance = harness.get_native_balance()

    if asset_id == "base:native":
        _require_balance(
            native_balance, amount_value + gas_value, asset_id=asset_id
        )
        to_address = recipient
        value = amount
        calldata = "0x"
    else:
        try:
            token_balance = harness.get_token_balance(asset_id)
        except HarnessError as exc:
            raise PlanningError(str(exc)) from exc
        _require_balance(token_balance, amount_value, asset_id=asset_id)
        _require_balance(native_balance, gas_value, asset_id="base:native")
        to_address = _address(asset.address)
        value = Amount(base_units="0", decimals=18)
        calldata = _erc20_transfer_calldata(recipient, amount_value)

    payload = {
        "chain_id": asset.chain_id,
        "kind": "transfer",
        "from_address": sender,
        "to_address": to_address,
        "recipient_address": recipient,
        "asset_id": asset_id,
        "value": value.model_dump(),
        "calldata": calldata,
        "amount": amount.model_dump(),
        "gas_reserve": gas_reserve.model_dump(),
    }
    return TransactionPlan(
        plan_id=_plan_id(payload),
        chain_id=asset.chain_id,
        kind="transfer",
        from_address=sender,
        to_address=to_address,
        recipient_address=recipient,
        asset_id=asset_id,
        value=value,
        calldata=calldata,
        expected_outgoing=[AssetDelta(asset_id=asset_id, amount=amount)],
        gas_reserve=gas_reserve,
    )


def quote_payload(quote: SwapQuote) -> dict[str, Any]:
    return quote.model_dump(exclude={"quote_id"}, mode="json")


def make_mock_quote(
    *,
    chain_id: int,
    input_asset_id: str,
    output_asset_id: str,
    amount_in: Amount,
    amount_out: Amount,
    max_slippage_bps: int,
    issued_at_block: int,
    expires_at: int,
) -> SwapQuote:
    """Create a deterministic fixture quote for the one approved route."""

    fields = {
        "chain_id": chain_id,
        "input_asset_id": input_asset_id,
        "output_asset_id": output_asset_id,
        "amount_in": amount_in.model_dump(mode="json"),
        "amount_out": amount_out.model_dump(mode="json"),
        "router_id": APPROVED_ROUTER_ID,
        "max_slippage_bps": max_slippage_bps,
        "issued_at_block": issued_at_block,
        "expires_at": expires_at,
    }
    return SwapQuote(quote_id=canonical_digest(fields), **fields)


def build_swap_plan(
    harness: MockReadOnlyHarness,
    *,
    quote: SwapQuote,
    now: int,
    gas_reserve: Amount,
) -> TransactionPlan:
    """Build an unsigned plan for the single pinned mock swap route."""

    if quote.quote_id != canonical_digest(quote_payload(quote)):
        raise PlanningError("quote integrity check failed")
    if now >= quote.expires_at:
        raise PlanningError("quote expired")
    if quote.chain_id != harness.portfolio.chain_id:
        raise PlanningError("quote chain does not match the wallet chain")
    if quote.input_asset_id == quote.output_asset_id:
        raise PlanningError("swap assets must differ")
    if quote.router_id != APPROVED_ROUTER_ID:
        raise PlanningError("quote uses an unapproved router")

    input_asset = harness.registry.resolve(quote.input_asset_id)
    output_asset = harness.registry.resolve(quote.output_asset_id)
    try:
        router = harness.registry.resolve(quote.router_id)
    except RegistryError as exc:
        raise PlanningError(str(exc)) from exc
    if any(
        entry.chain_id != quote.chain_id
        for entry in (input_asset, output_asset, router)
    ):
        raise PlanningError("quote contains a cross-chain registry entry")

    amount_in = _require_amount(
        quote.amount_in, input_asset.decimals, name="input amount"
    )
    _require_amount(quote.amount_out, output_asset.decimals, name="output amount")
    gas_value = _require_amount(gas_reserve, 18, name="gas reserve")
    native_balance = harness.get_native_balance()
    if quote.input_asset_id == "base:native":
        _require_balance(
            native_balance, amount_in + gas_value, asset_id="base:native"
        )
        value = quote.amount_in
    else:
        _require_balance(
            harness.get_token_balance(quote.input_asset_id),
            amount_in,
            asset_id=quote.input_asset_id,
        )
        _require_balance(native_balance, gas_value, asset_id="base:native")
        value = Amount(base_units="0", decimals=18)

    sender = _address(harness.portfolio.address)
    route_hash = canonical_digest(quote.model_dump(mode="json")).split(":", 1)[1]
    calldata = f"0x{_MOCK_SWAP_SELECTOR}{route_hash}"
    payload = {
        "chain_id": quote.chain_id,
        "kind": "swap",
        "from_address": sender,
        "to_address": router.address.lower(),
        "recipient_address": sender,
        "asset_id": quote.input_asset_id,
        "value": value.model_dump(),
        "calldata": calldata,
        "amount_in": quote.amount_in.model_dump(),
        "amount_out": quote.amount_out.model_dump(),
        "gas_reserve": gas_reserve.model_dump(),
        "max_slippage_bps": quote.max_slippage_bps,
        "quote_id": quote.quote_id,
        "quote_expires_at": quote.expires_at,
    }
    return TransactionPlan(
        plan_id=_plan_id(payload),
        chain_id=quote.chain_id,
        kind="swap",
        from_address=sender,
        to_address=router.address.lower(),
        recipient_address=sender,
        asset_id=quote.input_asset_id,
        value=value,
        calldata=calldata,
        expected_outgoing=[
            AssetDelta(asset_id=quote.input_asset_id, amount=quote.amount_in)
        ],
        expected_incoming=[
            AssetDelta(asset_id=quote.output_asset_id, amount=quote.amount_out)
        ],
        gas_reserve=gas_reserve,
        max_slippage_bps=quote.max_slippage_bps,
        quote_id=quote.quote_id,
        quote_expires_at=quote.expires_at,
    )
