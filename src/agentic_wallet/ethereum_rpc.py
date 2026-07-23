"""Strict, pinned-endpoint async JSON-RPC access for the Phase 8 signer path.

This module intentionally exposes only the narrow reads and broadcasts needed
by a native EIP-1559 transfer.  It does not accept an endpoint per request and
does not offer generic arbitrary-method execution to model-facing code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from ipaddress import ip_address
import json
import re
from typing import Any, Mapping, Protocol, Sequence, TypeAlias
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from eth_utils import to_checksum_address

from .account_state import RelevantAccountState
from .chain_metadata import get_chain_metadata, normalize_transaction_hash


_QUANTITY_RE = re.compile(r"^0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)$")
_DATA_RE = re.compile(r"^0x(?:[0-9a-fA-F]{2})*$")
_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")
BlockIdentifier: TypeAlias = str | Mapping[str, Any]


class EthereumRpcError(RuntimeError):
    """Base error for a failed or untrusted Ethereum JSON-RPC interaction."""


class RpcTransportError(EthereumRpcError):
    pass


class RpcResponseError(EthereumRpcError):
    pass


class RpcRemoteError(EthereumRpcError):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"RPC error {code}: {message}")


class RpcChainMismatchError(EthereumRpcError):
    pass


class RpcHashMismatchError(EthereumRpcError):
    pass


class AsyncJsonRpcTransport(Protocol):
    async def post_json(self, endpoint: str, payload: Mapping[str, Any]) -> Any:
        """Send JSON and return a decoded JSON value."""


class UrllibJsonRpcTransport:
    """Dependency-free production transport; tests should inject a fake."""

    def __init__(self, *, timeout_seconds: float = 15.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        self.timeout_seconds = timeout_seconds

    async def post_json(self, endpoint: str, payload: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self._post_json, endpoint, dict(payload))

    def _post_json(self, endpoint: str, payload: Mapping[str, Any]) -> Any:
        request = Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except (HTTPError, URLError, OSError) as exc:
            raise RpcTransportError("RPC transport failed") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RpcResponseError("RPC response was not valid JSON") from exc


@dataclass(frozen=True, slots=True)
class LatestBlock:
    number: int
    block_hash: str
    base_fee_per_gas: int


@dataclass(frozen=True, slots=True)
class FeeData:
    base_fee_per_gas: int
    max_priority_fee_per_gas: int


@dataclass(frozen=True, slots=True)
class TransactionReceipt:
    transaction_hash: str
    block_number: int
    block_hash: str
    status: int
    gas_used: int


def _validate_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("RPC endpoint must be a non-empty URL")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("RPC endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("RPC endpoint credentials must not be embedded in the URL")
    if parsed.scheme == "http":
        hostname = parsed.hostname
        is_loopback = hostname == "localhost"
        if hostname and not is_loopback:
            try:
                is_loopback = ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("non-loopback RPC endpoints must use HTTPS")
    return endpoint


def _quantity(value: Any, field: str) -> int:
    if not isinstance(value, str) or not _QUANTITY_RE.fullmatch(value):
        raise RpcResponseError(f"RPC {field} must be a canonical hex quantity")
    return int(value, 16)


def _data(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DATA_RE.fullmatch(value):
        raise RpcResponseError(f"RPC {field} must be 0x-prefixed even-length hex data")
    return value.lower()


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise RpcResponseError(f"RPC {field} must be a 32-byte transaction/block hash")
    return value.lower()


def _address(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("address must be a string")
    try:
        return to_checksum_address(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("address must be a valid EVM address") from exc


def _block_identifier(value: BlockIdentifier, field: str) -> str | dict[str, Any]:
    if isinstance(value, str):
        if value not in {"latest", "pending", "safe", "finalized"} and not _QUANTITY_RE.fullmatch(value):
            raise ValueError(
                f"{field} block must be a standard tag or canonical block quantity"
            )
        return value
    if not isinstance(value, Mapping) or set(value) != {
        "blockHash",
        "requireCanonical",
    }:
        raise ValueError(f"{field} block must be an EIP-1898 block-hash identifier")
    if value["requireCanonical"] is not True:
        raise ValueError(f"{field} block hash must require canonical ancestry")
    return {
        "blockHash": _hash(value["blockHash"], "block hash"),
        "requireCanonical": True,
    }


def _transaction_object(transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only the JSON-RPC transaction fields allowed for simulation/estimation."""

    if not isinstance(transaction, Mapping):
        raise ValueError("transaction must be a mapping")
    allowed = {
        "from", "to", "gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas",
        "value", "data", "nonce", "accessList", "type",
    }
    unexpected = set(transaction) - allowed
    if unexpected:
        raise ValueError(f"unsupported RPC transaction field(s): {', '.join(sorted(unexpected))}")
    result: dict[str, Any] = {}
    for key, value in transaction.items():
        if key in {"from", "to"}:
            result[key] = _address(value)
        elif key == "data":
            result[key] = _data(value, "transaction data")
        elif key in {"gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas", "value", "nonce"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"transaction {key} must be a non-negative integer")
            result[key] = hex(value)
        elif key == "type":
            if value != 2:
                raise ValueError("only EIP-1559 type 2 transaction simulation is allowed")
            result[key] = "0x2"
        else:  # accessList is structurally checked by the deterministic preimage schema.
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                raise ValueError("transaction accessList must be a sequence")
            result[key] = list(value)
    return result


class EthereumJsonRpcClient:
    """A narrow JSON-RPC client permanently bound to one configured endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        transport: AsyncJsonRpcTransport | None = None,
        expected_chain_id: int | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        if expected_chain_id is not None and (
            isinstance(expected_chain_id, bool)
            or not isinstance(expected_chain_id, int)
            or expected_chain_id <= 0
        ):
            raise ValueError("expected chain ID must be a positive integer")
        if expected_chain_id is not None:
            # An endpoint used for a live Phase 8 workflow must be bound to the
            # same small code-owned chain allowlist as its explorer metadata.
            get_chain_metadata(expected_chain_id)
        self._expected_chain_id = expected_chain_id
        self._transport = transport or UrllibJsonRpcTransport()
        self._next_request_id = 0
        self._request_lock = asyncio.Lock()

    @property
    def endpoint(self) -> str:
        """Configured endpoint; there is intentionally no per-call override."""

        return self._endpoint

    async def _request(self, method: str, params: list[Any]) -> Any:
        async with self._request_lock:
            self._next_request_id += 1
            request_id = self._next_request_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            response = await self._transport.post_json(self._endpoint, payload)
        except EthereumRpcError:
            raise
        except Exception as exc:
            raise RpcTransportError("RPC transport failed") from exc
        if not isinstance(response, Mapping):
            raise RpcResponseError("RPC response must be an object")
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise RpcResponseError("RPC response has an invalid JSON-RPC envelope")
        has_result = "result" in response
        has_error = "error" in response
        if has_result == has_error:
            raise RpcResponseError("RPC response must contain exactly one of result or error")
        if has_error:
            error = response["error"]
            if not isinstance(error, Mapping) or isinstance(error.get("code"), bool) or not isinstance(error.get("code"), int) or not isinstance(error.get("message"), str):
                raise RpcResponseError("RPC error has an invalid shape")
            raise RpcRemoteError(error["code"], error["message"])
        return response["result"]

    async def chain_id(self) -> int:
        return _quantity(await self._request("eth_chainId", []), "chain ID")

    async def require_expected_chain(self) -> int:
        chain_id = await self.chain_id()
        if self._expected_chain_id is not None and chain_id != self._expected_chain_id:
            raise RpcChainMismatchError(
                f"RPC chain ID {chain_id} did not match configured chain {self._expected_chain_id}"
            )
        get_chain_metadata(chain_id)
        return chain_id

    async def signer_balance(
        self, address: str, *, block: BlockIdentifier = "latest"
    ) -> int:
        return _quantity(
            await self._request(
                "eth_getBalance",
                [_address(address), _block_identifier(block, "balance")],
            ),
            "balance",
        )

    async def account_code(
        self, address: str, *, block: BlockIdentifier = "latest"
    ) -> str:
        """Return validated bytecode for an address at a trusted block tag."""

        return _data(
            await self._request(
                "eth_getCode", [_address(address), _block_identifier(block, "code")]
            ),
            "account code",
        )

    async def pending_nonce(self, address: str) -> int:
        return _quantity(await self._request("eth_getTransactionCount", [_address(address), "pending"]), "pending nonce")

    async def latest_block(self) -> LatestBlock:
        result = await self._request("eth_getBlockByNumber", ["latest", False])
        if not isinstance(result, Mapping):
            raise RpcResponseError("latest block must be an object")
        try:
            return LatestBlock(
                number=_quantity(result["number"], "block number"),
                block_hash=_hash(result["hash"], "block hash"),
                base_fee_per_gas=_quantity(result["baseFeePerGas"], "base fee"),
            )
        except KeyError as exc:
            raise RpcResponseError("latest block omitted a required EIP-1559 field") from exc

    async def fee_data(self) -> FeeData:
        latest, priority = await asyncio.gather(
            self.latest_block(), self._request("eth_maxPriorityFeePerGas", [])
        )
        return FeeData(
            base_fee_per_gas=latest.base_fee_per_gas,
            max_priority_fee_per_gas=_quantity(priority, "priority fee"),
        )

    async def eth_call(
        self, transaction: Mapping[str, Any], *, block: BlockIdentifier = "latest"
    ) -> str:
        return _data(
            await self._request(
                "eth_call",
                [_transaction_object(transaction), _block_identifier(block, "call")],
            ),
            "call result",
        )

    async def estimate_gas(self, transaction: Mapping[str, Any]) -> int:
        return _quantity(await self._request("eth_estimateGas", [_transaction_object(transaction)]), "gas estimate")

    async def send_raw_transaction(
        self, raw_transaction: str, *, expected_transaction_hash: str
    ) -> str:
        """Broadcast an already-signed transaction and enforce its local hash."""

        raw = _data(raw_transaction, "raw transaction")
        if raw == "0x":
            raise ValueError("raw transaction must not be empty")
        expected = normalize_transaction_hash(expected_transaction_hash)
        actual = _hash(await self._request("eth_sendRawTransaction", [raw]), "transaction hash")
        if actual != expected:
            raise RpcHashMismatchError("RPC returned a transaction hash different from the signed transaction")
        return actual

    async def transaction_receipt(self, transaction_hash: str) -> TransactionReceipt | None:
        expected = normalize_transaction_hash(transaction_hash)
        result = await self._request("eth_getTransactionReceipt", [expected])
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise RpcResponseError("transaction receipt must be an object or null")
        try:
            receipt_hash = _hash(result["transactionHash"], "receipt transaction hash")
            if receipt_hash != expected:
                raise RpcHashMismatchError("receipt transaction hash did not match request")
            status = _quantity(result["status"], "receipt status")
            if status not in {0, 1}:
                raise RpcResponseError("receipt status must be 0x0 or 0x1")
            return TransactionReceipt(
                transaction_hash=receipt_hash,
                block_number=_quantity(result["blockNumber"], "receipt block number"),
                block_hash=_hash(result["blockHash"], "receipt block hash"),
                status=status,
                gas_used=_quantity(result["gasUsed"], "receipt gas used"),
            )
        except KeyError as exc:
            raise RpcResponseError("transaction receipt omitted a required field") from exc

    async def relevant_account_state(self, address: str) -> RelevantAccountState:
        """Read transaction-relevant account facts plus block provenance."""

        chain_id = await self.require_expected_chain()
        latest = await self.latest_block()
        # Bind balance to the concrete block read above, while pending nonce
        # deliberately includes mempool state and is therefore read at pending.
        snapshot = {
            "blockHash": latest.block_hash,
            "requireCanonical": True,
        }
        balance, nonce = await asyncio.gather(
            self.signer_balance(address, block=snapshot),
            self.pending_nonce(address),
        )
        return RelevantAccountState(
            chain_id=chain_id,
            address=_address(address),
            pending_nonce=nonce,
            balance=balance,
            block_number=latest.number,
            block_hash=latest.block_hash,
        )

    async def relevant_state_anchor(self, address: str) -> str:
        """Return the canonical state-anchor string used in approval checks."""

        return (await self.relevant_account_state(address)).state_anchor

    # Explicit read-oriented aliases keep call sites legible without widening
    # the client into a generic RPC interface.
    get_chain_id = chain_id
    get_signer_balance = signer_balance
    get_account_code = account_code
    get_pending_nonce = pending_nonce
    get_latest_block = latest_block
    get_fee_data = fee_data
    get_transaction_receipt = transaction_receipt
    read_relevant_account_state = relevant_account_state


# Descriptive alias used by callers and documentation that say "Ethereum RPC".
EthereumRpcClient = EthereumJsonRpcClient
