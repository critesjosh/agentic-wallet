"""Minimal typed JSON-RPC client used only by the signer process."""

from __future__ import annotations

from typing import Protocol

from eth_utils import keccak

from ..ethereum_rpc import EthereumJsonRpcClient
from ..schemas.signing import Eip1559Transaction
from .chains import NativeAssetError, native_asset_id_for_chain


class RpcError(RuntimeError):
    """Safe RPC failure message; URLs, credentials, and raw transactions stay private."""


class EthereumRpc(Protocol):
    async def chain_id(self) -> int: ...

    async def pending_nonce(self, address: str) -> int: ...

    async def relevant_state_anchor(self, address: str) -> str: ...

    async def verify_simulation(self, transaction: Eip1559Transaction) -> None: ...

    async def submit_raw_transaction(self, raw_transaction: bytes) -> str: ...


class JsonRpcEthereumRpc:
    """Signer adapter over the narrow, pinned application RPC client.

    This deliberately reuses the canonical relevant-account-state anchor used
    by Phase 8 approval generation, so the two sides cannot compare different
    anchor encodings for the same account state.
    """

    def __init__(self, rpc_url: str) -> None:
        self._client = EthereumJsonRpcClient(rpc_url)

    async def chain_id(self) -> int:
        return await self._client.chain_id()

    async def pending_nonce(self, address: str) -> int:
        return await self._client.pending_nonce(address)

    async def relevant_state_anchor(self, address: str) -> str:
        try:
            native_asset_id_for_chain(await self.chain_id())
        except NativeAssetError as error:
            raise RpcError("RPC returned an unsupported chain") from error
        return await self._client.relevant_state_anchor(address)

    async def verify_simulation(self, transaction: Eip1559Transaction) -> None:
        call = transaction.eth_account_dict()
        call["from"] = transaction.from_address
        if await self._client.account_code(transaction.to_address) != "0x":
            raise RpcError("contract recipients are not enabled")
        # A value transfer's eth_call and gas estimate must both succeed against
        # the current state; neither result can replace the approved preimage.
        await self._client.eth_call(call)
        estimated = await self._client.estimate_gas(call)
        if estimated > int(transaction.gas_limit):
            raise RpcError("live simulation exceeds approved gas limit")

    async def submit_raw_transaction(self, raw_transaction: bytes) -> str:
        transaction_hash = "0x" + keccak(raw_transaction).hex()
        return await self._client.send_raw_transaction(
            "0x" + raw_transaction.hex(), expected_transaction_hash=transaction_hash
        )
