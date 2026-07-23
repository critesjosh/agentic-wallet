"""Focused tests for the isolated Phase 8 RPC and submission support layer."""

from __future__ import annotations

import asyncio
from dataclasses import fields
import threading

import pytest

from agentic_wallet.account_state import RelevantAccountState
from agentic_wallet.chain_metadata import (
    InvalidTransactionHashError,
    UnknownChainError,
    explorer_transaction_url,
    get_chain_metadata,
)
from agentic_wallet.ethereum_rpc import (
    EthereumJsonRpcClient,
    RpcChainMismatchError,
    RpcHashMismatchError,
    RpcResponseError,
)
from agentic_wallet.transaction_store import (
    TransactionRecord,
    TransactionStatus,
    TransactionStore,
    TransactionStoreError,
)


ADDRESS = "0x1111111111111111111111111111111111111111"
HASH_A = "0x" + "a" * 64
HASH_B = "0x" + "b" * 64
SIGNING_HASH = "0x" + "c" * 64
PLAN_DIGEST = "sha256:" + "1" * 64
ENVELOPE_DIGEST = "sha256:" + "2" * 64


class MockTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post_json(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        response = self.responses.pop(0)
        if callable(response):
            return response(payload)
        return response


def response(result):
    return lambda request: {"jsonrpc": "2.0", "id": request["id"], "result": result}


def test_trusted_chain_metadata_and_explorer_url_fail_closed():
    assert get_chain_metadata(1).native_asset_id == "ethereum:native"
    assert get_chain_metadata(8453).native_asset_id == "base:native"
    assert get_chain_metadata(84532).native_asset_id == "base:sepolia-native"
    assert explorer_transaction_url(1, HASH_A) == f"https://etherscan.io/tx/{HASH_A}"
    assert explorer_transaction_url(8453, HASH_A) == f"https://basescan.org/tx/{HASH_A}"
    assert explorer_transaction_url(84532, HASH_A) == f"https://sepolia.basescan.org/tx/{HASH_A}"

    with pytest.raises(UnknownChainError):
        explorer_transaction_url(137, HASH_A)
    with pytest.raises(InvalidTransactionHashError):
        explorer_transaction_url(8453, "https://attacker.invalid/tx/" + HASH_A)
    with pytest.raises(UnknownChainError):
        EthereumJsonRpcClient("https://rpc.example", expected_chain_id=137)


def test_rpc_rejects_malformed_response_and_chain_mismatch():
    malformed = MockTransport([{"jsonrpc": "2.0", "id": 999, "result": "0x1"}])
    client = EthereumJsonRpcClient("https://rpc.example", transport=malformed)
    with pytest.raises(RpcResponseError, match="envelope"):
        asyncio.run(client.chain_id())

    mismatch = MockTransport([response("0x1")])
    client = EthereumJsonRpcClient(
        "https://rpc.example", transport=mismatch, expected_chain_id=8453
    )
    with pytest.raises(RpcChainMismatchError):
        asyncio.run(client.require_expected_chain())


def test_rpc_requires_https_except_for_loopback():
    with pytest.raises(ValueError, match="must use HTTPS"):
        EthereumJsonRpcClient("http://rpc.example")
    assert EthereumJsonRpcClient("http://127.0.0.1:8545").endpoint.endswith(
        ":8545"
    )


def test_rpc_raw_broadcast_requires_returned_hash_match_and_pins_endpoint():
    transport = MockTransport([response(HASH_B)])
    client = EthereumJsonRpcClient("https://rpc.example/secret-route", transport=transport)
    with pytest.raises(RpcHashMismatchError):
        asyncio.run(client.send_raw_transaction("0x1234", expected_transaction_hash=HASH_A))
    assert transport.calls[0][0] == "https://rpc.example/secret-route"
    assert transport.calls[0][1]["method"] == "eth_sendRawTransaction"


def test_rpc_reads_relevant_account_anchor_from_validated_facts():
    transport = MockTransport(
        [
            response("0x2105"),
            response({"number": "0x10", "hash": HASH_A, "baseFeePerGas": "0x7"}),
            response("0x2a"),
            response("0x3"),
        ]
    )
    client = EthereumJsonRpcClient(
        "https://rpc.example", transport=transport, expected_chain_id=8453
    )
    state = asyncio.run(client.relevant_account_state(ADDRESS))
    assert state == RelevantAccountState(
        chain_id=8453,
        address=ADDRESS,
        pending_nonce=3,
        balance=42,
        block_number=16,
        block_hash=HASH_A,
    )
    assert state.anchor.startswith("sha256:")
    assert len(state.anchor) == 71
    assert transport.calls[2][1]["params"][1] == {
        "blockHash": HASH_A,
        "requireCanonical": True,
    }


def test_account_anchor_ignores_new_block_but_binds_balance_and_nonce():
    first = RelevantAccountState(
        chain_id=8453,
        address=ADDRESS,
        pending_nonce=3,
        balance=42,
        block_number=16,
        block_hash=HASH_A,
    )
    next_block = RelevantAccountState(
        chain_id=8453,
        address=ADDRESS,
        pending_nonce=3,
        balance=42,
        block_number=17,
        block_hash=HASH_B,
    )
    changed_balance = RelevantAccountState(
        chain_id=8453,
        address=ADDRESS,
        pending_nonce=3,
        balance=41,
        block_number=17,
        block_hash=HASH_B,
    )
    changed_nonce = RelevantAccountState(
        chain_id=8453,
        address=ADDRESS,
        pending_nonce=4,
        balance=42,
        block_number=17,
        block_hash=HASH_B,
    )

    assert first.anchor == next_block.anchor
    assert first.anchor != changed_balance.anchor
    assert first.anchor != changed_nonce.anchor


def test_rpc_fee_call_and_estimate_paths_use_typed_values_only():
    transport = MockTransport(
        [
            response({"number": "0x11", "hash": HASH_A, "baseFeePerGas": "0x7"}),
            response("0x2"),
            response("0xdeadbeef"),
            response("0x5208"),
        ]
    )
    client = EthereumJsonRpcClient("https://rpc.example", transport=transport)
    fees = asyncio.run(client.fee_data())
    assert fees.base_fee_per_gas == 7
    assert fees.max_priority_fee_per_gas == 2
    transaction = {"from": ADDRESS, "to": ADDRESS, "data": "0x1234", "value": 0, "type": 2}
    assert asyncio.run(client.eth_call(transaction)) == "0xdeadbeef"
    assert asyncio.run(client.estimate_gas(transaction)) == 21_000
    assert transport.calls[2][1]["params"][0]["value"] == "0x0"


def test_rpc_account_code_is_a_narrow_validated_read():
    transport = MockTransport([response("0x60016000")])
    client = EthereumJsonRpcClient("https://rpc.example", transport=transport)

    assert asyncio.run(client.account_code(ADDRESS)) == "0x60016000"
    assert transport.calls[0][1]["method"] == "eth_getCode"


def test_rpc_code_and_call_support_canonical_hash_pinning():
    transport = MockTransport([response("0x"), response("0x")])
    client = EthereumJsonRpcClient("https://rpc.example", transport=transport)
    block = {"blockHash": HASH_A, "requireCanonical": True}
    transaction = {
        "from": ADDRESS,
        "to": ADDRESS,
        "data": "0x",
        "value": 0,
        "type": 2,
    }

    assert asyncio.run(client.account_code(ADDRESS, block=block)) == "0x"
    assert asyncio.run(client.eth_call(transaction, block=block)) == "0x"
    assert transport.calls[0][1]["params"][1] == block
    assert transport.calls[1][1]["params"][1] == block

    with pytest.raises(ValueError, match="canonical ancestry"):
        asyncio.run(
            client.account_code(
                ADDRESS,
                block={"blockHash": HASH_A, "requireCanonical": False},
            )
        )


def test_rpc_rejects_malformed_quantity_result():
    client = EthereumJsonRpcClient(
        "https://rpc.example", transport=MockTransport([response("17")])
    )
    with pytest.raises(RpcResponseError, match="canonical hex quantity"):
        asyncio.run(client.pending_nonce(ADDRESS))


def test_receipt_rejects_hash_mismatch():
    client = EthereumJsonRpcClient(
        "https://rpc.example",
        transport=MockTransport(
            [
                response(
                    {
                        "transactionHash": HASH_B,
                        "blockNumber": "0x1",
                        "blockHash": HASH_A,
                        "status": "0x1",
                        "gasUsed": "0x5208",
                    }
                )
            ]
        ),
    )
    with pytest.raises(RpcHashMismatchError):
        asyncio.run(client.transaction_receipt(HASH_A))


def _store_record(store: TransactionStore, transaction_hash: str, *, session: str = "session-a"):
    return store.record_submission(
        session_id=session,
        workflow_id="workflow-a",
        plan_digest=PLAN_DIGEST,
        envelope_digest=ENVELOPE_DIGEST,
        chain_id=8453,
        sender=ADDRESS,
        transaction_hash=transaction_hash,
        signing_hash=SIGNING_HASH,
        now=10,
    )


def test_transaction_store_session_scoping_eviction_and_secret_exclusion():
    store = TransactionStore(max_records=2)
    _store_record(store, HASH_A, session="session-a")
    _store_record(store, HASH_B, session="session-b")
    hash_c = "0x" + "d" * 64
    _store_record(store, hash_c, session="session-c")

    assert store.lookup(HASH_A) is None
    assert store.lookup(HASH_B).session_id == "session-b"
    assert store.lookup_for_session("session-b", HASH_B).session_id == "session-b"
    assert store.lookup_for_session("session-a", HASH_B) is None
    assert [item.transaction_hash for item in store.records()] == [HASH_B, hash_c]
    assert store.lookup(hash_c).explorer_url == f"https://basescan.org/tx/{hash_c}"

    field_names = {field.name for field in fields(TransactionRecord)}
    forbidden = {"raw_transaction", "signature", "capability", "approval", "rpc_endpoint", "private_key"}
    assert not field_names & forbidden
    with pytest.raises(TypeError):
        TransactionRecord(raw_transaction="0x1234")  # type: ignore[call-arg]


def test_transaction_store_rejects_cross_session_hash_rebinding():
    store = TransactionStore(max_records=2)
    original = _store_record(store, HASH_A, session="session-a")

    with pytest.raises(
        TransactionStoreError, match="already bound to another record"
    ):
        _store_record(store, HASH_A, session="session-b")

    assert store.lookup(HASH_A) == original
    assert store.lookup_for_session("session-a", HASH_A) == original
    assert store.lookup_for_session("session-b", HASH_A) is None


def test_transaction_store_locking_keeps_concurrent_operations_bounded():
    store = TransactionStore(max_records=64)
    errors = []

    def worker(index: int) -> None:
        try:
            transaction_hash = "0x" + f"{index:064x}"
            _store_record(store, transaction_hash, session=f"session-{index}")
            store.update_status(
                transaction_hash,
                status=TransactionStatus.UNKNOWN,
                now=11,
                error_code="POST_BROADCAST_TIMEOUT",
            )
        except Exception as exc:  # test captures thread failures deterministically
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store) == 40
    assert all(record.status is TransactionStatus.UNKNOWN for record in store.records())
