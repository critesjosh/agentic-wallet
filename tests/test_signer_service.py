"""Focused tests for the private signer boundary; no persistent test key exists."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.typed_transactions import TypedTransaction
from eth_utils import keccak

from agentic_wallet.schemas.approval import ApprovalEnvelope
from agentic_wallet.schemas.common import Amount
from agentic_wallet.schemas.policy import PolicyResult
from agentic_wallet.schemas.signing import Eip1559Transaction
from agentic_wallet.schemas.simulation_result import SimulationResult
from agentic_wallet.schemas.transaction_plan import AssetDelta, TransactionPlan
from agentic_wallet.signer.capability import (
    CapabilityError,
    create_approval_capability,
    decode_approval_hmac_key,
)
from agentic_wallet.signer.capability_store import (
    AtomicFileCapabilityUseStore,
    CapabilityAlreadyUsed,
    InMemoryCapabilityUseStore,
)
from agentic_wallet.signer.client import (
    SignerClientError,
    StdioSignerClient,
    signer_child_environment,
)
from agentic_wallet.signer.key_store import KeyStoreError, require_secure_keyring_backend
from agentic_wallet.signer.outcome_store import (
    AtomicFileOutcomeStore,
    InMemoryOutcomeStore,
    OutcomeStoreError,
)
from agentic_wallet.signer.server import create_signer_server
from agentic_wallet.signer.service import SignerDenied, SignerService
from agentic_wallet.signer_outcome import (
    SignerOutcome,
    SignerOutcomeCode,
    SignerOutcomeStatus,
)

TEST_PRIVATE_KEY = "0x" + "11" * 32
SIGNER = Account.from_key(TEST_PRIVATE_KEY).address
RECIPIENT = "0x3333333333333333333333333333333333333333"
SECRET = b"capability-test-secret-must-be-32-bytes"
NOW = 1_000
ANCHOR = "base:21000001:0xabc"


class FakeKeyStore:
    def __init__(self) -> None:
        self.loads = 0

    def load_private_key(self) -> str:
        self.loads += 1
        return TEST_PRIVATE_KEY


class FakeRpc:
    def __init__(
        self,
        *,
        nonce: int = 7,
        anchor: str = ANCHOR,
        chain_id: int = 8453,
        preflight_error: bool = False,
        submission: str = "success",
    ) -> None:
        self.nonce = nonce
        self.anchor = anchor
        self._chain_id = chain_id
        self.preflight_error = preflight_error
        self.submission = submission
        self.simulations = 0
        self.submissions = 0

    async def chain_id(self) -> int:
        return self._chain_id

    async def pending_nonce(self, address: str) -> int:
        assert address.lower() == SIGNER.lower()
        return self.nonce

    async def relevant_state_anchor(self, address: str) -> str:
        assert address.lower() == SIGNER.lower()
        return self.anchor

    async def verify_simulation(self, transaction: Eip1559Transaction) -> None:
        assert transaction.data == "0x"
        self.simulations += 1
        if self.preflight_error:
            raise RuntimeError("provider detail must not escape")

    async def submit_raw_transaction(self, raw_transaction: bytes) -> str:
        self.submissions += 1
        if self.submission == "error":
            raise RuntimeError("ambiguous provider response")
        if self.submission == "mismatch":
            return "0x" + "f" * 64
        return "0x" + keccak(raw_transaction).hex()


def _envelope(*, allowed: bool = True, chain_id: int = 8453) -> ApprovalEnvelope:
    native_asset_id = "base:sepolia-native" if chain_id == 84532 else "base:native"
    value = Amount(base_units="123", decimals=18)
    plan = TransactionPlan(
        plan_id="native-transfer-1",
        chain_id=chain_id,
        kind="transfer",
        from_address=SIGNER,
        to_address=RECIPIENT,
        recipient_address=RECIPIENT,
        asset_id=native_asset_id,
        value=value,
        calldata="0x",
        expected_outgoing=[AssetDelta(asset_id=native_asset_id, amount=value)],
        gas_reserve=Amount(base_units="1", decimals=18),
    )
    simulation = SimulationResult(
        plan_id=plan.plan_id,
        success=True,
        block=21_000_001,
        gas_used=21_000,
    )
    transaction = Eip1559Transaction(
        chain_id=chain_id,
        nonce="7",
        max_priority_fee_per_gas="1",
        max_fee_per_gas="2",
        gas_limit="21000",
        from_address=SIGNER,
        to_address=RECIPIENT,
        value="123",
        data="0x",
    )
    simulation = simulation.model_copy(
        update={"transaction_digest": transaction.digest()}
    )
    return ApprovalEnvelope(
        chain_id=chain_id,
        plan=plan,
        simulation=simulation,
        policy=PolicyResult(allowed=allowed),
        expires_at=NOW + 100,
        state_anchor=ANCHOR,
        nonce=7,
        registry_digest="sha256:" + "0" * 64,
        simulated_transaction_digest=transaction.digest(),
        snapshot_block_hash="0x" + "ab" * 32,
        signing_transaction=transaction,
    )


def _service(
    rpc: FakeRpc,
    *,
    key_store: FakeKeyStore | None = None,
    capability_use_store: InMemoryCapabilityUseStore | None = None,
    outcome_store: InMemoryOutcomeStore | None = None,
) -> SignerService:
    return SignerService(
        key_store=key_store or FakeKeyStore(),
        rpc=rpc,
        approval_hmac_secret=SECRET,
        capability_use_store=capability_use_store or InMemoryCapabilityUseStore(),
        outcome_store=outcome_store or InMemoryOutcomeStore(),
        clock=lambda: NOW,
    )


def _capability(envelope: ApprovalEnvelope) -> str:
    return create_approval_capability(
        envelope_digest=envelope.digest(),
        envelope_expires_at=envelope.expires_at,
        secret=SECRET,
        now=NOW,
    )


def test_signer_revalidates_and_returns_only_safe_submission_metadata():
    rpc = FakeRpc()
    envelope = _envelope()
    outcome_store = InMemoryOutcomeStore()
    service = _service(rpc, outcome_store=outcome_store)
    result = asyncio.run(
        service.sign_and_submit_approved(
            envelope=envelope.model_dump(mode="json"), approval_capability=_capability(envelope)
        )
    )

    assert result.status == "SUBMITTED"
    assert result.from_address == SIGNER
    expected_signing_hash = "0x" + TypedTransaction.from_dict(
        envelope.signing_transaction.eth_account_dict()
    ).hash().hex()
    assert result.transaction_signing_hash == expected_signing_hash
    assert result.transaction_hash != result.transaction_signing_hash
    assert result.transaction_hash.startswith("0x")
    assert "raw_transaction" not in result.model_dump()
    assert "private_key" not in result.model_dump()
    assert rpc.simulations == 1
    assert rpc.submissions == 1
    assert service.lookup_submission_outcome(envelope.digest()) == result


def test_base_sepolia_is_not_in_the_live_signer_scope():
    rpc = FakeRpc(chain_id=84532)
    envelope = _envelope(chain_id=84532)
    key_store = FakeKeyStore()

    with pytest.raises(SignerDenied, match="unsupported signing chain"):
        asyncio.run(
            _service(rpc, key_store=key_store).sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"),
                approval_capability=_capability(envelope),
            )
        )

    assert key_store.loads == 0
    assert rpc.submissions == 0


def test_capability_is_claimed_before_key_access_and_cannot_be_replayed():
    rpc = FakeRpc()
    key_store = FakeKeyStore()
    use_store = InMemoryCapabilityUseStore()
    envelope = _envelope()
    capability = _capability(envelope)
    service = _service(rpc, key_store=key_store, capability_use_store=use_store)

    asyncio.run(
        service.sign_and_submit_approved(
            envelope=envelope.model_dump(mode="json"), approval_capability=capability
        )
    )
    with pytest.raises(SignerDenied, match="already used"):
        asyncio.run(
            service.sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"), approval_capability=capability
            )
        )

    assert key_store.loads == 1
    assert rpc.submissions == 1


def test_shared_capability_store_allows_one_concurrent_signer_process_only():
    rpc = FakeRpc()
    use_store = InMemoryCapabilityUseStore()
    envelope = _envelope()
    capability = _capability(envelope)
    services = [
        _service(rpc, capability_use_store=use_store),
        _service(rpc, capability_use_store=use_store),
    ]

    async def submit(service: SignerService) -> str:
        try:
            await service.sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"), approval_capability=capability
            )
            return "submitted"
        except SignerDenied as error:
            return str(error)

    async def submit_all() -> list[str]:
        return list(await asyncio.gather(*(submit(service) for service in services)))

    results = asyncio.run(submit_all())

    assert results.count("submitted") == 1
    assert any("already used" in result for result in results)
    assert rpc.submissions == 1


def test_atomic_file_capability_ledger_prevents_cross_instance_replay(tmp_path: Path):
    fingerprint = "a" * 64
    stores = [
        AtomicFileCapabilityUseStore(state_dir=tmp_path),
        AtomicFileCapabilityUseStore(state_dir=tmp_path),
    ]

    def claim(store: AtomicFileCapabilityUseStore) -> str:
        try:
            store.claim(capability_fingerprint=fingerprint, expires_at=NOW + 10, now=NOW)
            return "claimed"
        except CapabilityAlreadyUsed:
            return "replayed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, stores))

    assert sorted(results) == ["claimed", "replayed"]
    records = list((tmp_path / "agentic-wallet" / "signer-capabilities").glob("*"))
    assert [record.name for record in records] == [f"{fingerprint}.claim"]
    assert records[0].read_text(encoding="ascii") == f"{NOW + 10}\n"


def test_incomplete_atomic_claim_is_a_fail_closed_tombstone(tmp_path: Path):
    fingerprint = "b" * 64
    store = AtomicFileCapabilityUseStore(state_dir=tmp_path)
    directory = tmp_path / "agentic-wallet" / "signer-capabilities"
    # Model the interval after another process wins O_EXCL and before it writes.
    (directory / f"{fingerprint}.claim").touch(mode=0o600)

    with pytest.raises(CapabilityAlreadyUsed, match="already used"):
        store.claim(
            capability_fingerprint=fingerprint,
            expires_at=NOW + 10,
            now=NOW,
        )

    assert (directory / f"{fingerprint}.claim").exists()


def test_stdio_child_environment_forwards_only_keyring_and_signer_needs():
    source = {
        "HOME": "/home/tester",
        "PATH": "/usr/bin",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "LANG": "C.UTF-8",
        "AGENTIC_WALLET_SIGNER_RPC_URL": "https://rpc.example.invalid",
        "AGENTIC_WALLET_APPROVAL_HMAC_KEY": "x" * 32,
        "AWS_SECRET_ACCESS_KEY": "must-not-reach-signer",
        "UNRELATED_TOKEN": "must-not-reach-signer",
    }

    child = signer_child_environment(source)

    assert child == {
        key: value
        for key, value in source.items()
        if key
        in {
            "HOME",
            "PATH",
            "DBUS_SESSION_BUS_ADDRESS",
            "XDG_RUNTIME_DIR",
            "LANG",
            "AGENTIC_WALLET_SIGNER_RPC_URL",
            "AGENTIC_WALLET_APPROVAL_HMAC_KEY",
        }
    }


@pytest.mark.parametrize(
    ("rpc", "code"),
    [
        (FakeRpc(nonce=8), SignerOutcomeCode.PENDING_NONCE_CHANGED),
        (
            FakeRpc(anchor="base:21000002:0xdef"),
            SignerOutcomeCode.RELEVANT_STATE_CHANGED,
        ),
        (FakeRpc(chain_id=1), SignerOutcomeCode.RPC_CHAIN_CHANGED),
        (FakeRpc(preflight_error=True), SignerOutcomeCode.LIVE_PREFLIGHT_FAILED),
    ],
)
def test_live_chain_state_drift_returns_resimulation_without_key_or_submission(
    rpc: FakeRpc, code: SignerOutcomeCode
):
    envelope = _envelope()
    key_store = FakeKeyStore()
    use_store = InMemoryCapabilityUseStore()
    capability = _capability(envelope)
    outcome = asyncio.run(
        _service(
            rpc, key_store=key_store, capability_use_store=use_store
        ).sign_and_submit_approved(
            envelope=envelope.model_dump(mode="json"),
            approval_capability=capability,
        )
    )

    assert outcome.status is SignerOutcomeStatus.RESIMULATION_REQUIRED
    assert outcome.code is code
    assert outcome.transaction_hash is None
    assert outcome.transaction_signing_hash is None
    assert key_store.loads == 0
    assert rpc.submissions == 0

    # Freshness rejection burns the one-time capability.
    with pytest.raises(SignerDenied, match="already used"):
        asyncio.run(
            _service(
                rpc, key_store=key_store, capability_use_store=use_store
            ).sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"),
                approval_capability=capability,
            )
        )
    assert key_store.loads == 0
    assert rpc.submissions == 0


@pytest.mark.parametrize(
    ("submission", "code"),
    [
        ("error", SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN),
        ("mismatch", SignerOutcomeCode.BROADCAST_HASH_MISMATCH),
    ],
)
def test_ambiguous_broadcast_returns_local_hash_metadata_and_cannot_retry(
    submission: str, code: SignerOutcomeCode
):
    rpc = FakeRpc(submission=submission)
    use_store = InMemoryCapabilityUseStore()
    outcome_store = InMemoryOutcomeStore()
    envelope = _envelope()
    capability = _capability(envelope)
    service = _service(
        rpc,
        capability_use_store=use_store,
        outcome_store=outcome_store,
    )

    outcome = asyncio.run(
        service.sign_and_submit_approved(
            envelope=envelope.model_dump(mode="json"),
            approval_capability=capability,
        )
    )

    assert outcome.status is SignerOutcomeStatus.UNKNOWN
    assert outcome.code is code
    assert outcome.transaction_hash is not None
    assert outcome.transaction_signing_hash is not None
    assert "raw_transaction" not in outcome.model_dump()
    assert "private_key" not in outcome.model_dump()
    persisted = service.lookup_submission_outcome(envelope.digest())
    assert persisted is not None
    assert persisted.status is SignerOutcomeStatus.UNKNOWN
    assert persisted.transaction_hash == outcome.transaction_hash
    assert persisted.transaction_signing_hash == outcome.transaction_signing_hash
    # The journal is written before broadcast and therefore preserves the
    # conservative UNKNOWN code. The live caller can report a more specific
    # returned-hash mismatch without overstating what recovery can prove.
    assert persisted.code is SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN
    with pytest.raises(SignerDenied, match="already used"):
        asyncio.run(
            service.sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"),
                approval_capability=capability,
            )
        )
    assert rpc.submissions == 1


def test_outcome_store_failure_prevents_broadcast():
    class FailingOutcomeStore(InMemoryOutcomeStore):
        def record_unknown(self, outcome: SignerOutcome) -> None:
            raise OutcomeStoreError("disk detail must not escape")

    rpc = FakeRpc()
    envelope = _envelope()
    capability = _capability(envelope)
    service = _service(rpc, outcome_store=FailingOutcomeStore())

    with pytest.raises(SignerDenied, match="unavailable before broadcast"):
        asyncio.run(
            service.sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"),
                approval_capability=capability,
            )
        )

    assert rpc.submissions == 0
    with pytest.raises(SignerDenied, match="already used"):
        asyncio.run(
            service.sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"),
                approval_capability=capability,
            )
        )


def test_atomic_outcome_journal_contains_only_safe_metadata(tmp_path: Path):
    store = AtomicFileOutcomeStore(state_dir=tmp_path)
    envelope = _envelope()
    unknown = SignerOutcome(
        status=SignerOutcomeStatus.UNKNOWN,
        code=SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN,
        envelope_digest=envelope.digest(),
        from_address=SIGNER,
        transaction_hash="0x" + "1" * 64,
        transaction_signing_hash="0x" + "2" * 64,
    )
    submitted = unknown.model_copy(
        update={
            "status": SignerOutcomeStatus.SUBMITTED,
            "code": SignerOutcomeCode.SUBMITTED,
        }
    )

    store.record_unknown(unknown)
    reopened = AtomicFileOutcomeStore(state_dir=tmp_path)
    assert reopened.lookup(envelope.digest()) == unknown
    store.mark_submitted(submitted)
    assert reopened.lookup(envelope.digest()) == submitted

    journal = tmp_path / "agentic-wallet" / "signer-outcomes"
    record = next(journal.glob("*.json"))
    payload = record.read_text(encoding="utf-8")
    assert record.stat().st_mode & 0o077 == 0
    assert journal.stat().st_mode & 0o077 == 0
    assert "raw_transaction" not in payload
    assert "signature" not in payload
    assert "private_key" not in payload
    assert "approval_capability" not in payload


def test_hmac_environment_key_decoder_requires_urlsafe_base64_random_bytes():
    encoded = base64.urlsafe_b64encode(SECRET).rstrip(b"=").decode("ascii")

    assert decode_approval_hmac_key(encoded) == SECRET
    with pytest.raises(CapabilityError, match="URL-safe base64"):
        decode_approval_hmac_key("!" * 44)
    with pytest.raises(CapabilityError, match="at least 32 bytes"):
        decode_approval_hmac_key(base64.urlsafe_b64encode(b"short").decode("ascii"))


def test_stdio_client_validates_a_typed_signer_outcome():
    expected = SignerOutcome(
        status=SignerOutcomeStatus.RESIMULATION_REQUIRED,
        code=SignerOutcomeCode.PENDING_NONCE_CHANGED,
        envelope_digest="sha256:" + "0" * 64,
        from_address=SIGNER,
    )

    class FakeClient(StdioSignerClient):
        async def _call(self, name: str, arguments: dict) -> dict:
            assert name == "sign_and_submit_approved"
            return expected.model_dump(mode="json")

    observed = asyncio.run(
        FakeClient().sign_and_submit_approved(
            envelope={}, approval_capability="not-used-by-fake"
        )
    )

    assert observed == expected


def test_stdio_client_recovers_after_partial_sign_response():
    recovered = SignerOutcome(
        status=SignerOutcomeStatus.UNKNOWN,
        code=SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN,
        envelope_digest="sha256:" + "0" * 64,
        from_address=SIGNER,
        transaction_hash="0x" + "1" * 64,
        transaction_signing_hash="0x" + "2" * 64,
    )

    class PartialResponseClient(StdioSignerClient):
        async def _call(self, name: str, arguments: dict) -> dict:
            if name == "sign_and_submit_approved":
                return {"status": "UNKNOWN"}
            assert name == "lookup_submission_outcome"
            return {"found": True, "outcome": recovered.model_dump(mode="json")}

    client = PartialResponseClient()
    with pytest.raises(SignerClientError, match="invalid outcome"):
        asyncio.run(
            client.sign_and_submit_approved(
                envelope={}, approval_capability="not-used-by-fake"
            )
        )

    assert (
        asyncio.run(client.lookup_submission_outcome(recovered.envelope_digest))
        == recovered
    )


def test_capability_for_another_envelope_cannot_authorize_submission():
    rpc = FakeRpc()
    envelope = _envelope()
    other = envelope.model_copy(update={"expires_at": envelope.expires_at - 1})
    with pytest.raises(SignerDenied, match="not for this envelope"):
        asyncio.run(
            _service(rpc).sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"), approval_capability=_capability(other)
            )
        )
    assert rpc.submissions == 0


def test_unsafe_policy_is_rejected_before_key_or_rpc_use():
    rpc = FakeRpc()
    envelope = _envelope(allowed=False)
    with pytest.raises(SignerDenied, match="policy"):
        asyncio.run(
            _service(rpc).sign_and_submit_approved(
                envelope=envelope.model_dump(mode="json"), approval_capability=_capability(envelope)
            )
        )
    assert rpc.submissions == 0


def test_keyring_adapter_refuses_plaintext_fail_and_unknown_backends():
    for name in ("PlaintextKeyring", "FailKeyring", "NullKeyring", "UnexpectedKeyring"):
        backend_type = type(name, (), {})
        backend_type.__module__ = "test_keyring"
        with pytest.raises(KeyStoreError, match="secure OS store|approved"):
            require_secure_keyring_backend(backend_type())


def test_mcp_surface_has_exactly_three_safe_internal_tools():
    server = create_signer_server(_service(FakeRpc()))
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {
        "get_signer_address",
        "lookup_submission_outcome",
        "sign_and_submit_approved",
    }
    submission = next(tool for tool in tools if tool.name == "sign_and_submit_approved")
    serialized = submission.model_dump_json()
    assert "raw_transaction" not in serialized
    assert "private_key" not in serialized

    async def lookup_missing() -> dict:
        _, structured = await server.call_tool(
            "lookup_submission_outcome",
            {"envelope_digest": "sha256:" + "0" * 64},
        )
        return structured

    assert asyncio.run(lookup_missing()) == {"found": False}
