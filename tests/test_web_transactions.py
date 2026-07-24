"""Adversarial and end-to-end web tests for the Phase 8 transaction surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentic_wallet.account_state import RelevantAccountState
from agentic_wallet.chain_metadata import normalize_transaction_hash
from agentic_wallet.registry import BASE_REGISTRY
from agentic_wallet.signer.capability import create_approval_capability
from agentic_wallet.signer_outcome import (
    SignerOutcome,
    SignerOutcomeCode,
    SignerOutcomeStatus,
)
from agentic_wallet.transaction_store import TransactionStore
from agentic_wallet.web.chat import DemoChatAgent
from agentic_wallet.web.transactions import BrowserSessionStore, TransactionController


SENDER = "0x1111111111111111111111111111111111111111"
RECIPIENT = "0x3333333333333333333333333333333333333333"
TX_HASH = "0x" + "a" * 64
SIGNING_HASH = "0x" + "b" * 64
SECRET = b"transaction-web-test-secret-that-is-long-enough"


class FakeRpc:
    def __init__(self) -> None:
        self.chain_id = 8453
        self.nonce = 7
        self.block = 200
        self.balance = 10**18
        self.receipt = None
        self.calls: list[dict] = []
        self.code_blocks: list[object] = []
        self.call_blocks: list[object] = []

    async def require_expected_chain(self) -> int:
        return self.chain_id

    async def relevant_account_state(self, address: str) -> RelevantAccountState:
        assert address.lower() == SENDER.lower()
        return RelevantAccountState(
            chain_id=8453, address=SENDER, pending_nonce=self.nonce,
            balance=self.balance, block_number=self.block,
            block_hash="0x" + f"{self.block:064x}",
        )

    async def fee_data(self):
        return SimpleNamespace(base_fee_per_gas=2, max_priority_fee_per_gas=1)

    async def account_code(self, address: str, *, block: object = "latest") -> str:
        self.code_blocks.append(block)
        return "0x"

    async def estimate_gas(self, transaction: dict) -> int:
        self.calls.append(transaction)
        return 21_000

    async def eth_call(self, transaction: dict, *, block: object = "latest") -> str:
        self.calls.append(transaction)
        self.call_blocks.append(block)
        return "0x"

    async def transaction_receipt(self, transaction_hash: str):
        assert normalize_transaction_hash(transaction_hash) == TX_HASH
        return self.receipt


class FakeSigner:
    def __init__(
        self,
        *,
        status: SignerOutcomeStatus = SignerOutcomeStatus.SUBMITTED,
    ) -> None:
        self.calls: list[dict] = []
        self.status = status

    async def get_signer_address(self) -> str:
        return SENDER

    async def sign_and_submit_approved(
        self, *, envelope: dict, approval_capability: str
    ) -> SignerOutcome:
        self.calls.append({"envelope": envelope, "approval_capability": approval_capability})
        if self.status is SignerOutcomeStatus.RESIMULATION_REQUIRED:
            return SignerOutcome(
                status=self.status,
                code=SignerOutcomeCode.PENDING_NONCE_CHANGED,
                from_address=SENDER,
                envelope_digest=envelope_digest(envelope),
            )
        return SignerOutcome(
            status=self.status,
            code=(
                SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN
                if self.status is SignerOutcomeStatus.UNKNOWN
                else SignerOutcomeCode.SUBMITTED
            ),
            transaction_hash=TX_HASH,
            transaction_signing_hash=SIGNING_HASH,
            from_address=SENDER,
            envelope_digest=envelope_digest(envelope),
        )


def envelope_digest(envelope: dict) -> str:
    from agentic_wallet.schemas.approval import ApprovalEnvelope

    return ApprovalEnvelope.model_validate(envelope).digest()


def controller(
    rpc: FakeRpc | None = None,
    signer: FakeSigner | None = None,
    transaction_store: TransactionStore | None = None,
) -> TransactionController:
    return TransactionController(
        registry=BASE_REGISTRY, rpc=rpc or FakeRpc(), signer=signer or FakeSigner(),
        approval_capability_factory=create_approval_capability,
        approval_capability_secret=SECRET,
        transaction_store=(
            transaction_store
            if transaction_store is not None
            else TransactionStore(max_records=8)
        ),
        clock=lambda: 1_000,
    )


@pytest.mark.anyio
async def test_controller_requires_exact_approval_then_submits_safe_metadata_only():
    rpc, signer = FakeRpc(), FakeSigner()
    subject = controller(rpc, signer)
    review = await subject.propose_native_transfer(
        session_id="session-one", chain_id=8453, recipient=RECIPIENT,
        amount_base_units="1000000000000000",
    )
    assert review["state"] == "AWAITING_CONFIRMATION"
    assert review["recipient"].lower() == RECIPIENT.lower()
    assert review["simulation"]["success"] is True
    assert "raw_transaction" not in str(review)
    assert "capability" not in str(review)
    snapshot = {
        "blockHash": "0x" + f"{rpc.block:064x}",
        "requireCanonical": True,
    }
    assert rpc.code_blocks == [snapshot]
    assert rpc.call_blocks == [snapshot]
    assert review["snapshot_block_hash"] == snapshot["blockHash"]
    assert review["state_anchor"].startswith("sha256:")
    assert review["nonce"] == "7"
    assert review["transaction_type"] == 2
    assert review["calldata"] == "0x"
    assert review["access_list"] == []
    assert review["registry_digest"].startswith("sha256:")
    assert review["simulated_transaction_digest"].startswith("sha256:")

    with pytest.raises(Exception, match="exact envelope"):
        await subject.approve(
            session_id="session-one", workflow_id=review["workflow_id"],
            envelope_digest="sha256:" + "0" * 64,
        )
    approved = await subject.approve(
        session_id="session-one", workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )
    assert approved["state"] == "READY_TO_SIGN"
    submitted = await subject.submit(
        session_id="session-one", workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )
    assert submitted["status"] == "SUBMITTED"
    assert submitted["explorer_url"] == f"https://basescan.org/tx/{TX_HASH}"
    assert "raw_transaction" not in str(submitted)
    assert "approval_capability" not in str(submitted)
    assert len(signer.calls) == 1
    assert "raw_transaction" not in str(subject.store.records())


@pytest.mark.anyio
async def test_controller_drift_invalidates_approval_before_signer():
    rpc, signer = FakeRpc(), FakeSigner()
    subject = controller(rpc, signer)
    review = await subject.propose_native_transfer(
        session_id="session-one", chain_id=8453, recipient=RECIPIENT, amount_base_units="1"
    )
    await subject.approve(
        session_id="session-one", workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )
    rpc.nonce += 1
    with pytest.raises(Exception, match="nonce changed"):
        await subject.submit(
            session_id="session-one", workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )
    assert signer.calls == []


@pytest.mark.anyio
async def test_signer_side_drift_forces_resimulation_and_cannot_retry():
    rpc = FakeRpc()
    signer = FakeSigner(status=SignerOutcomeStatus.RESIMULATION_REQUIRED)
    subject = controller(rpc, signer)
    review = await subject.propose_native_transfer(
        session_id="session-one",
        chain_id=8453,
        recipient=RECIPIENT,
        amount_base_units="1",
    )
    await subject.approve(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    with pytest.raises(Exception, match="new simulation"):
        await subject.submit(
            session_id="session-one",
            workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )

    active = subject._workflows[review["workflow_id"]]
    assert active.workflow.state_machine.state.value == "SIMULATING"
    assert active.workflow.approval_guard.approved_digest is None
    with pytest.raises(Exception, match="READY_TO_SIGN"):
        await subject.submit(
            session_id="session-one",
            workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )
    assert len(signer.calls) == 1


@pytest.mark.anyio
async def test_ambiguous_broadcast_persists_unknown_hash_and_never_resigns():
    rpc = FakeRpc()
    signer = FakeSigner(status=SignerOutcomeStatus.UNKNOWN)
    subject = controller(rpc, signer)
    review = await subject.propose_native_transfer(
        session_id="session-one",
        chain_id=8453,
        recipient=RECIPIENT,
        amount_base_units="1",
    )
    await subject.approve(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    result = await subject.submit(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    assert result["status"] == "UNKNOWN"
    assert result["transaction_hash"] == TX_HASH
    assert result["explorer_url"] == f"https://basescan.org/tx/{TX_HASH}"
    assert result["error_code"] == "BROADCAST_RESULT_UNKNOWN"
    assert subject._workflows[
        review["workflow_id"]
    ].workflow.state_machine.state.value == "SUBMISSION_UNKNOWN"
    with pytest.raises(Exception, match="READY_TO_SIGN"):
        await subject.submit(
            session_id="session-one",
            workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )
    assert len(signer.calls) == 1
    assert subject.store.lookup_for_session("session-one", TX_HASH).status.value == "UNKNOWN"


@pytest.mark.anyio
async def test_lost_signer_response_recovers_durable_hash_without_resigning():
    class LostResponseSigner(FakeSigner):
        def __init__(self) -> None:
            super().__init__(status=SignerOutcomeStatus.SUBMITTED)
            self.recovered: SignerOutcome | None = None

        async def sign_and_submit_approved(
            self, *, envelope: dict, approval_capability: str
        ) -> SignerOutcome:
            self.recovered = await super().sign_and_submit_approved(
                envelope=envelope, approval_capability=approval_capability
            )
            raise RuntimeError("stdio response was lost")

        async def lookup_submission_outcome(
            self, envelope_digest_value: str
        ) -> SignerOutcome | None:
            assert self.recovered is not None
            assert self.recovered.envelope_digest == envelope_digest_value
            return self.recovered

    signer = LostResponseSigner()
    subject = controller(signer=signer)
    review = await subject.propose_native_transfer(
        session_id="session-one",
        chain_id=8453,
        recipient=RECIPIENT,
        amount_base_units="1",
    )
    await subject.approve(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    result = await subject.submit(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    assert result["status"] == "SUBMITTED"
    assert result["transaction_hash"] == TX_HASH
    assert len(signer.calls) == 1
    assert subject.store.lookup_for_session("session-one", TX_HASH) is not None


@pytest.mark.anyio
async def test_lost_signer_response_and_lookup_failure_is_terminal_unknown():
    class UnrecoverableResponseSigner(FakeSigner):
        async def sign_and_submit_approved(
            self, *, envelope: dict, approval_capability: str
        ) -> SignerOutcome:
            self.calls.append((envelope, approval_capability))
            raise RuntimeError("stdio response was lost")

        async def lookup_submission_outcome(
            self, envelope_digest_value: str
        ) -> SignerOutcome | None:
            raise RuntimeError("journal lookup was unavailable")

    signer = UnrecoverableResponseSigner()
    subject = controller(signer=signer)
    review = await subject.propose_native_transfer(
        session_id="session-one",
        chain_id=8453,
        recipient=RECIPIENT,
        amount_base_units="1",
    )
    await subject.approve(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    with pytest.raises(Exception, match="status is unknown.*must not be retried"):
        await subject.submit(
            session_id="session-one",
            workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )

    workflow = subject._workflows[review["workflow_id"]].workflow
    assert workflow.state_machine.state.value == "SUBMISSION_UNKNOWN"
    assert len(signer.calls) == 1
    with pytest.raises(Exception, match="READY_TO_SIGN"):
        await subject.submit(
            session_id="session-one",
            workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )
    assert len(signer.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("signer_status", "terminal_state"),
    [
        (SignerOutcomeStatus.SUBMITTED, "SUBMITTED"),
        (SignerOutcomeStatus.UNKNOWN, "SUBMISSION_UNKNOWN"),
    ],
)
async def test_app_store_failure_after_signer_outcome_cannot_resubmit(
    signer_status: SignerOutcomeStatus, terminal_state: str
):
    class FailingTransactionStore(TransactionStore):
        def record_submission(self, **kwargs):
            raise RuntimeError("injected app-state failure")

    signer = FakeSigner(status=signer_status)
    subject = controller(
        signer=signer,
        transaction_store=FailingTransactionStore(max_records=8),
    )
    review = await subject.propose_native_transfer(
        session_id="session-one",
        chain_id=8453,
        recipient=RECIPIENT,
        amount_base_units="1",
    )
    await subject.approve(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    result = await subject.submit(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    workflow = subject._workflows[review["workflow_id"]].workflow
    assert workflow.state_machine.state.value == terminal_state
    assert result["transaction_hash"] == TX_HASH
    assert result["explorer_url"] == f"https://basescan.org/tx/{TX_HASH}"
    assert result["app_state_saved"] is False
    assert result["storage_error_code"] == "APP_STATE_RECORD_FAILED"
    with pytest.raises(Exception, match="READY_TO_SIGN"):
        await subject.submit(
            session_id="session-one",
            workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )
    assert len(signer.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("recipient", "message"),
    [
        ("0x0000000000000000000000000000000000000000", "zero-address"),
        (SENDER, "self-transfers"),
    ],
)
async def test_controller_rejects_unsafe_native_recipient_classes(
    recipient: str, message: str
):
    subject = controller()
    with pytest.raises(Exception, match=message):
        await subject.propose_native_transfer(
            session_id="session-one",
            chain_id=8453,
            recipient=recipient,
            amount_base_units="1",
        )


@pytest.mark.anyio
async def test_controller_rejects_contract_recipient_without_trace_simulation():
    rpc = FakeRpc()

    async def contract_code(address: str, *, block: str = "latest") -> str:
        return "0x6000"

    rpc.account_code = contract_code
    subject = controller(rpc=rpc)
    with pytest.raises(Exception, match="trace-capable simulation"):
        await subject.propose_native_transfer(
            session_id="session-one",
            chain_id=8453,
            recipient=RECIPIENT,
            amount_base_units="1",
        )


@pytest.mark.anyio
async def test_controller_live_scope_rejects_non_base_before_rpc_use():
    rpc, signer = FakeRpc(), FakeSigner()
    subject = controller(rpc=rpc, signer=signer)

    with pytest.raises(Exception, match="signs only on Base"):
        await subject.propose_native_transfer(
            session_id="session-one",
            chain_id=1,
            recipient=RECIPIENT,
            amount_base_units="1",
        )

    assert rpc.calls == []


@pytest.mark.anyio
async def test_controller_hides_workflows_and_transactions_from_other_sessions():
    subject = controller()
    review = await subject.propose_native_transfer(
        session_id="session-one", chain_id=8453, recipient=RECIPIENT, amount_base_units="1"
    )
    with pytest.raises(Exception, match="not found"):
        await subject.approve(
            session_id="session-two", workflow_id=review["workflow_id"],
            envelope_digest=review["envelope_digest"],
        )
    await subject.approve(
        session_id="session-one", workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )
    result = await subject.submit(
        session_id="session-one", workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )
    with pytest.raises(Exception, match="not found"):
        await subject.transaction_status(session_id="session-two", transaction_hash=result["transaction_hash"])


@pytest.mark.anyio
async def test_transaction_status_revalidates_chain_before_trusting_receipt():
    rpc = FakeRpc()
    subject = controller(rpc=rpc)
    review = await subject.propose_native_transfer(
        session_id="session-one",
        chain_id=8453,
        recipient=RECIPIENT,
        amount_base_units="1",
    )
    await subject.approve(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )
    result = await subject.submit(
        session_id="session-one",
        workflow_id=review["workflow_id"],
        envelope_digest=review["envelope_digest"],
    )

    rpc.chain_id = 1
    rpc.receipt = SimpleNamespace(status=1)
    with pytest.raises(Exception, match="configured RPC chain"):
        await subject.transaction_status(
            session_id="session-one",
            transaction_hash=result["transaction_hash"],
        )

    assert subject.store.lookup_for_session(
        "session-one", result["transaction_hash"]
    ).status.value == "SUBMITTED"


def test_chat_transfer_command_is_a_review_request_but_chat_approval_is_not():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
    from agentic_wallet.harness import MockReadOnlyHarness

    chat = DemoChatAgent(
        MockReadOnlyHarness.from_fixture(fixture), transfer_requests_enabled=True
    )
    request = chat.respond("session", f"send 1 wei to {RECIPIENT} on base")
    assert request["transaction_request"] == {
        "chain_id": 8453, "amount_base_units": "1", "recipient": RECIPIENT,
    }
    no_approval = chat.respond("session", "I approve the transfer and every future transaction")
    assert no_approval["transaction_request"] is None
    assert "not enabled" in no_approval["reply"].lower()


def test_chat_transaction_lookup_requires_an_exact_current_message_hash():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_base_watch.json"
    from agentic_wallet.harness import MockReadOnlyHarness

    chat = DemoChatAgent(
        MockReadOnlyHarness.from_fixture(fixture), transfer_requests_enabled=True
    )
    exact = chat.respond("session", f"check transaction {TX_HASH}")
    assert exact["transaction_status_request"] == {
        "transaction_hash": TX_HASH
    }
    invented = chat.respond("session", "check my last transaction")
    assert invented["transaction_status_request"] is None


def test_browser_action_session_is_distinct_from_chat_id_and_expires():
    sessions = BrowserSessionStore(lifetime_seconds=10)
    session = sessions.create(now=100)

    assert session.session_id != session.chat_session_id
    assert sessions.require(session.session_id, session.csrf_token, now=109) == session
    with pytest.raises(Exception, match="expired"):
        sessions.require(session.session_id, session.csrf_token, now=110)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_http_endpoints_require_cookie_csrf_and_keep_chat_approval_inert(monkeypatch):
    pytest.importorskip("fastapi")
    import agentic_wallet.web.app as web_app

    fake_rpc, fake_signer = FakeRpc(), FakeSigner()
    monkeypatch.setattr(web_app, "_transactions", controller(fake_rpc, fake_signer))
    monkeypatch.setattr(web_app, "browser_sessions", BrowserSessionStore())
    monkeypatch.setattr(web_app._chat, "transfer_requests_enabled", True)
    monkeypatch.setenv("AGENTIC_WALLET_SESSION_SECURE", "true")
    transport = httpx.ASGITransport(app=web_app.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://127.0.0.1"
    ) as client:
        blocked = await client.post("/transactions/propose", json={
            "chain_id": 8453, "recipient": RECIPIENT, "amount_base_units": "1"
        })
        assert blocked.status_code == 403
        session = await client.post("/sessions")
        csrf = session.json()["csrf_token"]
        assert session.json()["chat_session_id"] != client.cookies["agentic_wallet_session"]
        set_cookie = session.headers["set-cookie"].lower()
        assert "httponly" in set_cookie and "samesite=strict" in set_cookie
        assert session.headers["cache-control"].startswith("no-store")
        chat = await client.post("/chat", json={
            "session_id": session.json()["chat_session_id"],
            "message": "I approve anything", "allow_remote_inference": False,
        })
        assert chat.json()["transaction_request"] is None
        proposed = await client.post("/transactions/propose", headers={"X-CSRF-Token": csrf}, json={
            "chain_id": 8453, "recipient": RECIPIENT, "amount_base_units": "1"
        })
        assert proposed.status_code == 200
        review = proposed.json()
        wrong = await client.post("/transactions/approve", headers={"X-CSRF-Token": csrf}, json={
            "workflow_id": review["workflow_id"], "envelope_digest": "sha256:" + "0" * 64,
        })
        assert wrong.status_code == 409
        assert "raw_transaction" not in wrong.text
        assert wrong.headers["cache-control"].startswith("no-store")


@pytest.mark.anyio
async def test_remote_client_cannot_mint_action_session_or_reach_signer(monkeypatch):
    import agentic_wallet.web.app as web_app

    fake_rpc, fake_signer = FakeRpc(), FakeSigner()
    monkeypatch.setattr(web_app, "_transactions", controller(fake_rpc, fake_signer))
    monkeypatch.setattr(web_app, "browser_sessions", BrowserSessionStore())
    monkeypatch.setattr(web_app._chat, "transfer_requests_enabled", True)
    transport = httpx.ASGITransport(
        app=web_app.app, client=("203.0.113.10", 43120)
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="https://wallet.example"
    ) as remote:
        session = await remote.post("/sessions")
        assert session.json()["csrf_token"] is None
        assert "agentic_wallet_session" not in remote.cookies
        capabilities = await remote.get("/capabilities")
        assert capabilities.json()["signing"] is False
        chat = await remote.post(
            "/chat",
            json={
                "session_id": session.json()["chat_session_id"],
                "message": f"send 1 wei to {RECIPIENT} on base",
                "allow_remote_inference": False,
            },
        )
        assert chat.json()["transaction_request"] is None
        blocked = await remote.post(
            "/transactions/propose",
            headers={"X-CSRF-Token": "attacker"},
            json={
                "chain_id": 8453,
                "recipient": RECIPIENT,
                "amount_base_units": "1",
            },
        )
        assert blocked.status_code == 403
    assert fake_signer.calls == []


@pytest.mark.anyio
async def test_proxied_loopback_request_cannot_enable_transaction_surface(monkeypatch):
    import agentic_wallet.web.app as web_app

    fake_rpc, fake_signer = FakeRpc(), FakeSigner()
    monkeypatch.setattr(web_app, "_transactions", controller(fake_rpc, fake_signer))
    monkeypatch.setattr(web_app, "browser_sessions", BrowserSessionStore())
    transport = httpx.ASGITransport(
        app=web_app.app, client=("127.0.0.1", 43120)
    )
    proxy_headers = {
        "Forwarded": "for=203.0.113.10;proto=https",
        "X-Forwarded-For": "203.0.113.10",
    }
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://127.0.0.1",
        headers=proxy_headers,
    ) as proxied:
        session = await proxied.post("/sessions")
        assert session.json()["csrf_token"] is None
        assert "agentic_wallet_session" not in proxied.cookies
        capabilities = await proxied.get("/capabilities")
        assert capabilities.json()["signing"] is False
        blocked = await proxied.post(
            "/transactions/propose",
            headers={"X-CSRF-Token": "attacker"},
            json={
                "chain_id": 8453,
                "recipient": RECIPIENT,
                "amount_base_units": "1",
            },
        )
        assert blocked.status_code == 403
    assert fake_signer.calls == []


def test_approval_card_renders_every_current_scope_preimage_binding():
    html = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentic_wallet"
        / "web"
        / "static"
        / "index.html"
    ).read_text(encoding="utf-8")
    for field in (
        "review.chain_id",
        "review.sender",
        "review.recipient",
        "review.amount_base_units",
        "review.nonce",
        "review.transaction_type",
        "review.gas_limit",
        "review.max_fee_per_gas",
        "review.max_priority_fee_per_gas",
        "review.maximum_gas_fee_base_units",
        "review.calldata",
        "review.access_list",
        "review.snapshot_block_hash",
        "review.state_anchor",
        "review.plan_digest",
        "review.registry_digest",
        "review.simulated_transaction_digest",
        "review.envelope_digest",
        "review.expires_at",
        "simulation.gas_used",
        "simulation.balance_changes",
        "policy.violations",
    ):
        assert field in html
    assert 'id="submit-transfer" disabled' in html
    assert 'approved.state !== "READY_TO_SIGN"' in html
