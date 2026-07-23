"""Web-facing deterministic native-transfer workflow.

This module deliberately has no model inputs.  A chat/model may suggest that a
user open a transfer review, but the exact recipient, amount, approval digest,
and submit operation enter here through separate typed HTTP requests.  The
private signer only receives an already-approved envelope and a short-lived
capability; it never exposes a key, signature, or raw transaction to this
layer.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import secrets
import threading
import time
from typing import Any, Protocol

from eth_utils import to_checksum_address

from ..account_state import RelevantAccountState
from ..approval_guard import ApprovalInvalidated
from ..chain_metadata import (
    explorer_transaction_url,
    get_chain_metadata,
    normalize_transaction_hash,
)
from ..ethereum_rpc import BlockIdentifier, EthereumJsonRpcClient, TransactionReceipt
from ..harness import MockReadOnlyHarness
from ..planning import PlanningError, build_eip1559_transaction
from ..policy_engine import WalletPolicy
from ..registry import Registry
from ..schemas.common import Amount
from ..schemas.portfolio import Portfolio
from ..signer_outcome import SignerOutcome, SignerOutcomeStatus
from ..state_machine import TransitionError, WorkflowState
from ..transaction_store import TransactionStatus, TransactionStore
from ..unsigned_workflow import UnsignedTransactionWorkflow, UnsignedWorkflowError

_LIVE_CHAIN_ID = 8453


class TransactionFlowError(RuntimeError):
    """A safe, client-displayable reason that a transaction flow was rejected."""


class TransactionRpc(Protocol):
    async def require_expected_chain(self) -> int: ...

    async def relevant_account_state(self, address: str) -> RelevantAccountState: ...

    async def fee_data(self) -> Any: ...

    async def account_code(
        self, address: str, *, block: BlockIdentifier = "latest"
    ) -> str: ...

    async def estimate_gas(self, transaction: dict[str, Any]) -> int: ...

    async def eth_call(
        self, transaction: dict[str, Any], *, block: BlockIdentifier = "latest"
    ) -> str: ...

    async def transaction_receipt(self, transaction_hash: str) -> TransactionReceipt | None: ...


class ApprovedSigner(Protocol):
    async def get_signer_address(self) -> str: ...

    async def lookup_submission_outcome(
        self, envelope_digest: str
    ) -> SignerOutcome | None: ...

    async def sign_and_submit_approved(
        self, *, envelope: dict[str, Any], approval_capability: str
    ) -> SignerOutcome: ...


@dataclass(frozen=True, slots=True)
class BrowserSession:
    session_id: str
    chat_session_id: str
    csrf_token: str
    created_at: int
    expires_at: int


class BrowserSessionStore:
    """Bounded in-memory session/CSRF store for consequential endpoints."""

    def __init__(
        self, *, max_sessions: int = 256, lifetime_seconds: int = 14_400
    ) -> None:
        if max_sessions <= 0 or lifetime_seconds <= 0:
            raise ValueError("session capacity and lifetime must be positive")
        self._max_sessions = max_sessions
        self._lifetime_seconds = lifetime_seconds
        self._sessions: OrderedDict[str, BrowserSession] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _token() -> str:
        # URL-safe tokens contain only characters accepted by TransactionStore.
        return secrets.token_urlsafe(24)

    def create(self, *, now: int | None = None) -> BrowserSession:
        timestamp = int(time.time()) if now is None else now
        session = BrowserSession(
            session_id=self._token(),
            chat_session_id=self._token(),
            csrf_token=self._token(),
            created_at=timestamp,
            expires_at=timestamp + self._lifetime_seconds,
        )
        with self._lock:
            self._sessions[session.session_id] = session
            self._sessions.move_to_end(session.session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return session

    def require(
        self,
        session_id: str | None,
        csrf_token: str | None,
        *,
        now: int | None = None,
    ) -> BrowserSession:
        if not session_id or not csrf_token:
            raise TransactionFlowError("a browser session and CSRF token are required")
        timestamp = int(time.time()) if now is None else now
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise TransactionFlowError("browser session or CSRF token is invalid")
            if timestamp >= session.expires_at:
                self._sessions.pop(session_id, None)
                raise TransactionFlowError("browser session has expired")
            if not secrets.compare_digest(session.csrf_token, csrf_token):
                raise TransactionFlowError("browser session or CSRF token is invalid")
            self._sessions.move_to_end(session_id)
            return session


@dataclass(slots=True)
class _ActiveWorkflow:
    session_id: str
    workflow: UnsignedTransactionWorkflow
    sender: str


def _now() -> int:
    return int(time.time())


def _rpc_transaction(transaction: Any) -> dict[str, Any]:
    """Translate the typed preimage to the narrow RPC representation."""

    return {
        "from": transaction.from_address,
        "to": transaction.to_address,
        "value": int(transaction.value),
        "data": transaction.data,
        "nonce": int(transaction.nonce),
        "gas": int(transaction.gas_limit),
        "maxFeePerGas": int(transaction.max_fee_per_gas),
        "maxPriorityFeePerGas": int(transaction.max_priority_fee_per_gas),
        "type": 2,
        "accessList": [
            {"address": item.address, "storageKeys": item.storage_keys}
            for item in transaction.access_list
        ],
    }


class TransactionController:
    """Owns server-side workflows; browser clients receive only safe summaries."""

    def __init__(
        self,
        *,
        registry: Registry,
        rpc: TransactionRpc,
        signer: ApprovedSigner,
        approval_capability_factory: Any,
        approval_capability_secret: bytes,
        approval_ttl_seconds: int = 120,
        max_workflows: int = 128,
        transaction_store: TransactionStore | None = None,
        clock: Any = _now,
    ) -> None:
        if approval_ttl_seconds <= 0 or max_workflows <= 0:
            raise ValueError("approval TTL and workflow capacity must be positive")
        if len(approval_capability_secret) < 32:
            raise ValueError("approval capability secret must be at least 32 bytes")
        self.registry = registry
        self.rpc = rpc
        self.signer = signer
        self._create_capability = approval_capability_factory
        self._capability_secret = approval_capability_secret
        self.approval_ttl_seconds = approval_ttl_seconds
        self.store = (
            transaction_store
            if transaction_store is not None
            else TransactionStore()
        )
        self._clock = clock
        self._max_workflows = max_workflows
        self._workflows: OrderedDict[str, _ActiveWorkflow] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _workflow_id() -> str:
        return secrets.token_urlsafe(18)

    async def ready(self) -> bool:
        """Verify the isolated signer and configured RPC before advertising live use."""

        try:
            sender = to_checksum_address(await self.signer.get_signer_address())
            chain_id = await self.rpc.require_expected_chain()
            native = self.registry.native_asset(chain_id)
        except Exception:
            return False
        return (
            chain_id == _LIVE_CHAIN_ID
            and native.asset_id == get_chain_metadata(chain_id).native_asset_id
            and native.is_native
            and bool(sender)
        )

    async def _put_workflow(self, workflow_id: str, active: _ActiveWorkflow) -> None:
        async with self._lock:
            self._workflows[workflow_id] = active
            self._workflows.move_to_end(workflow_id)
            while len(self._workflows) > self._max_workflows:
                self._workflows.popitem(last=False)

    async def _require_workflow(self, session_id: str, workflow_id: str) -> _ActiveWorkflow:
        async with self._lock:
            active = self._workflows.get(workflow_id)
            if active is None or active.session_id != session_id:
                # Do not reveal whether another session owns this opaque ID.
                raise TransactionFlowError("transaction review was not found")
            self._workflows.move_to_end(workflow_id)
            return active

    @staticmethod
    def _summary(workflow_id: str, active: _ActiveWorkflow) -> dict[str, Any]:
        workflow = active.workflow
        envelope = workflow.envelope
        if envelope is None or workflow.plan is None or workflow.simulation is None or workflow.policy_result is None:
            raise TransactionFlowError("transaction review is incomplete")
        transaction = envelope.signing_transaction
        if transaction is None:
            raise TransactionFlowError("transaction review is not signing-bound")
        return {
            "workflow_id": workflow_id,
            "state": workflow.state_machine.state.value,
            "envelope_digest": envelope.digest(),
            "expires_at": envelope.expires_at,
            "plan_digest": envelope.plan.plan_id,
            "registry_digest": envelope.registry_digest,
            "simulated_transaction_digest": envelope.simulated_transaction_digest,
            "state_anchor": envelope.state_anchor,
            "snapshot_block_hash": envelope.snapshot_block_hash,
            "chain_id": envelope.chain_id,
            "chain_name": get_chain_metadata(envelope.chain_id).name,
            "sender": transaction.from_address,
            "recipient": transaction.to_address,
            "asset_id": workflow.plan.asset_id,
            "amount_base_units": workflow.plan.expected_outgoing[0].amount.base_units,
            "amount_decimals": workflow.plan.expected_outgoing[0].amount.decimals,
            "gas_limit": transaction.gas_limit,
            "max_fee_per_gas": transaction.max_fee_per_gas,
            "max_priority_fee_per_gas": transaction.max_priority_fee_per_gas,
            "nonce": transaction.nonce,
            "transaction_type": transaction.transaction_type,
            "calldata": transaction.data,
            "access_list": [item.model_dump() for item in transaction.access_list],
            "maximum_gas_fee_base_units": str(
                int(transaction.gas_limit) * int(transaction.max_fee_per_gas)
            ),
            "simulation": {
                "mode": "pinned_native_eoa_preflight",
                "block": workflow.simulation.block,
                "gas_used": workflow.simulation.gas_used,
                "success": workflow.simulation.success,
                "mismatch": workflow.simulation.mismatch,
                "balance_changes": [item.model_dump() for item in workflow.simulation.balance_changes],
            },
            "policy": workflow.policy_result.model_dump(),
        }

    async def propose_native_transfer(
        self,
        *,
        session_id: str,
        chain_id: int,
        recipient: str,
        amount_base_units: str,
    ) -> dict[str, Any]:
        """Build and simulate an exact native EIP-1559 preimage for review."""

        try:
            if chain_id != _LIVE_CHAIN_ID:
                raise TransactionFlowError(
                    "the live transaction proof of concept supports Base only"
                )
            metadata = get_chain_metadata(chain_id)
            actual_chain = await self.rpc.require_expected_chain()
            if actual_chain != metadata.chain_id:
                raise TransactionFlowError("configured RPC chain does not match request")
            native = self.registry.native_asset(chain_id)
            if not native.is_native or native.decimals != 18:
                raise TransactionFlowError("trusted native asset is unavailable for this chain")
            sender = to_checksum_address(await self.signer.get_signer_address())
            state = await self.rpc.relevant_account_state(sender)
            if state.chain_id != chain_id:
                raise TransactionFlowError("account state chain does not match request")
            amount = Amount(base_units=amount_base_units, decimals=native.decimals)
            if int(amount.base_units) <= 0:
                raise TransactionFlowError("transfer amount must be greater than zero")
            normalized_recipient = to_checksum_address(recipient)
            if normalized_recipient == "0x0000000000000000000000000000000000000000":
                raise TransactionFlowError("zero-address transfers are not enabled")
            if normalized_recipient.lower() == sender.lower():
                raise TransactionFlowError("self-transfers are not enabled")
            block_tag: BlockIdentifier = {
                "blockHash": state.block_hash,
                "requireCanonical": True,
            }
            if await self.rpc.account_code(
                normalized_recipient, block=block_tag
            ) != "0x":
                raise TransactionFlowError(
                    "contract recipients require a trace-capable simulation and are not enabled"
                )
            # Native value transfers have no calldata; estimate before binding
            # the final gas field, then bind the maximum EIP-1559 fee exactly.
            fees = await self.rpc.fee_data()
            priority = int(fees.max_priority_fee_per_gas)
            maximum = int(fees.base_fee_per_gas) * 2 + priority
            estimate_input = {
                "from": sender, "to": normalized_recipient, "value": int(amount.base_units),
                "data": "0x", "nonce": state.pending_nonce,
                "maxFeePerGas": maximum, "maxPriorityFeePerGas": priority,
                "type": 2, "accessList": [],
            }
            gas_limit = await self.rpc.estimate_gas(estimate_input)
            if gas_limit <= 0:
                raise TransactionFlowError("RPC returned an invalid gas estimate")
            gas_fee = Amount(
                base_units=str(gas_limit * maximum), decimals=native.decimals
            )
            portfolio = Portfolio(
                chain_id=chain_id, address=sender, native_balance=Amount(
                    base_units=str(state.balance), decimals=native.decimals
                ), as_of_block=state.block_number,
            )
            workflow = UnsignedTransactionWorkflow(
                MockReadOnlyHarness(portfolio, self.registry),
                WalletPolicy(chain_id=chain_id, wallet_address=sender),
                approval_ttl_seconds=self.approval_ttl_seconds,
            )
            plan = workflow.plan_transfer(
                asset_id=native.asset_id,
                amount=amount,
                recipient=normalized_recipient,
                gas_reserve=gas_fee,
            )
            transaction = build_eip1559_transaction(
                plan, nonce=state.pending_nonce, gas_limit=gas_limit,
                max_priority_fee_per_gas=priority, max_fee_per_gas=maximum,
            )
            # This confirms current EVM execution accepts the exact preimage.
            await self.rpc.eth_call(
                _rpc_transaction(transaction), block=block_tag
            )
            envelope = workflow.simulate_and_check(
                now=self._clock(), block=state.block_number,
                state_anchor=state.state_anchor,
                snapshot_block_hash=state.block_hash,
                nonce=state.pending_nonce,
                gas_used=gas_limit, gas_fee=gas_fee, signing_transaction=transaction,
            )
            if envelope is None:
                raise TransactionFlowError("simulation or policy rejected the transfer")
        except TransactionFlowError:
            raise
        except (PlanningError, UnsignedWorkflowError, ValueError) as error:
            raise TransactionFlowError(str(error)) from error
        except Exception as error:
            # RPC/signer implementation details (and endpoint data) remain server-side.
            raise TransactionFlowError("could not create a fresh transaction review") from error

        workflow_id = self._workflow_id()
        active = _ActiveWorkflow(session_id=session_id, workflow=workflow, sender=sender)
        await self._put_workflow(workflow_id, active)
        return self._summary(workflow_id, active)

    async def approve(
        self, *, session_id: str, workflow_id: str, envelope_digest: str
    ) -> dict[str, Any]:
        active = await self._require_workflow(session_id, workflow_id)
        try:
            active.workflow.record_user_approval(envelope_digest, now=self._clock())
        except (ApprovalInvalidated, UnsignedWorkflowError) as error:
            raise TransactionFlowError(str(error)) from error
        return self._summary(workflow_id, active)

    async def submit(
        self, *, session_id: str, workflow_id: str, envelope_digest: str
    ) -> dict[str, Any]:
        active = await self._require_workflow(session_id, workflow_id)
        workflow = active.workflow
        envelope = workflow.envelope
        if envelope is None or not secrets.compare_digest(envelope.digest(), envelope_digest):
            raise TransactionFlowError("submit requires the exact reviewed envelope digest")
        try:
            current = await self.rpc.relevant_account_state(active.sender)
            workflow.authorize_submission(
                now=self._clock(), state_anchor=current.state_anchor, nonce=current.pending_nonce
            )
        except (ApprovalInvalidated, UnsignedWorkflowError) as error:
            raise TransactionFlowError(str(error)) from error
        try:
            capability = self._create_capability(
                envelope_digest=envelope.digest(), envelope_expires_at=envelope.expires_at,
                secret=self._capability_secret, now=self._clock(), lifetime_seconds=60,
            )
            try:
                raw_outcome = await self.signer.sign_and_submit_approved(
                    envelope=envelope.model_dump(mode="json"),
                    approval_capability=capability,
                )
            except Exception as signing_error:
                # The stdio response can be lost after signing or broadcast.
                # Recover only signer-journaled safe metadata for this exact
                # envelope; never mint a second capability or sign again.
                try:
                    raw_outcome = await self.signer.lookup_submission_outcome(
                        envelope.digest()
                    )
                except Exception:
                    workflow.mark_submission_unknown_after_lookup_failure()
                    raise TransactionFlowError(
                        "the signer response and recovery lookup were unavailable; "
                        "submission status is unknown and this transaction must not "
                        "be retried"
                    ) from signing_error
                if raw_outcome is None:
                    workflow.mark_submission_unknown_after_lookup_failure()
                    raise TransactionFlowError(
                        "the signer response was lost before a recoverable hash was "
                        "available; submission status is unknown and this transaction "
                        "must not be retried"
                    ) from signing_error
            outcome = SignerOutcome.model_validate(raw_outcome)
            if outcome.status is SignerOutcomeStatus.RESIMULATION_REQUIRED:
                workflow.handle_signer_outcome(outcome)
                raise TransactionFlowError(
                    "the signer detected fresh chain-state drift; a new simulation "
                    "and exact review are required"
                )
            # Guard validation and the no-retry state transition happen before
            # persisting any signer-supplied hashes.
            workflow.handle_signer_outcome(outcome)
            transaction_hash = normalize_transaction_hash(
                outcome.transaction_hash or ""
            )
            signing_hash = normalize_transaction_hash(
                outcome.transaction_signing_hash or ""
            )
            record_status = (
                TransactionStatus.UNKNOWN
                if outcome.status is SignerOutcomeStatus.UNKNOWN
                else TransactionStatus.SUBMITTED
            )
            try:
                record = self.store.record_submission(
                    session_id=session_id, workflow_id=workflow_id,
                    plan_digest=workflow.plan.plan_id, envelope_digest=envelope.digest(),
                    chain_id=envelope.chain_id, sender=active.sender,
                    transaction_hash=transaction_hash, signing_hash=signing_hash,
                    status=record_status,
                    error_code=(
                        outcome.code.value
                        if outcome.status is SignerOutcomeStatus.UNKNOWN
                        else None
                    ),
                    now=self._clock(),
                )
            except Exception:
                # The signer outcome is already validated and the workflow is
                # terminal, so never hide a known hash or invite a retry merely
                # because bounded in-memory indexing failed.
                return {
                    "workflow_id": workflow_id,
                    "transaction_hash": transaction_hash,
                    "status": record_status.value,
                    "explorer_url": explorer_transaction_url(
                        envelope.chain_id, transaction_hash
                    ),
                    "chain_id": envelope.chain_id,
                    "sender": active.sender,
                    "updated_at": self._clock(),
                    "error_code": (
                        outcome.code.value
                        if outcome.status is SignerOutcomeStatus.UNKNOWN
                        else None
                    ),
                    "app_state_saved": False,
                    "storage_error_code": "APP_STATE_RECORD_FAILED",
                }
        except TransactionFlowError:
            if workflow.state_machine.state is WorkflowState.SUBMITTING:
                workflow.state_machine.transition(WorkflowState.FAILED)
            raise
        except Exception as error:
            if workflow.state_machine.state is WorkflowState.SUBMITTING:
                workflow.state_machine.transition(WorkflowState.FAILED)
            raise TransactionFlowError("signer submission failed") from error
        return self._record_summary(record)

    @staticmethod
    def _record_summary(record: Any) -> dict[str, Any]:
        return {
            "workflow_id": record.workflow_id,
            "transaction_hash": record.transaction_hash,
            "status": record.status.value,
            "explorer_url": record.explorer_url,
            "chain_id": record.chain_id,
            "sender": record.sender,
            "updated_at": record.updated_at,
            "error_code": record.error_code,
            "app_state_saved": True,
            "storage_error_code": None,
        }

    async def transaction_status(self, *, session_id: str, transaction_hash: str) -> dict[str, Any]:
        try:
            record = self.store.lookup_for_session(session_id, transaction_hash)
        except Exception as error:
            raise TransactionFlowError("transaction was not found") from error
        if record is None:
            raise TransactionFlowError("transaction was not found")
        if record.status in {
            TransactionStatus.SUBMITTED,
            TransactionStatus.UNKNOWN,
        }:
            try:
                chain_id = await self.rpc.require_expected_chain()
                if chain_id != record.chain_id:
                    raise TransactionFlowError(
                        "configured RPC chain does not match the saved transaction"
                    )
                receipt = await self.rpc.transaction_receipt(record.transaction_hash)
            except TransactionFlowError:
                raise
            except Exception as error:
                raise TransactionFlowError(
                    "could not verify the transaction receipt chain"
                ) from error
            if receipt is not None:
                status = TransactionStatus.CONFIRMED if receipt.status == 1 else TransactionStatus.FAILED
                record = self.store.update_status(record.transaction_hash, status=status, now=self._clock())
        return self._record_summary(record)


def configured_transaction_controller(
    *, registry: Registry, rpc_url: str, hmac_secret: bytes
) -> TransactionController:
    """Production wiring, kept explicit so disabled demos own no signer client."""

    from ..signer.capability import create_approval_capability
    from ..signer.client import StdioSignerClient

    return TransactionController(
        registry=registry,
        rpc=EthereumJsonRpcClient(rpc_url, expected_chain_id=_LIVE_CHAIN_ID),
        signer=StdioSignerClient(),
        approval_capability_factory=create_approval_capability,
        approval_capability_secret=hmac_secret,
    )
