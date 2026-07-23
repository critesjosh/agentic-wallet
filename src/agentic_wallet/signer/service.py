"""Deterministic private-key signer service, isolated behind stdio MCP."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from eth_account import Account
from eth_account.typed_transactions import TypedTransaction
from eth_utils import keccak
from pydantic import ValidationError

from ..schemas.approval import ApprovalEnvelope
from ..schemas.signing import Eip1559Transaction
from ..signer_outcome import SignerOutcome, SignerOutcomeCode, SignerOutcomeStatus
from .capability import (
    CapabilityError,
    decode_approval_hmac_key,
    verify_approval_capability,
)
from .capability_store import (
    AtomicFileCapabilityUseStore,
    CapabilityUseError,
    CapabilityUseStore,
    capability_fingerprint,
)
from .chains import NativeAssetError, native_asset_id_for_chain
from .key_store import KeyStore
from .outcome_store import (
    AtomicFileOutcomeStore,
    OutcomeStore,
)
from .rpc import EthereumRpc

_RPC_URL_ENV = "AGENTIC_WALLET_SIGNER_RPC_URL"
_CAPABILITY_SECRET_ENV = "AGENTIC_WALLET_APPROVAL_HMAC_KEY"
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


class SignerDenied(RuntimeError):
    """A submission failed closed before an accepted broadcast."""


SubmissionResult = SignerOutcome


class SignerService:
    """Verify all signing predicates independently, then sign and submit once."""

    def __init__(
        self,
        *,
        key_store: KeyStore,
        rpc: EthereumRpc,
        approval_hmac_secret: bytes,
        capability_use_store: CapabilityUseStore,
        outcome_store: OutcomeStore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(approval_hmac_secret) < 32:
            raise ValueError("approval capability secret must be at least 32 bytes")
        self._key_store = key_store
        self._rpc = rpc
        self._approval_hmac_secret = approval_hmac_secret
        self._capability_use_store = capability_use_store
        self._outcome_store = outcome_store
        self._clock = clock

    @classmethod
    def from_environment(cls, *, key_store: KeyStore, rpc: EthereumRpc) -> "SignerService":
        """Read fixed server configuration, never a private key, from the environment."""

        if not os.environ.get(_RPC_URL_ENV):
            raise SignerDenied("signer RPC endpoint is not configured")
        secret = os.environ.get(_CAPABILITY_SECRET_ENV)
        if secret is None:
            raise SignerDenied("approval capability verifier is not configured")
        try:
            decoded_secret = decode_approval_hmac_key(secret)
        except CapabilityError as error:
            raise SignerDenied(str(error)) from error
        return cls(
            key_store=key_store,
            rpc=rpc,
            approval_hmac_secret=decoded_secret,
            capability_use_store=AtomicFileCapabilityUseStore(),
            outcome_store=AtomicFileOutcomeStore(),
        )

    async def get_signer_address(self) -> dict[str, str]:
        try:
            return {"address": Account.from_key(self._key_store.load_private_key()).address}
        except Exception as error:
            if isinstance(error, SignerDenied):
                raise
            raise SignerDenied("signer key is unavailable") from error

    def lookup_submission_outcome(
        self, envelope_digest: str
    ) -> SignerOutcome | None:
        """Recover safe post-sign metadata after a lost MCP response."""

        try:
            return self._outcome_store.lookup(envelope_digest)
        except Exception as error:
            raise SignerDenied("signer outcome journal is unavailable") from error

    def _parse_and_validate_envelope(self, payload: dict[str, Any]) -> ApprovalEnvelope:
        try:
            envelope = ApprovalEnvelope.model_validate(payload)
        except (ValidationError, TypeError) as error:
            raise SignerDenied("invalid approval envelope") from error
        transaction = envelope.signing_transaction
        if transaction is None:
            raise SignerDenied("approval envelope has no signing preimage")
        try:
            native_asset_id = native_asset_id_for_chain(envelope.chain_id)
        except NativeAssetError as error:
            raise SignerDenied(str(error)) from error
        if not envelope.policy.allowed or envelope.policy.violations:
            raise SignerDenied("approval policy does not allow signing")
        if not envelope.simulation.success or envelope.simulation.mismatch:
            raise SignerDenied("approval simulation is not safe")
        plan = envelope.plan
        if (
            plan.kind != "transfer"
            or plan.asset_id != native_asset_id
            or plan.calldata.lower() != "0x"
            or transaction.data.lower() != "0x"
            or transaction.access_list
            or plan.recipient_address is None
            or plan.recipient_address.lower() != plan.to_address.lower()
            or transaction.to_address.lower() != plan.recipient_address.lower()
            or plan.recipient_address.lower() == _ZERO_ADDRESS
            or plan.recipient_address.lower() == transaction.from_address.lower()
            or plan.value.decimals != 18
        ):
            raise SignerDenied("only native-token transfers are enabled")
        return envelope

    @staticmethod
    def _local_transaction_hash(raw_transaction: bytes) -> str:
        return "0x" + keccak(raw_transaction).hex()

    @staticmethod
    def _transaction_signing_hash(transaction: Eip1559Transaction) -> str:
        """Return the EIP-1559 unsigned preimage hash, not the transaction hash."""

        try:
            return "0x" + TypedTransaction.from_dict(transaction.eth_account_dict()).hash().hex()
        except Exception as error:
            raise SignerDenied("could not compute transaction signing hash") from error

    async def sign_and_submit_approved(
        self, *, envelope: dict[str, Any], approval_capability: str
    ) -> SignerOutcome:
        """Sign exactly one fresh approved preimage and return only safe metadata."""

        parsed = self._parse_and_validate_envelope(envelope)
        now = int(self._clock())
        digest = parsed.digest()
        if now >= parsed.expires_at:
            raise SignerDenied("approval envelope is expired")
        try:
            capability = verify_approval_capability(
                approval_capability,
                secret=self._approval_hmac_secret,
                envelope_digest=digest,
                envelope_expires_at=parsed.expires_at,
                now=now,
                clock=self._clock,
            )
        except CapabilityError as error:
            raise SignerDenied(str(error)) from error
        # Claim before *any* key access or signing.  A failed broadcast still
        # burns the capability rather than allowing an ambiguous retry to sign.
        try:
            self._capability_use_store.claim(
                capability_fingerprint=capability_fingerprint(approval_capability),
                expires_at=capability.expires_at,
                now=now,
            )
        except CapabilityUseError as error:
            raise SignerDenied(str(error)) from error

        transaction: Eip1559Transaction = parsed.signing_transaction  # checked above
        signer_address = transaction.from_address

        def resimulation(code: SignerOutcomeCode) -> SignerOutcome:
            return SignerOutcome(
                status=SignerOutcomeStatus.RESIMULATION_REQUIRED,
                code=code,
                envelope_digest=digest,
                from_address=signer_address,
            )

        # Mutable chain checks happen before key access. The already-claimed
        # capability remains burned for every outcome.
        try:
            live_chain_id = await self._rpc.chain_id()
        except Exception:
            return resimulation(SignerOutcomeCode.LIVE_PREFLIGHT_FAILED)
        if live_chain_id != parsed.chain_id:
            return resimulation(SignerOutcomeCode.RPC_CHAIN_CHANGED)
        try:
            live_nonce = await self._rpc.pending_nonce(signer_address)
        except Exception:
            return resimulation(SignerOutcomeCode.LIVE_PREFLIGHT_FAILED)
        if live_nonce != parsed.nonce:
            return resimulation(SignerOutcomeCode.PENDING_NONCE_CHANGED)
        try:
            live_anchor = await self._rpc.relevant_state_anchor(signer_address)
        except Exception:
            return resimulation(SignerOutcomeCode.LIVE_PREFLIGHT_FAILED)
        if live_anchor != parsed.state_anchor:
            return resimulation(SignerOutcomeCode.RELEVANT_STATE_CHANGED)
        try:
            await self._rpc.verify_simulation(transaction)
        except Exception:
            return resimulation(SignerOutcomeCode.LIVE_PREFLIGHT_FAILED)

        try:
            private_key = self._key_store.load_private_key()
            key_address = Account.from_key(private_key).address
        except Exception as error:
            raise SignerDenied("signer key is unavailable") from error
        if key_address.lower() != signer_address.lower():
            raise SignerDenied("approved sender does not match signer")

        signing_hash = self._transaction_signing_hash(transaction)
        try:
            signed = Account.sign_transaction(transaction.eth_account_dict(), private_key)
            raw_transaction = bytes(signed.raw_transaction)
            recovered = Account.recover_transaction(raw_transaction)
        except Exception as error:
            raise SignerDenied("transaction signing failed") from error
        if recovered.lower() != signer_address.lower():
            raise SignerDenied("signed transaction recovered an unexpected sender")
        local_hash = self._local_transaction_hash(raw_transaction)
        possible_broadcast = SignerOutcome(
            status=SignerOutcomeStatus.UNKNOWN,
            code=SignerOutcomeCode.BROADCAST_RESULT_UNKNOWN,
            envelope_digest=digest,
            from_address=signer_address,
            transaction_hash=local_hash,
            transaction_signing_hash=signing_hash,
        )
        try:
            self._outcome_store.record_unknown(possible_broadcast)
        except Exception as error:
            # No broadcast is attempted unless recovery metadata is durable.
            raise SignerDenied(
                "signer outcome journal is unavailable before broadcast"
            ) from error
        try:
            remote_hash = await self._rpc.submit_raw_transaction(raw_transaction)
        except Exception:
            return possible_broadcast
        if not isinstance(remote_hash, str) or remote_hash.lower() != local_hash.lower():
            return possible_broadcast.model_copy(
                update={"code": SignerOutcomeCode.BROADCAST_HASH_MISMATCH}
            )
        submitted = SignerOutcome(
            status=SignerOutcomeStatus.SUBMITTED,
            code=SignerOutcomeCode.SUBMITTED,
            transaction_hash=local_hash,
            transaction_signing_hash=signing_hash,
            from_address=signer_address,
            envelope_digest=digest,
        )
        try:
            self._outcome_store.mark_submitted(submitted)
        except Exception:
            # Matching broadcast occurred, but recovery cannot prove the durable
            # promotion. Preserve the conservative UNKNOWN public result.
            return possible_broadcast
        return submitted
