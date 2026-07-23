"""Exact EIP-1559 transaction fields bound into C1 approval."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..digest import canonical_digest
from .common import EvmAddress, HexData, StrictModel

UintString = Annotated[str, StringConstraints(pattern=r"^(0|[1-9]\d*)$")]
Bytes32 = Annotated[str, StringConstraints(pattern=r"^0x[0-9a-fA-F]{64}$")]


class AccessListEntry(StrictModel):
    address: EvmAddress
    storage_keys: list[Bytes32] = Field(default_factory=list)


class Eip1559Transaction(StrictModel):
    """The real transaction preimage fields, not a display-only summary."""

    transaction_type: Literal[2] = 2
    chain_id: int
    nonce: UintString
    max_priority_fee_per_gas: UintString
    max_fee_per_gas: UintString
    gas_limit: UintString
    from_address: EvmAddress
    to_address: EvmAddress
    value: UintString
    data: HexData = "0x"
    access_list: list[AccessListEntry] = Field(default_factory=list)

    @field_validator("chain_id")
    @classmethod
    def _positive_chain(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chain_id must be positive")
        return value

    @model_validator(mode="after")
    def _fee_and_gas_invariants(self) -> "Eip1559Transaction":
        if int(self.gas_limit) <= 0:
            raise ValueError("gas_limit must be positive")
        if int(self.max_fee_per_gas) < int(self.max_priority_fee_per_gas):
            raise ValueError("max fee must be at least the priority fee")
        return self

    def eth_account_dict(self) -> dict:
        return {
            "type": 2,
            "chainId": self.chain_id,
            "nonce": int(self.nonce),
            "maxPriorityFeePerGas": int(self.max_priority_fee_per_gas),
            "maxFeePerGas": int(self.max_fee_per_gas),
            "gas": int(self.gas_limit),
            "to": self.to_address,
            "value": int(self.value),
            "data": self.data,
            "accessList": [
                {
                    "address": entry.address,
                    "storageKeys": entry.storage_keys,
                }
                for entry in self.access_list
            ],
        }

    def digest(self) -> str:
        """Canonical digest of every EIP-1559 preimage field.

        This is an application binding digest, not a signed transaction hash.
        A signer still computes and verifies the chain-native signing hash.
        """

        return canonical_digest(self.model_dump(mode="json"))


class SignedTransactionResult(StrictModel):
    """Safe submission metadata; raw signed bytes never cross the signer boundary."""

    status: Literal["SUBMITTED", "UNKNOWN"]
    from_address: EvmAddress
    envelope_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    transaction_signing_hash: Bytes32
    transaction_hash: Bytes32
