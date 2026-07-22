"""Typed, bounded conversation state that never represents authorization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import AssetId, EvmAddress, StrictModel


class ConversationMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class ResolvedIntent(StrictModel):
    """Validated fields remembered for conversational continuity only."""

    chain_id: int | None = Field(default=None, gt=0)
    asset_id: AssetId | None = None
    amount: str | None = Field(default=None, pattern=r"^\d+(\.\d+)?$")
    amount_base_units: str | None = Field(default=None, pattern=r"^(0|[1-9]\d*)$")
    recipient: EvmAddress | None = None


class UserCorrection(StrictModel):
    field: Literal[
        "chain_id", "asset_id", "amount", "amount_base_units", "recipient"
    ]
    previous: str
    current: str


class VerifiedFact(StrictModel):
    """Trusted tool output retained as data, never as an instruction."""

    fact_type: Literal["portfolio", "balance", "allowances", "registry"]
    data: dict[str, Any]


class PriorProposal(StrictModel):
    """Historical typed proposal; still never authorization."""

    action: str
    arguments: dict[str, Any]
    status: Literal["proposed", "validated", "rejected"]


class ConversationLedger(StrictModel):
    """Bounded session ledger. It deliberately has no approval field."""

    workflow_state: str
    chain_id: int = Field(gt=0)
    resolved_intent: ResolvedIntent = Field(default_factory=ResolvedIntent)
    missing_fields: list[str] = Field(default_factory=list, max_length=8)
    active_plan_id: str | None = None
    active_quote_id: str | None = None
    corrections: list[UserCorrection] = Field(default_factory=list, max_length=4)
    verified_facts: list[VerifiedFact] = Field(default_factory=list, max_length=4)
    prior_proposals: list[PriorProposal] = Field(default_factory=list, max_length=4)
    recent_messages: list[ConversationMessage] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _deduplicate_missing_fields(self) -> "ConversationLedger":
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("missing_fields must be unique")
        return self

    def record_message(self, role: Literal["user", "assistant"], content: str) -> None:
        self.recent_messages.append(ConversationMessage(role=role, content=content))
        self.recent_messages = self.recent_messages[-8:]

    def record_verified_fact(self, fact_type: str, data: dict[str, Any]) -> None:
        self.verified_facts.append(
            VerifiedFact.model_validate({"fact_type": fact_type, "data": data})
        )
        self.verified_facts = self.verified_facts[-4:]

    def record_proposal(
        self,
        action: str,
        arguments: dict[str, Any],
        status: Literal["proposed", "validated", "rejected"] = "validated",
    ) -> None:
        self.prior_proposals.append(
            PriorProposal(action=action, arguments=arguments, status=status)
        )
        self.prior_proposals = self.prior_proposals[-4:]

    def record_validated_arguments(self, arguments: dict[str, Any]) -> None:
        remembered = self.resolved_intent.model_dump()
        aliases = {"input_asset_id": "asset_id"}
        for source, value in arguments.items():
            field = aliases.get(source, source)
            if field not in remembered or value is None:
                continue
            previous = remembered[field]
            rendered = str(value)
            if previous is not None and str(previous) != rendered:
                self.corrections.append(
                    UserCorrection(
                        field=field,
                        previous=str(previous),
                        current=rendered,
                    )
                )
                self.corrections = self.corrections[-4:]
            remembered[field] = value
        self.resolved_intent = ResolvedIntent.model_validate(remembered)
