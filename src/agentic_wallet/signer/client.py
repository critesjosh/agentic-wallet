"""Async stdio-only client for deterministic application code.

The caller has no RPC URL or key parameter.  The spawned signer receives its
fixed server configuration from its own inherited server environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..signer_outcome import SignerOutcome


class SignerClientError(RuntimeError):
    pass


def signer_child_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Return the small environment required by keyring and signer startup.

    In particular, DBus keyrings commonly require HOME, the session bus and
    runtime directory.  Do not pass through arbitrary parent-process secrets.
    """

    source = os.environ if environment is None else environment
    allowed = (
        "HOME",
        "PATH",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
        "LANG",
        "AGENTIC_WALLET_SIGNER_RPC_URL",
        "AGENTIC_WALLET_APPROVAL_HMAC_KEY",
    )
    return {name: source[name] for name in allowed if name in source}


@dataclass(frozen=True)
class StdioSignerClient:
    command: str = sys.executable
    args: tuple[str, ...] = ("-m", "agentic_wallet.signer.server")

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # MCP intentionally inherits only a safe environment allowlist.  Forward
        # the signer's two server-owned configuration values explicitly; neither
        # is a private key and neither is exposed through an MCP argument.
        parameters = StdioServerParameters(
            command=self.command, args=list(self.args), env=signer_child_environment()
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        if result.isError:
            raise SignerClientError("private signer rejected the request")
        if result.structuredContent is None or not isinstance(result.structuredContent, dict):
            raise SignerClientError("private signer returned an invalid response")
        return result.structuredContent

    async def get_signer_address(self) -> str:
        response = await self._call("get_signer_address", {})
        address = response.get("address")
        if not isinstance(address, str):
            raise SignerClientError("private signer returned an invalid address")
        return address

    async def sign_and_submit_approved(
        self, *, envelope: dict[str, Any], approval_capability: str
    ) -> SignerOutcome:
        response = await self._call(
            "sign_and_submit_approved",
            {"envelope": envelope, "approval_capability": approval_capability},
        )
        forbidden = {"raw_transaction", "private_key", "signature"}
        if forbidden.intersection(response):
            raise SignerClientError("private signer returned forbidden data")
        try:
            return SignerOutcome.model_validate(response)
        except Exception as error:
            raise SignerClientError("private signer returned an invalid outcome") from error

    async def lookup_submission_outcome(
        self, envelope_digest: str
    ) -> SignerOutcome | None:
        """Recover a durable outcome after sign-call transport/shape failure."""

        response = await self._call(
            "lookup_submission_outcome", {"envelope_digest": envelope_digest}
        )
        if response == {"found": False}:
            return None
        if response.get("found") is not True or set(response) != {"found", "outcome"}:
            raise SignerClientError("private signer returned an invalid lookup response")
        try:
            return SignerOutcome.model_validate(response["outcome"])
        except Exception as error:
            raise SignerClientError("private signer returned an invalid lookup outcome") from error
