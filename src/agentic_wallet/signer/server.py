"""Private FastMCP v1 server.  This module runs on stdio only."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .key_store import OSKeyringKeyStore
from .rpc import JsonRpcEthereumRpc
from .service import SignerDenied, SignerService


def create_signer_server(service: SignerService) -> FastMCP:
    """Create the deliberately tiny internal MCP surface."""

    server = FastMCP(
        name="agentic-wallet-private-signer",
        instructions=(
            "Private deterministic signing boundary. Not a model tool. "
            "Only an exact approved envelope and capability may be submitted."
        ),
    )

    @server.tool(name="get_signer_address", description="Return this process's signer address.")
    async def get_signer_address() -> dict[str, str]:
        try:
            return await service.get_signer_address()
        except SignerDenied as error:
            raise ValueError(str(error)) from error

    @server.tool(
        name="sign_and_submit_approved",
        description="Verify and submit one exact, explicitly approved native transfer.",
    )
    async def sign_and_submit_approved(
        envelope: dict[str, Any], approval_capability: str
    ) -> dict[str, Any]:
        try:
            return (
                await service.sign_and_submit_approved(
                    envelope=envelope, approval_capability=approval_capability
                )
            ).model_dump(mode="json")
        except SignerDenied as error:
            raise ValueError(str(error)) from error

    @server.tool(
        name="lookup_submission_outcome",
        description="Recover safe durable metadata for one exact envelope digest.",
    )
    async def lookup_submission_outcome(envelope_digest: str) -> dict[str, Any]:
        try:
            outcome = service.lookup_submission_outcome(envelope_digest)
        except SignerDenied as error:
            raise ValueError(str(error)) from error
        if outcome is None:
            return {"found": False}
        return {"found": True, "outcome": outcome.model_dump(mode="json")}

    return server


def build_server_from_environment() -> FastMCP:
    """Read server-owned RPC configuration; the client cannot supply an endpoint."""

    rpc_url = os.environ.get("AGENTIC_WALLET_SIGNER_RPC_URL")
    if not rpc_url:
        raise RuntimeError("AGENTIC_WALLET_SIGNER_RPC_URL is required for the signer server")
    key_store = OSKeyringKeyStore()
    rpc = JsonRpcEthereumRpc(rpc_url)
    return create_signer_server(SignerService.from_environment(key_store=key_store, rpc=rpc))


def main() -> None:
    # Do not add HTTP/SSE transport selection here: this process is stdio-only.
    build_server_from_environment().run(transport="stdio")


if __name__ == "__main__":
    main()
