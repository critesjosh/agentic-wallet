#!/usr/bin/env python3
"""Create a signer account through the private stdio MCP signer.

The key is generated inside the signer process and written straight to the OS
secure store. This script never sees, prints, or stores a private key, and the
signer refuses to replace an existing one.
"""

from __future__ import annotations

import os
import sys

import anyio

from agentic_wallet.signer.client import SignerClientError, StdioSignerClient


def main() -> int:
    if not os.environ.get("AGENTIC_WALLET_SIGNER_RPC_URL"):
        print(
            "AGENTIC_WALLET_SIGNER_RPC_URL is required so the signer process "
            "can start; account creation itself makes no RPC call.",
            file=sys.stderr,
        )
        return 2
    try:
        address = anyio.run(StdioSignerClient().create_signer_account)
    except SignerClientError as error:
        print(f"account creation failed: {error}", file=sys.stderr)
        return 1
    print(f"Created signer account {address}")
    print(
        "Fund this address on your configured chain before proposing a "
        "transfer. Back it up now: this tool cannot recover or export the key."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
