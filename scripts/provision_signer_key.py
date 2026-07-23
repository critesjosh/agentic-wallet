#!/usr/bin/env python3
"""Interactively provision the private signer key into an OS secure store."""

from __future__ import annotations

import getpass
import sys

from agentic_wallet.signer.key_store import KeyStoreError, OSKeyringKeyStore


def main() -> int:
    if not sys.stdin.isatty() or not sys.stderr.isatty():
        print("refusing to provision without an interactive TTY", file=sys.stderr)
        return 2
    private_key = ""
    try:
        private_key = getpass.getpass("Private key to store (input hidden): ")
        address = OSKeyringKeyStore().provision_private_key(private_key)
    except KeyStoreError as error:
        print(f"provisioning failed: {error}", file=sys.stderr)
        return 1
    finally:
        # Best-effort reference removal; neither key nor its representation is printed.
        private_key = ""
    print(f"Signer key stored for address {address}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
