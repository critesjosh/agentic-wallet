"""Native transfer scope derived from the shared code-owned chain metadata."""

from __future__ import annotations

from ..chain_metadata import ChainMetadataError, get_chain_metadata


class NativeAssetError(ValueError):
    pass


LIVE_SIGNING_CHAIN_ID = 8453


def native_asset_id_for_chain(chain_id: int) -> str:
    """Return the explicitly pinned native asset ID for a supported chain."""

    if chain_id != LIVE_SIGNING_CHAIN_ID:
        raise NativeAssetError("unsupported signing chain")
    try:
        return get_chain_metadata(chain_id).native_asset_id
    except ChainMetadataError as error:
        raise NativeAssetError("unsupported signing chain") from error
