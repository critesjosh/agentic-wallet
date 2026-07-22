from agentic_wallet.registry import Registry, RegistryEntry


def _entry(*, symbol: str = "TKN", decimals: int = 6) -> RegistryEntry:
    return RegistryEntry(
        asset_id="base:token",
        chain_id=8453,
        address="0x1111111111111111111111111111111111111111",
        symbol=symbol,
        decimals=decimals,
    )


def test_registry_digest_binds_symbol_and_decimals():
    original = Registry([_entry()]).version_digest()
    assert Registry([_entry(symbol="EVIL")]).version_digest() != original
    assert Registry([_entry(decimals=18)]).version_digest() != original


def test_registry_rejects_duplicate_canonical_ids():
    try:
        Registry([_entry(), _entry(symbol="OTHER")])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate registry ids must fail closed")
