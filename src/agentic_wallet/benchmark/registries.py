"""Distinct trusted registry universes for train and held-out benchmark families."""

from ..registry import Registry, RegistryEntry

# These are synthetic, family-local identities used only to test C3 separation.
TRAIN_REGISTRY = Registry(
    [
        RegistryEntry("base:native", 8453, "native", "ETH", 18),
        RegistryEntry("base:usdc", 8453, "0x1111111111111111111111111111111111111111", "USDC", 6),
        RegistryEntry("base:weth", 8453, "0x2222222222222222222222222222222222222222", "WETH", 18),
        RegistryEntry("base:fixture-swap-router", 8453, "0x7777777777777777777777777777777777777777", "TRAIN_ROUTER", 0),
    ]
)

EVAL_REGISTRY = Registry(
    [
        RegistryEntry("base:native", 8453, "native", "ETH", 18),
        RegistryEntry("base:dai", 8453, "0x4444444444444444444444444444444444444444", "DAI", 18),
        RegistryEntry("base:cbeth", 8453, "0x5555555555555555555555555555555555555555", "cbETH", 18),
        RegistryEntry("base:aerodrome-router", 8453, "0x6666666666666666666666666666666666666666", "EVAL_ROUTER", 0),
    ]
)

BENCHMARK_REGISTRIES = {"train": TRAIN_REGISTRY, "eval": EVAL_REGISTRY}
