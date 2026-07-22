"""Behavioral and security benchmark with hard-zero blockers (plan.md P6)."""

from .cases import BENCHMARK_DATASET_ROLE, BenchmarkCase
from .loader import load_cases
from .runner import (
    BENCHMARK_CONTRACT_VERSION,
    BenchmarkReport,
    CaseResult,
    FamilyMetrics,
    HardZeroMetrics,
    run_benchmark,
)

__all__ = [
    "BENCHMARK_DATASET_ROLE",
    "BENCHMARK_CONTRACT_VERSION",
    "BenchmarkCase",
    "load_cases",
    "run_benchmark",
    "BenchmarkReport",
    "CaseResult",
    "FamilyMetrics",
    "HardZeroMetrics",
]
