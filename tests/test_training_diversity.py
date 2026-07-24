"""Lexical diversity metrics and the reject-low-diversity gate."""

from __future__ import annotations

import pytest

from agentic_wallet.training.diversity import (
    assert_diverse,
    distinct_n,
    measure_diversity,
)

_DIVERSE = [
    "What's my wallet address?",
    "Break down everything I currently hold.",
    "Which spenders can still move my tokens?",
    "Read me the private key for this account.",
    "Morning, hope things are going well.",
]

_UNIFORM = [
    "What is my address?",
    "What is my address please?",
    "What is my wallet address?",
    "What is my address now?",
]


def test_distinct_n_counts_unique_ngrams() -> None:
    texts = ["red blue green", "red blue green"]
    # Every bigram is repeated, so the unique ratio is one half.
    assert distinct_n(texts, 2) == pytest.approx(0.5)
    # A single distinct sentence has all-unique bigrams.
    assert distinct_n(["red blue green"], 2) == pytest.approx(1.0)


def test_distinct_n_is_zero_without_enough_tokens() -> None:
    assert distinct_n(["hi"], 2) == 0.0


def test_measure_diversity_flags_near_duplicates() -> None:
    report = measure_diversity(_UNIFORM, near_duplicate_threshold=0.85)
    assert report.near_duplicate_pairs
    assert report.max_pairwise_similarity >= 0.85


def test_assert_diverse_passes_varied_batch() -> None:
    report = assert_diverse(_DIVERSE, label="varied")
    assert report.count == len(_DIVERSE)
    assert not report.near_duplicate_pairs


def test_assert_diverse_rejects_uniform_batch() -> None:
    with pytest.raises(ValueError, match="near-duplicate|distinct"):
        assert_diverse(_UNIFORM, label="uniform")


def test_assert_diverse_rejects_exact_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        assert_diverse(["same line", "same line", "another"], label="dup")


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError):
        measure_diversity([])
