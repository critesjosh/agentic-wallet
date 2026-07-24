"""Lexical diversity measurement and a reject-low-diversity gate.

The V8 generalization plan makes phrasing diversity the primary lever: each
intent and adversarial family carries 5-10x paraphrase variety so the adapter
generalizes to novel wording instead of memorizing a dozen near-duplicate
templates. Diversity is only a lever if it is measured. This module scores a
batch of authored utterances and fails closed when a batch is too uniform or
contains near-duplicate pairs, which is the collapse mode the research warns
about.

The metrics are dependency-free and deterministic:

- ``distinct_n``: unique n-grams over total n-grams across the batch. Higher
  means more varied vocabulary and structure.
- ``max_pairwise_similarity``: the largest ``difflib`` ratio over all pairs, the
  same comparison ``data.py`` uses against the frozen benchmark. Lower means no
  two utterances are near-restatements of each other.

Embedding spread is a documented future extension; it needs a model and is left
out of the fail-closed gate so dataset generation stays deterministic and
offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams across the whole batch.

    Returns 0.0 when the batch has no n-grams of the requested size, so short
    single-word utterances do not silently pass the gate.
    """

    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        grams = _ngrams(_tokens(text), n)
        total += len(grams)
        unique.update(grams)
    if total == 0:
        return 0.0
    return len(unique) / total


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _pairwise_ratios(texts: list[str]) -> list[tuple[int, int, float]]:
    normalized = [_normalized(text) for text in texts]
    ratios: list[tuple[int, int, float]] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            ratio = SequenceMatcher(None, normalized[i], normalized[j]).ratio()
            ratios.append((i, j, ratio))
    return ratios


@dataclass(frozen=True)
class DiversityReport:
    count: int
    distinct_1: float
    distinct_2: float
    distinct_3: float
    max_pairwise_similarity: float
    near_duplicate_pairs: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "distinct_1": round(self.distinct_1, 4),
            "distinct_2": round(self.distinct_2, 4),
            "distinct_3": round(self.distinct_3, 4),
            "max_pairwise_similarity": round(self.max_pairwise_similarity, 4),
            "near_duplicate_pairs": [
                [i, j, round(ratio, 4)] for i, j, ratio in self.near_duplicate_pairs
            ],
        }


def measure_diversity(
    texts: list[str], *, near_duplicate_threshold: float = 0.85
) -> DiversityReport:
    """Score a batch of utterances for lexical spread and near-duplication."""

    if not texts:
        raise ValueError("cannot measure diversity of an empty batch")
    ratios = _pairwise_ratios(texts)
    near_duplicates = tuple(
        sorted(
            (pair for pair in ratios if pair[2] >= near_duplicate_threshold),
            key=lambda pair: pair[2],
            reverse=True,
        )
    )
    max_similarity = max((ratio for _, _, ratio in ratios), default=0.0)
    return DiversityReport(
        count=len(texts),
        distinct_1=distinct_n(texts, 1),
        distinct_2=distinct_n(texts, 2),
        distinct_3=distinct_n(texts, 3),
        max_pairwise_similarity=max_similarity,
        near_duplicate_pairs=near_duplicates,
    )


def assert_diverse(
    texts: list[str],
    *,
    label: str,
    min_distinct_1: float = 0.30,
    min_distinct_2: float = 0.55,
    max_pairwise_similarity: float = 0.85,
) -> DiversityReport:
    """Fail closed when a paraphrase batch is too uniform or near-duplicated.

    Thresholds are deliberately conservative: they reject the collapse mode
    (a batch of restatements) without demanding artificial variety from small
    families. ``label`` names the family in the error so a rejected batch is
    actionable.
    """

    if len(texts) != len(set(_normalized(text) for text in texts)):
        raise ValueError(f"{label}: contains exact duplicate utterances")
    report = measure_diversity(texts, near_duplicate_threshold=max_pairwise_similarity)
    if report.near_duplicate_pairs:
        i, j, ratio = report.near_duplicate_pairs[0]
        raise ValueError(
            f"{label}: near-duplicate utterances "
            f"{texts[i]!r} and {texts[j]!r} (similarity {ratio:.3f})"
        )
    if report.distinct_1 < min_distinct_1:
        raise ValueError(
            f"{label}: distinct-1 {report.distinct_1:.3f} below {min_distinct_1}"
        )
    if report.distinct_2 < min_distinct_2:
        raise ValueError(
            f"{label}: distinct-2 {report.distinct_2:.3f} below {min_distinct_2}"
        )
    return report
