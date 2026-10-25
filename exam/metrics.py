"""Ranking metrics with explicit duplicate and empty-truth semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

K = 5


def apk(actual: Sequence[int] | set[int], predicted: Sequence[int], k: int = K) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(actual)
    if not relevant:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for rank, item in enumerate(predicted[:k], 1):
        if item in relevant and item not in seen:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(k, len(relevant))


def mapk(
    truth: Mapping[int, Sequence[int]], predictions: Mapping[int, Sequence[int]], k: int = K
) -> float:
    if not truth:
        raise ValueError("MAP is undefined for an empty evaluation cohort")
    missing = set(truth).difference(predictions)
    if missing:
        raise ValueError(f"Predictions missing for {len(missing)} evaluated users")
    return math.fsum(apk(items, predictions[user], k) for user, items in truth.items()) / len(truth)
