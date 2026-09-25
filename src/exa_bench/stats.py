"""Pooled proportions with cluster-bootstrap confidence intervals (stdlib only)."""

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000

# (numerator, denominator) for one independent unit, e.g. one query's results.
type Cluster = tuple[int, int]


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int
    value: float | None  # numerator / denominator; None when the denominator is zero
    ci_low: float | None  # 95% percentile cluster-bootstrap interval; None when undefined
    ci_high: float | None


def rate(clusters: Sequence[Cluster], *, seed: int, resamples: int = DEFAULT_RESAMPLES) -> Rate:
    """Pool the clusters' counts into one proportion with a 95% cluster-bootstrap CI.

    Pooling (sum of numerators over sum of denominators) weights each cluster by its
    size. Resampling whole clusters keeps within-cluster correlation from narrowing
    the interval.
    """
    _validate(clusters)
    numerator = sum(n for n, _ in clusters)
    denominator = sum(d for _, d in clusters)
    if denominator == 0:
        return Rate(numerator, denominator, None, None, None)
    interval = cluster_bootstrap_ci(clusters, seed=seed, resamples=resamples)
    low, high = interval if interval is not None else (None, None)
    return Rate(numerator, denominator, numerator / denominator, low, high)


def cluster_bootstrap_ci(
    clusters: Sequence[Cluster], *, seed: int, resamples: int = DEFAULT_RESAMPLES
) -> tuple[float, float] | None:
    """2.5th and 97.5th percentiles of the pooled rate over `resamples` cluster resamples.

    Resamples whose pooled denominator is zero are skipped; None if fewer than two remain.
    """
    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    _validate(clusters)
    rng = random.Random(seed)
    pooled: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(clusters, k=len(clusters)) if clusters else []
        denominator = sum(d for _, d in sample)
        if denominator:
            pooled.append(sum(n for n, _ in sample) / denominator)
    if len(pooled) < 2:
        return None
    return _interval_95(pooled)


def _interval_95(values: Sequence[float]) -> tuple[float, float]:
    cuts = statistics.quantiles(values, n=40, method="inclusive")  # 2.5%, 5%, ..., 97.5%
    return cuts[0], cuts[-1]


def _validate(clusters: Sequence[Cluster]) -> None:
    for numerator, denominator in clusters:
        if numerator < 0 or denominator < 0 or numerator > denominator:
            raise ValueError(f"invalid cluster counts: {(numerator, denominator)}")
