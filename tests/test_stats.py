"""Tests for pooled rates and cluster-bootstrap intervals."""

import math
import random

import pytest

from exa_bench.core.stats import Rate, _interval_95, cluster_bootstrap_ci, rate


def test_rate_pools_counts_rather_than_averaging_cluster_rates() -> None:
    # Per-cluster rates are 0% and 100%; the mean would be 50%, the pooled rate is 90%.
    result = rate([(0, 1), (9, 9)], seed=1)

    assert (result.numerator, result.denominator) == (9, 10)
    assert result.value == pytest.approx(0.9)


def test_identical_clusters_give_a_degenerate_interval() -> None:
    result = rate([(3, 10)] * 20, seed=1)

    assert result.value == pytest.approx(0.3)
    assert result.ci_low == pytest.approx(0.3)
    assert result.ci_high == pytest.approx(0.3)


def test_two_opposite_clusters_give_the_widest_interval() -> None:
    # Resamples of two clusters land on 0%, 50%, or 100%, so the 95% interval spans everything.
    result = rate([(0, 10), (10, 10)], seed=1)

    assert result.value == pytest.approx(0.5)
    assert (result.ci_low, result.ci_high) == (0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_interval_brackets_the_estimate_and_stays_in_range(seed: int) -> None:
    rng = random.Random(seed)
    clusters = [(rng.randint(0, d), d) for d in (rng.randint(1, 10) for _ in range(30))]

    result = rate(clusters, seed=seed)

    assert result.value is not None
    assert result.ci_low is not None
    assert result.ci_high is not None
    assert 0.0 <= result.ci_low <= result.value <= result.ci_high <= 1.0


def width(result: Rate) -> float:
    assert result.ci_low is not None
    assert result.ci_high is not None
    return result.ci_high - result.ci_low


def test_interval_narrows_with_more_clusters() -> None:
    clusters = [(1, 10), (5, 10), (9, 10)]

    few = rate(clusters * 3, seed=1)
    many = rate(clusters * 60, seed=1)

    assert few.value == many.value
    assert width(many) < width(few)


def test_single_cluster_interval_collapses_to_its_rate() -> None:
    result = rate([(3, 10)], seed=1)

    assert (result.ci_low, result.ci_high) == (0.3, 0.3)


def test_interval_matches_the_normal_approximation_for_independent_units() -> None:
    # With one result per cluster this is an ordinary bootstrap of a proportion, whose
    # 95% interval is p +/- 1.96 * sqrt(p * (1 - p) / n).
    n, p = 400, 0.25
    clusters = [(1, 1)] * int(n * p) + [(0, 1)] * (n - int(n * p))

    result = rate(clusters, seed=1)

    half_width = 1.96 * math.sqrt(p * (1 - p) / n)
    assert result.ci_low == pytest.approx(p - half_width, abs=0.01)
    assert result.ci_high == pytest.approx(p + half_width, abs=0.01)


def varied_clusters() -> list[tuple[int, int]]:
    """Enough distinct cluster sizes that resampled rates rarely repeat."""
    rng = random.Random(99)
    return [(rng.randint(0, d), d) for d in (rng.randint(1, 10) for _ in range(30))]


def test_same_seed_is_deterministic_and_seeds_matter() -> None:
    clusters = varied_clusters()

    assert rate(clusters, seed=7) == rate(clusters, seed=7)
    assert rate(clusters, seed=7) != rate(clusters, seed=8)


def test_zero_denominator_is_undefined_not_zero() -> None:
    assert rate([], seed=1) == Rate(0, 0, None, None, None)
    assert rate([(0, 0), (0, 0)], seed=1) == Rate(0, 0, None, None, None)


def test_interval_undefined_when_resamples_cannot_be_evaluated() -> None:
    # One empty cluster among real ones: a resample of only empty clusters has no rate.
    assert cluster_bootstrap_ci([(0, 0)], seed=1) is None


def test_percentile_cuts_are_the_2_5th_and_97_5th() -> None:
    assert _interval_95([float(v) for v in range(101)]) == (2.5, 97.5)


@pytest.mark.parametrize("clusters", [[(-1, 5)], [(2, -5)], [(6, 5)]])
def test_invalid_cluster_counts_are_rejected(clusters: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match="invalid cluster counts"):
        rate(clusters, seed=1)


def test_too_few_resamples_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        cluster_bootstrap_ci([(1, 2)], seed=1, resamples=1)
