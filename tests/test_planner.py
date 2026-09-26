"""Tests for the planners, binomial sizing, priors, and budgets."""

import json
import math
from datetime import date
from pathlib import Path

import pytest

from exa_filters.constraints import DateField, DateOp, NumberField, NumberOp
from exa_filters.evaluate import CountryFilter, DateFilter, Filter, NumberFilter, StageFilter
from exa_filters.planner import (
    LENIENT_PRIORS,
    MAX_NUM_RESULTS,
    STRICT_PRIORS,
    AdaptivePlanner,
    Budget,
    CallRecord,
    FixedPlanner,
    PriorPlanner,
    binomial_tail,
    exhausted,
    expected_pass_rate,
    list_price,
    prior_key,
    prior_planner,
    required_results,
)
from exa_filters.results import NullPolicy

BENCHMARK = Path(__file__).parent.parent / "results" / "company_constraint_benchmark.json"


def test_list_price_matches_exas_published_rates() -> None:
    assert list_price(10) == pytest.approx(0.007)
    assert list_price(1) == pytest.approx(0.007)
    assert list_price(25) == pytest.approx(0.022)
    assert list_price(100) == pytest.approx(0.097)


def test_priors_match_the_committed_benchmark() -> None:
    by_constraint = json.loads(BENCHMARK.read_text())["strict"]["by_constraint"]
    for key, strict in STRICT_PRIORS.items():
        stats = by_constraint[key]
        total = stats["passed"] + stats["failed"] + stats["unknown"]
        assert strict == pytest.approx(stats["passed"] / total, abs=0.0005), key
        assert LENIENT_PRIORS[key] == pytest.approx(
            (stats["passed"] + stats["unknown"]) / total, abs=0.0005
        ), key


def test_prior_key_and_expected_pass_rate() -> None:
    filters: list[Filter] = [
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 30),
        CountryFilter(("Singapore",)),
        StageFilter("Series A"),
        DateFilter(DateField.FUNDING_DATE, DateOp.GTE, date(2024, 1, 1)),
    ]

    assert [prior_key(f) for f in filters] == [
        "employees",
        "country",
        "funding_stage",
        "funding_date",
    ]
    assert expected_pass_rate(filters, STRICT_PRIORS) == pytest.approx(
        0.885 * 0.969 * 0.562 * 0.830
    )
    assert expected_pass_rate([], STRICT_PRIORS) == 1.0


def test_binomial_tail_known_values() -> None:
    assert binomial_tail(3, 0.5, 2) == pytest.approx(0.5)  # 2 or 3 heads of 3
    assert binomial_tail(10, 1.0, 10) == pytest.approx(1.0)
    assert binomial_tail(10, 0.0, 1) == pytest.approx(0.0)
    assert binomial_tail(5, 0.3, 0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="p must be"):
        binomial_tail(5, 1.5, 1)


def brute_force_required(k: int, p: float, target: float, cap: int) -> int:
    for n in range(k, cap + 1):
        tail = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
        if tail >= target:
            return n
    return cap


@pytest.mark.parametrize(("k", "p", "target"), [(10, 0.9, 0.9), (10, 0.56, 0.9), (5, 0.7, 0.99)])
def test_required_results_matches_brute_force(k: int, p: float, target: float) -> None:
    assert required_results(k, p, target=target) == brute_force_required(k, p, target, 100)


def test_required_results_edges() -> None:
    assert required_results(10, 1.0) == 10  # everything passes: ask for exactly k
    assert required_results(10, 0.0) == MAX_NUM_RESULTS  # nothing passes: hit the cap
    assert required_results(10, 0.5, cap=15) == 15
    assert required_results(2, 0.5, target=0.5) == 3  # tail(3, 0.5, 2) is exactly 0.5
    assert required_results(10, 0.9, target=0.99) >= required_results(10, 0.9, target=0.5)
    with pytest.raises(ValueError, match="k must be"):
        required_results(0, 0.5)
    with pytest.raises(ValueError, match="target must be"):
        required_results(5, 0.5, target=1.0)
    with pytest.raises(ValueError, match="below k"):
        required_results(20, 0.5, cap=10)


def test_fixed_planner_asks_exactly_once() -> None:
    planner = FixedPlanner(25)

    assert planner.next_num_results(10, [], 0) == 25
    assert planner.next_num_results(10, [CallRecord(25, 25)], 3) is None
    with pytest.raises(ValueError, match="num_results"):
        FixedPlanner(0)
    with pytest.raises(ValueError, match="num_results"):
        FixedPlanner(101)


def test_adaptive_planner_escalates_while_short() -> None:
    planner = AdaptivePlanner()

    assert planner.next_num_results(10, [], 0) == 10
    assert planner.next_num_results(10, [CallRecord(10, 10)], 7) == 25
    assert planner.next_num_results(10, [CallRecord(10, 10), CallRecord(25, 25)], 9) == 100
    calls = [CallRecord(10, 10), CallRecord(25, 25), CallRecord(100, 90)]
    assert planner.next_num_results(10, calls, 9) is None  # schedule exhausted
    assert planner.next_num_results(10, [CallRecord(10, 10)], 10) is None  # already filled
    assert planner.next_num_results(10, [CallRecord(10, 8)], 5) is None  # Exa has no more


def test_adaptive_planner_validates_steps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        AdaptivePlanner((25, 10))
    with pytest.raises(ValueError, match="strictly increasing"):
        AdaptivePlanner((10, 10))
    with pytest.raises(ValueError, match="exceed"):
        AdaptivePlanner((10, 200))


def test_prior_planner_sizes_the_first_call_then_falls_back_once() -> None:
    planner = PriorPlanner(pass_rate=0.56, target=0.9)
    first = planner.next_num_results(10, [], 0)

    assert first == required_results(10, 0.56, target=0.9)
    assert planner.next_num_results(10, [CallRecord(first, first)], 6) == 100
    assert planner.next_num_results(10, [CallRecord(first, first)], 10) is None
    assert planner.next_num_results(10, [CallRecord(first, first - 1)], 6) is None
    assert planner.next_num_results(10, [CallRecord(first, first), CallRecord(100, 100)], 8) is None
    assert (
        PriorPlanner(pass_rate=0.56, fallback=None).next_num_results(
            10, [CallRecord(first, first)], 6
        )
        is None
    )


def test_exhausted_when_the_last_call_returned_fewer_than_asked() -> None:
    assert not exhausted([])
    assert not exhausted([CallRecord(10, 10)])
    assert exhausted([CallRecord(25, 24)])
    assert not exhausted([CallRecord(10, 9), CallRecord(25, 25)])


def test_prior_planner_from_filters_and_null_policy() -> None:
    filters: list[Filter] = [
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 30),
        CountryFilter(("Singapore",)),
    ]

    strict = prior_planner(filters, NullPolicy.STRICT)
    lenient = prior_planner(filters, NullPolicy.LENIENT, target=0.99)

    assert strict.pass_rate == pytest.approx(0.885 * 0.969)
    assert strict.target == 0.9
    assert lenient.pass_rate == pytest.approx(0.885 * 0.983)
    assert lenient.target == 0.99


def test_prior_planner_never_falls_back_to_a_smaller_call() -> None:
    planner = PriorPlanner(pass_rate=0.0)  # first call already at the cap

    assert planner.next_num_results(10, [], 0) == MAX_NUM_RESULTS
    assert planner.next_num_results(10, [CallRecord(100, 100)], 4) is None


def test_budget_limits_calls_and_dollars() -> None:
    budget = Budget(max_calls=2, max_cost_usd=0.03)

    assert budget.allows(0, 0.0, 0.007)
    assert budget.allows(1, 0.007, 0.022)  # 0.029 <= 0.03
    assert not budget.allows(1, 0.007, 0.097)
    assert not budget.allows(2, 0.0, 0.007)
    assert Budget(max_calls=1).allows(0, 0.0, 10.0)  # no dollar cap
    with pytest.raises(ValueError, match="max_calls"):
        Budget(max_calls=0)
