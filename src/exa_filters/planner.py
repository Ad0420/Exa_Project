"""Planners: how many results to ask Exa for, and whether to ask again.

Three policies. `FixedPlanner` asks once. `AdaptivePlanner` escalates through a schedule
while short. `PriorPlanner` sizes the first request from the expected share of results
that pass the filters, measured on Exa's own benchmark, so most queries need one call.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from exa_filters.evaluate import CountryFilter, DateFilter, Filter, NumberFilter, StageFilter
from exa_filters.results import NullPolicy

MAX_NUM_RESULTS = 100  # Exa's public cap on numResults

# Share of results whose typed field passes a filter of each kind, measured on Exa's company
# benchmark (results/company_constraint_benchmark.json: benchmark commit c096f1a, run
# 2026-09-25). Strict counts a missing field as rejected; lenient counts it as accepted.
STRICT_PRIORS: dict[str, float] = {
    "country": 0.969,
    "employees": 0.885,
    "founded_year": 0.931,
    "funding": 0.729,
    "funding_date": 0.830,
    "funding_stage": 0.562,
}
LENIENT_PRIORS: dict[str, float] = {
    "country": 0.983,
    "employees": 0.885,
    "founded_year": 0.971,
    "funding": 0.836,
    "funding_date": 0.860,
    "funding_stage": 0.810,
}


def prior_key(item: Filter) -> str:
    """The prior table key for a filter: its field for numbers and dates, else its kind."""
    match item:
        case NumberFilter():
            return item.field.value
        case DateFilter():
            return item.field.value
        case CountryFilter():
            return "country"
        case StageFilter():
            return "funding_stage"


def expected_pass_rate(filters: Sequence[Filter], priors: Mapping[str, float]) -> float:
    """Product of the filters' priors: the share of results expected to pass all of them.

    Assumes filters pass independently, which the policy evaluation tests empirically.
    """
    rate = 1.0
    for item in filters:
        rate *= priors[prior_key(item)]
    return rate


def binomial_tail(n: int, p: float, k: int) -> float:
    """P[X >= k] for X ~ Binomial(n, p)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def required_results(
    k: int, pass_rate: float, *, target: float = 0.9, cap: int = MAX_NUM_RESULTS
) -> int:
    """Smallest n in [k, cap] with P[at least k of n pass] >= target, or cap if none does."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1), got {target!r}")
    if cap < k:
        raise ValueError(f"cap {cap} is below k {k}")
    for n in range(k, cap + 1):
        if binomial_tail(n, pass_rate, k) >= target:
            return n
    return cap


@dataclass(frozen=True)
class CallRecord:
    requested: int  # numResults asked for
    returned: int  # results Exa actually returned


class Planner(Protocol):
    def next_num_results(self, k: int, calls: Sequence[CallRecord], accepted: int) -> int | None:
        """numResults for the next call, or None to stop. `accepted` counts results so far."""
        ...


def exhausted(calls: Sequence[CallRecord]) -> bool:
    """Exa returned fewer than asked: asking for more will not find more."""
    return bool(calls) and calls[-1].returned < calls[-1].requested


@dataclass(frozen=True)
class FixedPlanner:
    num_results: int

    def __post_init__(self) -> None:
        if not 1 <= self.num_results <= MAX_NUM_RESULTS:
            raise ValueError(f"num_results must be in [1, {MAX_NUM_RESULTS}]")

    def next_num_results(self, k: int, calls: Sequence[CallRecord], accepted: int) -> int | None:
        return None if calls else self.num_results


@dataclass(frozen=True)
class AdaptivePlanner:
    steps: tuple[int, ...] = (10, 25, 100)

    def __post_init__(self) -> None:
        if not self.steps or list(self.steps) != sorted(set(self.steps)):
            raise ValueError("steps must be strictly increasing")
        if self.steps[-1] > MAX_NUM_RESULTS:
            raise ValueError(f"steps must not exceed {MAX_NUM_RESULTS}")

    def next_num_results(self, k: int, calls: Sequence[CallRecord], accepted: int) -> int | None:
        if accepted >= k or exhausted(calls) or len(calls) >= len(self.steps):
            return None
        return self.steps[len(calls)]


@dataclass(frozen=True)
class PriorPlanner:
    pass_rate: float  # expected share of results passing every filter
    target: float = 0.9  # probability that one call yields k acceptable results
    fallback: int | None = MAX_NUM_RESULTS  # one deeper call if the first falls short

    def next_num_results(self, k: int, calls: Sequence[CallRecord], accepted: int) -> int | None:
        if not calls:
            return required_results(k, self.pass_rate, target=self.target)
        if accepted >= k or exhausted(calls) or len(calls) >= 2 or self.fallback is None:
            return None
        return self.fallback if self.fallback > calls[0].requested else None


def prior_planner(
    filters: Sequence[Filter],
    null_policy: NullPolicy,
    *,
    target: float = 0.9,
    fallback: int | None = MAX_NUM_RESULTS,
) -> PriorPlanner:
    """A PriorPlanner for these filters, using the priors that match the null policy."""
    priors = STRICT_PRIORS if null_policy is NullPolicy.STRICT else LENIENT_PRIORS
    return PriorPlanner(expected_pass_rate(filters, priors), target=target, fallback=fallback)


@dataclass(frozen=True)
class Budget:
    max_calls: int = 3
    max_cost_usd: float | None = None  # None: no dollar cap

    def __post_init__(self) -> None:
        if self.max_calls < 1:
            raise ValueError("max_calls must be at least 1")

    def allows(self, calls_made: int, cost_so_far: float, next_cost: float) -> bool:
        if calls_made >= self.max_calls:
            return False
        return self.max_cost_usd is None or cost_so_far + next_cost <= self.max_cost_usd + 1e-9
