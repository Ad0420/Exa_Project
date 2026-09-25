"""Day-0 probe: do Exa's company results carry the typed fields our constraints need?"""

import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from exa_bench.benchmark_data import BenchmarkQuery
from exa_bench.coverage import CoverageReport
from exa_bench.exa_api import SearchCall

# The benchmark constraint keys our evaluator can check, and the coverage field each maps to.
CHECKABLE_KEYS: dict[str, str] = {
    "founded_year": "founded_year",
    "employees": "employees",
    "funding": "funding",
    "country": "country",
    "funding_stage": "funding_stage",
}
# Go/no-go: each of these must be usable in at least this share of results.
GATE_FIELDS = ("founded_year", "country", "employees", "funding")
GATE_MIN_FILL_RATE = 0.30
PRICE_PER_SEARCH_USD = 0.007  # Exa list price: $7 per 1k requests, up to 10 results


def select_queries(
    queries: Iterable[BenchmarkQuery], count: int, seed: int
) -> list[BenchmarkQuery]:
    """Pick `count` retrieval queries with checkable constraints, spread across buckets.

    Queries are grouped by bucket, each group is shuffled with `seed`, and one query is
    taken from each group in turn until `count` is reached. Deterministic for a given seed.
    """
    by_bucket: dict[str, list[BenchmarkQuery]] = {}
    for query in queries:
        if query.track == "retrieval" and CHECKABLE_KEYS.keys() & query.constraints.keys():
            by_bucket.setdefault(query.bucket, []).append(query)
    rng = random.Random(seed)
    groups = [sorted(group, key=lambda q: q.query_id) for _, group in sorted(by_bucket.items())]
    for group in groups:
        rng.shuffle(group)
    selected: list[BenchmarkQuery] = []
    while len(selected) < count and any(groups):
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
    return selected


def estimated_cost_usd(query_count: int) -> float:
    return query_count * PRICE_PER_SEARCH_USD


@dataclass(frozen=True)
class ProbeSummary:
    """Aggregate, committable results: counts, rates, formats. No raw responses."""

    seed: int
    query_ids: list[str]
    results: int
    results_with_company: int
    fill_rates: dict[str, float]
    gate_passed: bool
    gate_min_fill_rate: float
    gate_fill_rates: dict[str, float]
    value_samples: dict[str, dict[str, int]]
    cost_usd_reported_total: float
    calls_with_cost: int
    latency_ms_median: float | None
    latency_ms_max: float | None


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    fill_rates: Mapping[str, float]  # every GATE_FIELDS entry, in order


def decide_gate(report: CoverageReport) -> GateDecision:
    rates = {name: report.fill_rate(name) for name in GATE_FIELDS}
    return GateDecision(
        passed=all(rate >= GATE_MIN_FILL_RATE for rate in rates.values()),
        fill_rates=rates,
    )


def summarize(
    selected: Sequence[BenchmarkQuery],
    calls: Sequence[SearchCall],
    report: CoverageReport,
    decision: GateDecision,
    *,
    seed: int,
    sample_limit: int = 10,
) -> ProbeSummary:
    costs = [call.cost_dollars for call in calls if call.cost_dollars is not None]
    latencies = [call.latency_ms for call in calls]
    return ProbeSummary(
        seed=seed,
        query_ids=[query.query_id for query in selected],
        results=report.results,
        results_with_company=report.with_company,
        fill_rates={name: report.fill_rate(name) for name in sorted(report.filled)},
        gate_passed=decision.passed,
        gate_min_fill_rate=GATE_MIN_FILL_RATE,
        gate_fill_rates=dict(decision.fill_rates),
        value_samples={
            name: dict(counter.most_common(sample_limit))
            for name, counter in report.samples.items()
        },
        cost_usd_reported_total=sum(costs),
        calls_with_cost=len(costs),
        latency_ms_median=statistics.median(latencies) if latencies else None,
        latency_ms_max=max(latencies, default=None),
    )
