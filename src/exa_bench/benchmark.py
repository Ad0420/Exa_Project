"""Grade Exa's company search on its own benchmark: every gradable query, top-10 results."""

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from exa_bench.analysis import Analysis, QueryGrade, analyze
from exa_bench.benchmark_data import COMMIT, BenchmarkQuery
from exa_bench.coverage import company_properties, search_results
from exa_bench.grader import checkable_constraints, grade
from exa_bench.response_cache import CachedSearch
from exa_bench.stats import DEFAULT_RESAMPLES

# Identical to the probe's search parameters, so its responses are reused from the cache.
CATEGORY = "company"
NUM_RESULTS = 10
SEARCH_TYPE = "auto"
EMPLOYEE_TOLERANCES = {"strict": 0.0, "tolerant": 0.2}  # 0.2 mirrors Exa's LLM grader


def select_gradable(queries: Iterable[BenchmarkQuery]) -> list[BenchmarkQuery]:
    """Retrieval queries with at least one constraint the typed fields can check, by id."""
    gradable = [
        query
        for query in queries
        if query.track == "retrieval" and checkable_constraints(query.constraints) > 0
    ]
    return sorted(gradable, key=lambda query: query.query_id)


def grade_response(
    query: BenchmarkQuery, body: Mapping[str, object], *, employee_tolerance: float
) -> QueryGrade:
    """Grade every result of one search; a result without a company entity is unevaluable."""
    graded = []
    for result in search_results(body):
        properties = company_properties(result)
        graded.append(
            grade(
                query.constraints,
                properties if properties is not None else {},
                employee_tolerance=employee_tolerance,
            )
        )
    return QueryGrade(query.query_id, query.bucket, query.split, tuple(graded))


@dataclass(frozen=True)
class RunMetadata:
    benchmark_commit: str
    run_date: str  # UTC, YYYY-MM-DD
    category: str
    num_results: int
    search_type: str
    queries: int
    calls_from_cache: int
    cost_usd_reported_total: float
    calls_with_cost: int
    latency_ms_median: float | None  # of the original fetches, cached or not
    latency_ms_max: float | None
    employee_tolerances: dict[str, float]


@dataclass(frozen=True)
class BenchmarkReport:
    metadata: RunMetadata
    strict: Analysis
    tolerant: Analysis


def build_report(
    queries: Sequence[BenchmarkQuery],
    searches: Sequence[CachedSearch],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    today: date | None = None,
) -> BenchmarkReport:
    if len(queries) != len(searches):
        raise ValueError(f"{len(queries)} queries but {len(searches)} searches")
    calls = [item.call for item in searches]
    costs = [call.cost_dollars for call in calls if call.cost_dollars is not None]
    latencies = [call.latency_ms for call in calls]
    metadata = RunMetadata(
        benchmark_commit=COMMIT,
        run_date=(today or datetime.now(UTC).date()).isoformat(),
        category=CATEGORY,
        num_results=NUM_RESULTS,
        search_type=SEARCH_TYPE,
        queries=len(queries),
        calls_from_cache=sum(item.from_cache for item in searches),
        cost_usd_reported_total=sum(costs),
        calls_with_cost=len(costs),
        latency_ms_median=statistics.median(latencies) if latencies else None,
        latency_ms_max=max(latencies, default=None),
        employee_tolerances=dict(EMPLOYEE_TOLERANCES),
    )
    analyses = {
        label: analyze(
            [
                grade_response(query, call.body, employee_tolerance=tolerance)
                for query, call in zip(queries, calls, strict=True)
            ],
            seed=seed,
            resamples=resamples,
        )
        for label, tolerance in EMPLOYEE_TOLERANCES.items()
    }
    return BenchmarkReport(metadata, analyses["strict"], analyses["tolerant"])
