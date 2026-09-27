"""Policy evaluation: run each fetch policy on the same queries through the filters feature."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from exa_bench.constraints.benchmark import non_clean_queries
from exa_bench.constraints.grader import filters_from_constraints
from exa_bench.core.benchmark_data import COMMIT, BenchmarkQuery
from exa_bench.core.response_cache import CachedSearch
from exa_bench.core.sampling import seeded_subset
from exa_filters.api import ApiCall
from exa_filters.client import PlanTrace, SearchResponse, filtered_search
from exa_filters.evaluate import Filter
from exa_filters.planner import (
    PRICE_EXTRA_RESULT_USD,
    AdaptivePlanner,
    Budget,
    FixedPlanner,
    Planner,
    prior_planner,
)
from exa_filters.results import NullPolicy

CLEAN_SAMPLE_COUNT = 30
CLEAN_SAMPLE_SEED = 20260926
K = 10  # results every policy tries to deliver
BUDGET = Budget(max_calls=3)

# Base price per request, up to 10 results, by search type (exa.ai/docs/admin/pricing, 2026-09-27).
BASE_PRICE_USD = {"auto": 0.007, "deep": 0.012}

type PlannerFactory = Callable[[Sequence[Filter], NullPolicy], Planner]
# (query text, numResults, search type) -> the cached or fresh call
type Fetch = Callable[[str, int, str], CachedSearch]
type Read = Callable[
    [str, int, str], ApiCall | None
]  # the cached call for these parameters, if any


@dataclass(frozen=True)
class Policy:
    planner: PlannerFactory
    search_type: str = "auto"


POLICIES: dict[str, Policy] = {
    "baseline": Policy(lambda filters, null_policy: FixedPlanner(10)),  # Exa's top 10, filtered
    "fixed-25": Policy(lambda filters, null_policy: FixedPlanner(25)),
    "prior": Policy(
        lambda filters, null_policy: prior_planner(filters, null_policy, fallback=None)
    ),
    "adaptive": Policy(lambda filters, null_policy: AdaptivePlanner((10, 25, 100))),
    "deep": Policy(lambda filters, null_policy: FixedPlanner(10), search_type="deep"),
}


@dataclass(frozen=True)
class Workload:
    queries: tuple[BenchmarkQuery, ...]  # by query id
    clean_ids: frozenset[str]  # the sampled queries whose cached top-10 already satisfied
    seed: int  # of the clean sample


def select_workload(
    queries: Sequence[BenchmarkQuery],
    bodies: Sequence[Mapping[str, object]],
    *,
    clean_count: int = CLEAN_SAMPLE_COUNT,
    seed: int = CLEAN_SAMPLE_SEED,
) -> Workload:
    """Every query whose cached top-10 is not clean, plus a seeded sample of clean ones.

    The clean sample answers whether over-fetching disturbs a top-10 that needed no help.
    """
    non_clean = non_clean_queries(queries, bodies)
    non_clean_ids = {query.query_id for query in non_clean}
    clean = [query for query in queries if query.query_id not in non_clean_ids]
    sample = seeded_subset(clean, count=clean_count, seed=seed)
    selected = sorted(non_clean + sample, key=lambda query: query.query_id)
    return Workload(tuple(selected), frozenset(query.query_id for query in sample), seed)


@dataclass(frozen=True)
class Run:
    policy: str
    null_policy: NullPolicy
    employee_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; choose from {', '.join(POLICIES)}")


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    split: str
    clean: bool  # from the clean sample: the cached top-10 already satisfied
    policy: str
    null_policy: str
    filters: int
    expected_pass_rate: float  # the prior planner's estimate, for the independence check
    accepted: int  # results delivered, at most K
    seen: int  # results fetched over every call
    duplicates: int
    violating: int
    unevaluable: int
    satisfying: int
    requested: tuple[int, ...]  # numResults per call
    returned: tuple[int, ...]
    from_cache: tuple[bool, ...]
    cost_usd: float  # Exa-reported cost of the calls, cached ones included
    spent_usd: float  # the part of cost_usd for calls sent this run rather than read from cache
    latency_ms: float  # per-call latencies as measured when each was first fetched
    stopped_by: str

    @property
    def filled(self) -> bool:
        return self.accepted >= K


def run_policy(fetch: Fetch, workload: Workload, run: Run) -> list[QueryOutcome]:
    """Run one policy over the workload. Every call goes through `fetch`: the cache or Exa."""
    return [
        _run_query(fetch, query, query.query_id in workload.clean_ids, run)
        for query in workload.queries
    ]


class UncachedCall(Exception):
    """A dry-run fetch met a call that is not on disk; nothing is sent."""

    def __init__(self, query: str, num_results: int) -> None:
        super().__init__(f"uncached call: {query!r} at numResults={num_results}")
        self.query = query
        self.num_results = num_results


@dataclass(frozen=True)
class PlannedCall:
    query_id: str
    num_results: int
    search_type: str = "auto"

    @property
    def list_price_usd(self) -> float:
        extra = max(0, self.num_results - 10) * PRICE_EXTRA_RESULT_USD
        return BASE_PRICE_USD[self.search_type] + extra


def plan_calls(read: Read, workload: Workload, run: Run) -> list[PlannedCall]:
    """The calls a run would send now: each query's first call that is not cached.

    A query's later calls can depend on that response, so for a policy that escalates this
    is a lower bound; plan again once those calls are cached.
    """
    fetch = _dry_run_fetch(read)
    search_type = POLICIES[run.policy].search_type
    planned: list[PlannedCall] = []
    for query in workload.queries:
        try:
            _run_query(fetch, query, query.query_id in workload.clean_ids, run)
        except UncachedCall as uncached:
            planned.append(PlannedCall(query.query_id, uncached.num_results, search_type))
    return planned


def _dry_run_fetch(read: Read) -> Fetch:
    def fetch(query: str, num_results: int, search_type: str) -> CachedSearch:
        call = read(query, num_results, search_type)
        if call is None:
            raise UncachedCall(query, num_results)
        return CachedSearch(call, from_cache=True)

    return fetch


class _RecordingSearch:
    """The search function for filtered_search; notes whether each call came from the cache."""

    def __init__(self, fetch: Fetch, search_type: str) -> None:
        self._fetch = fetch
        self._search_type = search_type
        self.from_cache: list[bool] = []

    def __call__(self, query: str, num_results: int) -> ApiCall:
        item = self._fetch(query, num_results, self._search_type)
        self.from_cache.append(item.from_cache)
        return item.call


def _run_query(fetch: Fetch, query: BenchmarkQuery, clean: bool, run: Run) -> QueryOutcome:
    filters = filters_from_constraints(query.constraints, employee_tolerance=run.employee_tolerance)
    policy = POLICIES[run.policy]
    search = _RecordingSearch(fetch, policy.search_type)
    response = filtered_search(
        search,
        query.text,
        filters,
        k=K,
        null_policy=run.null_policy,
        planner=policy.planner(filters, run.null_policy),
        budget=BUDGET,
    )
    return _outcome(query, clean, run, filters, response, search.from_cache)


def _outcome(
    query: BenchmarkQuery,
    clean: bool,
    run: Run,
    filters: Sequence[Filter],
    response: SearchResponse,
    from_cache: Sequence[bool],
) -> QueryOutcome:
    selection, trace = response.selection, response.trace
    unique = selection.seen - selection.excluded - selection.duplicates
    return QueryOutcome(
        query_id=query.query_id,
        split=query.split,
        clean=clean,
        policy=run.policy,
        null_policy=run.null_policy.value,
        filters=len(filters),
        expected_pass_rate=prior_planner(filters, run.null_policy).pass_rate,
        accepted=len(selection.results),
        seen=selection.seen,
        duplicates=selection.duplicates,
        violating=selection.violating,
        unevaluable=selection.unevaluable,
        satisfying=unique - selection.violating - selection.unevaluable,
        requested=tuple(call.requested for call in trace.calls),
        returned=tuple(call.returned for call in trace.calls),
        from_cache=tuple(from_cache),
        cost_usd=trace.cost_usd,
        spent_usd=_spent(trace, from_cache),
        latency_ms=trace.latency_ms,
        stopped_by=trace.stopped_by.value,
    )


def _spent(trace: PlanTrace, from_cache: Sequence[bool]) -> float:
    """The cost of the calls that were sent rather than served from the cache."""
    calls = zip(trace.calls, from_cache, strict=True)
    return sum(call.cost_usd for call, cached in calls if not cached)


@dataclass(frozen=True)
class PlanSummary:
    calls: int
    by_num_results: dict[int, int]  # numResults -> calls, ascending
    list_price_usd: float


def summarize_plan(planned: Sequence[PlannedCall]) -> PlanSummary:
    by_num_results: dict[int, int] = {}
    for call in planned:
        by_num_results[call.num_results] = by_num_results.get(call.num_results, 0) + 1
    return PlanSummary(
        len(planned),
        dict(sorted(by_num_results.items())),
        sum(call.list_price_usd for call in planned),
    )


@dataclass(frozen=True)
class RunMetadata:
    benchmark_commit: str
    run_date: str  # UTC, YYYY-MM-DD
    policy: str
    null_policy: str
    employee_tolerance: float
    k: int
    queries: int
    clean_sample: int
    clean_sample_seed: int
    planned_calls: int  # what the dry run listed before this run
    planned_list_price_usd: float
    calls_sent: int  # calls not served from the cache during this run
    spent_usd: float  # Exa-reported cost of those calls
    search_type: str = "auto"  # records written before this field existed are all auto


@dataclass(frozen=True)
class RunRecord:
    metadata: RunMetadata
    outcomes: tuple[QueryOutcome, ...]


def build_record(
    run: Run,
    workload: Workload,
    planned: Sequence[PlannedCall],
    outcomes: Sequence[QueryOutcome],
    *,
    today: date | None = None,
) -> RunRecord:
    plan = summarize_plan(planned)
    metadata = RunMetadata(
        benchmark_commit=COMMIT,
        run_date=(today or datetime.now(UTC).date()).isoformat(),
        policy=run.policy,
        null_policy=run.null_policy.value,
        employee_tolerance=run.employee_tolerance,
        k=K,
        queries=len(workload.queries),
        clean_sample=len(workload.clean_ids),
        clean_sample_seed=workload.seed,
        planned_calls=plan.calls,
        planned_list_price_usd=plan.list_price_usd,
        calls_sent=sum(not cached for outcome in outcomes for cached in outcome.from_cache),
        spent_usd=sum(outcome.spent_usd for outcome in outcomes),
        search_type=POLICIES[run.policy].search_type,
    )
    return RunRecord(metadata, tuple(outcomes))


def record_name(run: Run) -> str:
    """File name of a run's record: policy and null policy, plus the tolerance when not zero."""
    tolerance = f".tol{run.employee_tolerance:g}" if run.employee_tolerance else ""
    return f"{run.policy}.{run.null_policy.value}{tolerance}.json"
