"""Grade Exa Agent's companies with the same engine: look up each domain's entity, judge it."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from exa_bench.agent import AgentCompany, CachedRun, parse_run
from exa_bench.benchmark_data import COMMIT, BenchmarkQuery
from exa_bench.grader import filters_from_constraints
from exa_bench.response_cache import CachedSearch
from exa_filters.constraints import Verdict
from exa_filters.evaluate import evaluate_filters
from exa_filters.response import company_entity, search_results
from exa_filters.results import canonical_host

K = 10
type Lookup = Callable[[str, str], CachedSearch]  # (query text, domain) -> one /search call


@dataclass(frozen=True)
class PlannedLookup:
    query: str  # what to search for: the company's name, or its domain when unnamed
    domain: str  # the canonical host the search is restricted to


def lookup_plan(cached: CachedRun) -> tuple[list[PlannedLookup], int, int]:
    """One lookup per distinct domain: (lookups, duplicate domains, domains that do not parse)."""
    seen: set[str] = set()
    lookups: list[PlannedLookup] = []
    duplicates = unparsable = 0
    for company in parse_run(cached.result.body).companies:
        host = canonical_host(company.domain)
        if host is None:
            unparsable += 1
        elif host in seen:
            duplicates += 1
        else:
            seen.add(host)
            lookups.append(PlannedLookup(_lookup_query(company, host), host))
    return lookups, duplicates, unparsable


def _lookup_query(company: AgentCompany, host: str) -> str:
    return company.name or host


@dataclass(frozen=True)
class GradedAgentOutcome:
    query_id: str
    split: str
    clean: bool
    effort: str
    status: str
    companies: int  # returned with a usable domain
    duplicates: int  # repeated domains, dropped
    lookups: int  # distinct domains looked up
    found: int  # distinct domains with a company entity
    not_found: int  # no entity on the domain, or a domain that does not parse
    satisfying: int
    violating: int
    unevaluable: int  # entity found, a needed field missing
    accepted_strict: int  # satisfying, at most K
    accepted_lenient: int  # plus unevaluable and not found, at most K, as the lenient policy would
    cost_usd: float | None  # the Agent run, as Exa reported it: what a user pays
    lookup_cost_usd: float  # our grading lookups, Exa-reported; not part of the user's cost
    lookup_spent_usd: float  # the part of lookup_cost_usd for lookups sent this time
    lookups_from_cache: int
    server_ms: float | None
    client_ms: float

    @property
    def filled_strict(self) -> bool:
        return self.accepted_strict >= K

    @property
    def filled_lenient(self) -> bool:
        return self.accepted_lenient >= K


def grade_run(
    query: BenchmarkQuery,
    clean: bool,
    effort: str,
    cached: CachedRun,
    lookup: Lookup,
    *,
    employee_tolerance: float = 0.0,
) -> GradedAgentOutcome:
    """Judge every distinct company the run returned against the query's filters."""
    run = parse_run(cached.result.body)
    filters = filters_from_constraints(query.constraints, employee_tolerance=employee_tolerance)
    lookups, duplicates, unparsable = lookup_plan(cached)
    found = not_found = satisfying = violating = unevaluable = 0
    lookup_cost = lookup_spent = 0.0
    lookups_from_cache = 0
    for planned in lookups:
        item = lookup(planned.query, planned.domain)
        lookup_cost += item.call.cost_dollars or 0.0
        if item.from_cache:
            lookups_from_cache += 1
        else:
            lookup_spent += item.call.cost_dollars or 0.0
        results = search_results(item.call.body)
        entity = company_entity(results[0]) if results else None
        if entity is None:
            not_found += 1
            continue
        found += 1
        verdicts = evaluate_filters(filters, entity.properties)
        if Verdict.FAIL in verdicts:
            violating += 1
        elif Verdict.UNKNOWN in verdicts:
            unevaluable += 1
        else:
            satisfying += 1
    not_found += unparsable
    return GradedAgentOutcome(
        query_id=query.query_id,
        split=query.split,
        clean=clean,
        effort=effort,
        status=run.status,
        companies=len(run.companies),
        duplicates=duplicates,
        lookups=len(lookups),
        found=found,
        not_found=not_found,
        satisfying=satisfying,
        violating=violating,
        unevaluable=unevaluable,
        accepted_strict=min(K, satisfying),
        accepted_lenient=min(K, satisfying + unevaluable + not_found),
        cost_usd=run.cost_dollars,
        lookup_cost_usd=lookup_cost,
        lookup_spent_usd=lookup_spent,
        lookups_from_cache=lookups_from_cache,
        server_ms=run.server_ms,
        client_ms=cached.result.client_ms,
    )


@dataclass(frozen=True)
class GradedMetadata:
    benchmark_commit: str
    run_date: str  # UTC, YYYY-MM-DD
    effort: str
    k: int
    employee_tolerance: float
    queries: int
    subset_seed: int
    lookups: int
    lookups_sent: int  # lookups not served from the cache this time
    lookup_spent_usd: float  # Exa-reported cost of those lookups


@dataclass(frozen=True)
class GradedRecord:
    metadata: GradedMetadata
    outcomes: tuple[GradedAgentOutcome, ...]


def build_graded_record(
    effort: str,
    outcomes: Sequence[GradedAgentOutcome],
    *,
    employee_tolerance: float,
    seed: int,
    today: date | None = None,
) -> GradedRecord:
    lookups = sum(outcome.lookups for outcome in outcomes)
    from_cache = sum(outcome.lookups_from_cache for outcome in outcomes)
    metadata = GradedMetadata(
        benchmark_commit=COMMIT,
        run_date=(today or datetime.now(UTC).date()).isoformat(),
        effort=effort,
        k=K,
        employee_tolerance=employee_tolerance,
        queries=len(outcomes),
        subset_seed=seed,
        lookups=lookups,
        lookups_sent=lookups - from_cache,
        lookup_spent_usd=sum(outcome.lookup_spent_usd for outcome in outcomes),
    )
    return GradedRecord(metadata, tuple(outcomes))
