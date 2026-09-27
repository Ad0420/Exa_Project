"""Tests for grading Agent companies through entity lookups (no network)."""

from dataclasses import replace
from datetime import date

import pytest

from exa_bench.agent import CachedRun, RunResult
from exa_bench.agent_grade import (
    GradedAgentOutcome,
    PlannedLookup,
    build_graded_record,
    grade_run,
    lookup_plan,
)
from exa_bench.benchmark_data import COMMIT, BenchmarkQuery
from exa_bench.response_cache import CachedSearch
from exa_filters.api import ApiCall

QUERY = BenchmarkQuery(
    "q1", "small firms", "retrieval", "dynamic", "b", {"employees": {"lte": 100}}
)


def run_with(*companies: tuple[str, str], cost: float = 0.045) -> CachedRun:
    body: dict[str, object] = {
        "id": "r",
        "status": "completed",
        "createdAt": "2026-05-07T18:31:00Z",
        "completedAt": "2026-05-07T18:31:30Z",
        "costDollars": {"total": cost},
        "output": {
            "structured": {"companies": [{"company": n, "domain": d} for n, d in companies]}
        },
    }
    return CachedRun(RunResult(body, 31000.0, 5), from_cache=True)


def entity_result(host: str, employees: int | None) -> dict[str, object]:
    properties = {"workforce": {"total": employees}} if employees is not None else {}
    return {
        "url": f"https://{host}/about",
        "entities": [{"type": "company", "properties": properties}],
    }


class Lookups:
    """Scripted entity lookups by domain; a missing domain answers with no results."""

    def __init__(
        self, bodies: dict[str, dict[str, object]], cached: frozenset[str] = frozenset()
    ) -> None:
        self.bodies = bodies
        self.cached = cached
        self.seen: list[tuple[str, str]] = []

    def __call__(self, query: str, domain: str) -> CachedSearch:
        self.seen.append((query, domain))
        body = self.bodies.get(domain, {"results": []})
        call = ApiCall(body, 100.0, 1, None, None, 0.007)
        return CachedSearch(call, from_cache=domain in self.cached)


RUN = run_with(
    ("Acme", "https://www.acme.com/"),
    ("Big Corp", "big.com"),
    ("", "vague.io"),
    ("Gone", "gone.example"),
    ("Acme again", "acme.com"),
    ("Broken", "http://"),
)
BODIES: dict[str, dict[str, object]] = {
    "acme.com": {"results": [entity_result("acme.com", 50)]},
    "big.com": {"results": [entity_result("big.com", 500)]},
    "vague.io": {"results": [entity_result("vague.io", None)]},
}


def test_lookup_plan_dedupes_domains_and_names_the_query() -> None:
    lookups, duplicates, unparsable = lookup_plan(RUN)

    assert lookups == [
        PlannedLookup("Acme", "acme.com"),
        PlannedLookup("Big Corp", "big.com"),
        PlannedLookup("vague.io", "vague.io"),
        PlannedLookup("Gone", "gone.example"),
    ]
    assert (duplicates, unparsable) == (1, 1)


def test_grade_run_judges_each_distinct_company_by_its_entity() -> None:
    lookups = Lookups(BODIES, cached=frozenset({"acme.com"}))

    outcome = grade_run(QUERY, False, "low", RUN, lookups)

    assert lookups.seen == [
        ("Acme", "acme.com"),
        ("Big Corp", "big.com"),
        ("vague.io", "vague.io"),
        ("Gone", "gone.example"),
    ]
    assert replace(outcome, lookup_cost_usd=0.0, lookup_spent_usd=0.0) == GradedAgentOutcome(
        query_id="q1",
        split="dynamic",
        clean=False,
        effort="low",
        status="completed",
        companies=6,
        duplicates=1,
        lookups=4,
        found=3,
        not_found=2,
        satisfying=1,
        violating=1,
        unevaluable=1,
        accepted_strict=1,
        accepted_lenient=4,
        cost_usd=0.045,
        lookup_cost_usd=0.0,
        lookup_spent_usd=0.0,
        lookups_from_cache=1,
        server_ms=30000.0,
        client_ms=31000.0,
    )
    assert outcome.lookup_cost_usd == pytest.approx(0.028)
    assert outcome.lookup_spent_usd == pytest.approx(0.021)
    assert not outcome.filled_strict
    assert not outcome.filled_lenient


def test_accepted_counts_are_capped_at_k() -> None:
    run = run_with(*((f"Co {i}", f"co{i}.com") for i in range(11)))
    lookups = Lookups(
        {f"co{i}.com": {"results": [entity_result(f"co{i}.com", 5)]} for i in range(11)}
    )

    outcome = grade_run(QUERY, False, "low", run, lookups)

    assert (outcome.companies, outcome.satisfying) == (11, 11)
    assert (outcome.accepted_strict, outcome.accepted_lenient) == (10, 10)
    assert outcome.filled_strict
    assert outcome.filled_lenient


def test_grade_run_applies_the_headcount_tolerance() -> None:
    run = run_with(("Edge", "edge.com"))
    lookups = Lookups({"edge.com": {"results": [entity_result("edge.com", 110)]}})

    exact = grade_run(QUERY, False, "low", run, lookups)
    tolerant = grade_run(QUERY, False, "low", run, lookups, employee_tolerance=0.2)

    assert (exact.violating, exact.satisfying) == (1, 0)
    assert (tolerant.violating, tolerant.satisfying) == (0, 1)


def test_graded_record_counts_lookups_sent_this_time() -> None:
    fresh = grade_run(QUERY, False, "low", RUN, Lookups(BODIES))
    cached = grade_run(
        QUERY, True, "low", RUN, Lookups(BODIES, cached=frozenset(BODIES) | {"gone.example"})
    )

    record = build_graded_record(
        "low", [fresh, cached], employee_tolerance=0.0, seed=3, today=date(2026, 1, 1)
    )

    meta = record.metadata
    assert (meta.benchmark_commit, meta.run_date, meta.effort, meta.k) == (
        COMMIT,
        "2026-01-01",
        "low",
        10,
    )
    assert (meta.employee_tolerance, meta.queries, meta.subset_seed) == (0.0, 2, 3)
    assert (meta.lookups, meta.lookups_sent) == (8, 4)
    assert meta.lookup_spent_usd == pytest.approx(0.028)
    assert record.outcomes == (fresh, cached)
