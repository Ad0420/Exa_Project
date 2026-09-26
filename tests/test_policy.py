"""Tests for the policy evaluation: workload selection, running a policy, planning its calls."""

import pytest

from exa_bench.benchmark_data import BenchmarkQuery
from exa_bench.policy import (
    K,
    PlannedCall,
    QueryOutcome,
    Run,
    Workload,
    plan_calls,
    run_policy,
    select_workload,
)
from exa_bench.response_cache import CachedSearch
from exa_filters.api import ApiCall
from exa_filters.planner import LENIENT_PRIORS, STRICT_PRIORS, required_results
from exa_filters.results import NullPolicy


def query(
    query_id: str, constraints: dict[str, object] | None = None, split: str = "dynamic"
) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id, query_id, "retrieval", split, "bucket", constraints or {"employees": {"lte": 100}}
    )


def company(
    employees: int | None, country: str | None = "Germany", name: str | None = None
) -> dict[str, object]:
    properties = {"workforce": {"total": employees}, "headquarters": {"country": country}}
    entity = {"type": "company", "properties": properties}
    return {"url": f"https://{name or f'{employees}-{country}'}.test", "entities": [entity]}


def body(*employees: int | None) -> dict[str, object]:
    return {"results": [company(e) for e in employees]}


def call(
    results: list[dict[str, object]], *, cost: float = 0.007, latency_ms: float = 100.0
) -> ApiCall:
    return ApiCall(
        body={"results": results},
        latency_ms=latency_ms,
        attempts=1,
        request_id=None,
        queue_ms=None,
        cost_dollars=cost,
    )


def passing(count: int) -> list[dict[str, object]]:
    return [company(10 + i) for i in range(count)]


def failing(count: int) -> list[dict[str, object]]:
    return [company(500 + i) for i in range(count)]


class Responses:
    """Scripted calls by (query text, numResults), each marked cached or fresh."""

    def __init__(self, scripted: dict[tuple[str, int], tuple[ApiCall, bool]]) -> None:
        self.scripted = scripted
        self.fetched: list[tuple[str, int]] = []

    def fetch(self, query: str, num_results: int) -> CachedSearch:
        self.fetched.append((query, num_results))
        item, cached = self.scripted[(query, num_results)]
        return CachedSearch(item, from_cache=cached)

    def read(self, query: str, num_results: int) -> ApiCall | None:
        scripted = self.scripted.get((query, num_results))
        return scripted[0] if scripted is not None and scripted[1] else None


WORKLOAD = Workload((query("q0"), query("q1")), frozenset({"q1"}))


def test_workload_is_every_non_clean_query_plus_a_seeded_clean_sample() -> None:
    queries = [query(f"q{i}") for i in range(6)]
    bodies = [body(50, 500), body(50, 60), body(50, None), body(10, 20), body(1, 2), body(3, 4)]
    # q0 violates and q2 has an unknown; q1, q3, q4, q5 are clean

    workload = select_workload(queries, bodies, clean_count=2, seed=1)
    again = select_workload(list(reversed(queries)), list(reversed(bodies)), clean_count=2, seed=1)

    ids = [q.query_id for q in workload.queries]
    assert len(ids) == 4
    assert ids == sorted(ids)
    assert {"q0", "q2"} <= set(ids)
    assert workload.clean_ids == set(ids) - {"q0", "q2"}
    assert len(workload.clean_ids) == 2
    assert workload == again
    samples = {select_workload(queries, bodies, clean_count=2, seed=s).clean_ids for s in range(5)}
    assert len(samples) > 1
    everything = select_workload(queries, bodies, clean_count=9, seed=1)
    assert everything.clean_ids == {"q1", "q3", "q4", "q5"}
    assert len(everything.queries) == 6
    with pytest.raises(ValueError, match="queries but"):
        select_workload(queries, bodies[:1], clean_count=2, seed=1)


def test_fixed_25_records_one_call_per_query() -> None:
    responses = Responses(
        {
            ("q0", 25): (call(passing(10) + failing(3), cost=0.02, latency_ms=300.0), False),
            ("q1", 25): (call(passing(5), cost=0.012), True),
        }
    )

    outcomes = run_policy(responses.fetch, WORKLOAD, Run("fixed-25", NullPolicy.STRICT))

    assert responses.fetched == [("q0", 25), ("q1", 25)]
    assert outcomes[0] == QueryOutcome(
        query_id="q0",
        split="dynamic",
        clean=False,
        policy="fixed-25",
        null_policy="strict",
        filters=1,
        expected_pass_rate=STRICT_PRIORS["employees"],
        accepted=10,
        seen=13,
        duplicates=0,
        violating=3,
        unevaluable=0,
        satisfying=10,
        requested=(25,),
        returned=(13,),
        from_cache=(False,),
        cost_usd=0.02,
        latency_ms=300.0,
        stopped_by="filled",
    )
    assert outcomes[0].filled
    short = outcomes[1]
    assert (short.clean, short.accepted, short.filled) == (True, 5, False)
    assert (short.from_cache, short.cost_usd, short.stopped_by) == ((True,), 0.012, "exhausted")


def test_adaptive_escalates_and_merges_the_calls() -> None:
    responses = Responses(
        {
            ("q0", 10): (call(passing(7) + failing(3)), True),
            ("q0", 25): (call(passing(10) + failing(15), cost=0.022, latency_ms=250.0), False),
            ("q1", 10): (call(passing(10)), True),
        }
    )

    first, second = run_policy(responses.fetch, WORKLOAD, Run("adaptive", NullPolicy.STRICT))

    assert (first.requested, first.returned, first.from_cache) == (
        (10, 25),
        (10, 25),
        (True, False),
    )
    assert (first.seen, first.duplicates, first.violating, first.satisfying) == (35, 10, 15, 10)
    assert (first.accepted, first.stopped_by) == (10, "filled")
    assert first.cost_usd == pytest.approx(0.029)
    assert first.latency_ms == 350.0
    assert (second.requested, second.stopped_by) == ((10,), "filled")


def test_prior_sizes_one_call_and_never_falls_back() -> None:
    n = required_results(K, STRICT_PRIORS["employees"])
    responses = Responses(
        {
            ("q0", n): (call(passing(5) + failing(n - 5)), False),
            ("q1", n): (call(passing(10) + failing(n - 10)), False),
        }
    )

    short, filled = run_policy(responses.fetch, WORKLOAD, Run("prior", NullPolicy.STRICT))

    assert (short.requested, short.accepted, short.stopped_by) == ((n,), 5, "planner")
    assert (filled.requested, filled.accepted, filled.stopped_by) == ((n,), 10, "filled")
    assert short.expected_pass_rate == STRICT_PRIORS["employees"]


def test_lenient_accepts_unknowns_and_uses_lenient_priors() -> None:
    singapore = query("c0", {"country": {"eq": "Singapore"}}, split="static")
    workload = Workload((singapore,), frozenset())
    results = [company(50, "Singapore", name=f"sg{i}") for i in range(8)]
    results += [company(50, None, name=f"nc{i}") for i in range(2)]
    responses = Responses({("c0", 10): (call(results), True)})

    (strict,) = run_policy(responses.fetch, workload, Run("baseline", NullPolicy.STRICT))
    (lenient,) = run_policy(responses.fetch, workload, Run("baseline", NullPolicy.LENIENT))

    assert (strict.accepted, strict.unevaluable, strict.satisfying) == (8, 2, 8)
    assert (strict.stopped_by, strict.filled) == ("planner", False)
    assert (lenient.accepted, lenient.unevaluable, lenient.stopped_by) == (10, 2, "filled")
    assert strict.expected_pass_rate == STRICT_PRIORS["country"]
    assert lenient.expected_pass_rate == LENIENT_PRIORS["country"]
    assert (strict.null_policy, lenient.null_policy, strict.split) == (
        "strict",
        "lenient",
        "static",
    )


def test_employee_tolerance_widens_the_filters() -> None:
    responses = Responses(
        {
            ("q0", 10): (call([*passing(9), company(110)]), True),
            ("q1", 10): (call(passing(10)), True),
        }
    )
    exact_run = Run("baseline", NullPolicy.STRICT)
    tolerant_run = Run("baseline", NullPolicy.STRICT, employee_tolerance=0.2)

    exact = run_policy(responses.fetch, WORKLOAD, exact_run)[0]
    tolerant = run_policy(responses.fetch, WORKLOAD, tolerant_run)[0]

    assert (exact.satisfying, exact.violating) == (9, 1)
    assert (tolerant.satisfying, tolerant.violating) == (10, 0)


def test_plan_calls_lists_each_querys_first_uncached_call() -> None:
    responses = Responses(
        {
            ("q0", 10): (call(passing(7) + failing(3)), True),
            ("q1", 10): (call(passing(10)), True),
        }
    )

    assert plan_calls(responses.read, WORKLOAD, Run("adaptive", NullPolicy.STRICT)) == [
        PlannedCall("q0", 25)
    ]
    assert plan_calls(responses.read, WORKLOAD, Run("fixed-25", NullPolicy.STRICT)) == [
        PlannedCall("q0", 25),
        PlannedCall("q1", 25),
    ]
    assert plan_calls(responses.read, WORKLOAD, Run("baseline", NullPolicy.STRICT)) == []
    assert PlannedCall("q0", 25).list_price_usd == pytest.approx(0.022)


def test_run_rejects_unknown_policies() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        Run("fixed-100", NullPolicy.STRICT)
