"""Tests for the policy evaluation: workload selection."""

import pytest

from exa_bench.benchmark_data import BenchmarkQuery
from exa_bench.policy import select_workload


def query(query_id: str) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id, query_id, "retrieval", "dynamic", "employee_count", {"employees": {"lte": 100}}
    )


def company(employees: int | None) -> dict[str, object]:
    entity = {"type": "company", "properties": {"workforce": {"total": employees}}}
    return {"url": f"https://{employees}.test", "entities": [entity]}


def body(*employees: int | None) -> dict[str, object]:
    return {"results": [company(e) for e in employees]}


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
