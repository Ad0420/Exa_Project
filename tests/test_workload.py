"""Tests for the evaluation workload: selection and loading from the cache."""

from pathlib import Path

import httpx
import pytest

from exa_bench.constraints.benchmark import CATEGORY, NUM_RESULTS, SEARCH_TYPE
from exa_bench.core.benchmark_data import BenchmarkQuery
from exa_bench.core.response_cache import cached_search
from exa_bench.policies import workload as workload_module
from exa_bench.policies.workload import load_workload, select_workload


def query(query_id: str) -> BenchmarkQuery:
    constraints: dict[str, object] = {"employees": {"lte": 100}}
    return BenchmarkQuery(query_id, query_id, "retrieval", "dynamic", "bucket", constraints)


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
    assert workload.seed == 1
    assert workload == again
    samples = {select_workload(queries, bodies, clean_count=2, seed=s).clean_ids for s in range(5)}
    assert len(samples) > 1
    everything = select_workload(queries, bodies, clean_count=9, seed=1)
    assert everything.clean_ids == {"q1", "q3", "q4", "q5"}
    assert len(everything.queries) == 6
    with pytest.raises(ValueError, match="queries but"):
        select_workload(queries, bodies[:1], clean_count=2, seed=1)


def no_network() -> httpx.Client:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.Client(transport=httpx.MockTransport(refuse))


def cache_top_10(cache_dir: Path, text: str, response: dict[str, object]) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    )
    with client:
        cached_search(
            cache_dir / "exa",
            client,
            "test-key",
            text,
            category=CATEGORY,
            num_results=NUM_RESULTS,
            search_type=SEARCH_TYPE,
        )


def test_load_workload_reads_the_benchmark_and_cached_top_10s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rag = BenchmarkQuery("r0", "r0", "rag", "dynamic", "bucket", {"employees": {"lte": 100}})
    queries = [query("q0"), query("q1"), rag]  # the rag-track query is not gradable
    monkeypatch.setattr(workload_module, "load_company_queries", lambda cache_dir, client: queries)
    cache_top_10(tmp_path, "q0", body(50, 500))  # violates: in the workload
    cache_top_10(tmp_path, "q1", body(10, 20))  # clean: sampled

    with no_network() as client:
        workload = load_workload(tmp_path, client)

    assert [q.query_id for q in workload.queries] == ["q0", "q1"]
    assert workload.clean_ids == {"q1"}


def test_load_workload_names_a_query_without_a_cached_top_10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workload_module, "load_company_queries", lambda cache_dir, client: [query("q0")]
    )

    with no_network() as client, pytest.raises(FileNotFoundError, match="q0"):
        load_workload(tmp_path, client)
