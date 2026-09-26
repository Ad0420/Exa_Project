"""Tests for the company benchmark run (pure logic; no network)."""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx
import pytest

from exa_bench.benchmark import (
    CATEGORY,
    NUM_RESULTS,
    SEARCH_TYPE,
    build_report,
    grade_response,
    load_shallow,
    select_gradable,
)
from exa_bench.benchmark_data import COMMIT, BenchmarkQuery
from exa_bench.exa_api import ApiCall
from exa_bench.response_cache import CachedSearch, cached_search


def query(
    query_id: str, constraints: dict[str, object], track: str = "retrieval"
) -> BenchmarkQuery:
    return BenchmarkQuery(query_id, query_id, track, "dynamic", "employee_count", constraints)


def company(employees: int | None) -> dict[str, object]:
    entity = {"type": "company", "properties": {"workforce": {"total": employees}}}
    return {"url": "https://example.test", "entities": [entity]}


def call(body: dict[str, object], cost: float | None = 0.007, latency: float = 300.0) -> ApiCall:
    return ApiCall(body, latency, 1, None, None, cost)


def test_select_gradable_keeps_checkable_retrieval_queries_sorted_by_id() -> None:
    queries = [
        query("b", {"employees": {"lte": 30}}),
        query("rag", {"employees": {"lte": 30}}, track="rag"),
        query("none", {"category": {"contains": "ai"}}),
        query("a", {"country": {"eq": "France"}, "category": {"contains": "ai"}}),
    ]

    assert [q.query_id for q in select_gradable(queries)] == ["a", "b"]


def test_grade_response_grades_each_result_and_marks_missing_entities_unevaluable() -> None:
    body: dict[str, object] = {
        "results": [company(20), {"url": "https://news.example.test"}, company(200)]
    }

    graded = grade_response(query("q", {"employees": {"lte": 30}}), body, employee_tolerance=0.0)

    assert (graded.query_id, graded.bucket, graded.split) == ("q", "employee_count", "dynamic")
    assert [r.verdict for r in graded.results] == ["satisfies", "unevaluable", "violates"]


def test_grade_response_passes_the_tolerance_through() -> None:
    body: dict[str, object] = {"results": [company(120)]}
    q = query("q", {"employees": {"lte": 100}})

    assert grade_response(q, body, employee_tolerance=0.0).results[0].verdict == "violates"
    assert grade_response(q, body, employee_tolerance=0.2).results[0].verdict == "satisfies"


def test_grade_response_handles_a_body_without_results() -> None:
    assert (
        grade_response(query("q", {"employees": {"lte": 30}}), {}, employee_tolerance=0.0).results
        == ()
    )


def test_build_report_metadata_and_both_analyses() -> None:
    queries = [
        query("a", {"employees": {"lte": 100}}),
        query("b", {"employees": {"gte": 500}}),
        query("c", {"employees": {"lte": 10}}),
    ]
    searches = [
        CachedSearch(
            call({"results": [company(120), company(50)]}, latency=100.0), from_cache=True
        ),
        CachedSearch(call({"results": [company(600)]}, cost=None, latency=900.0), from_cache=False),
        CachedSearch(call({"results": [company(5)]}, latency=200.0), from_cache=True),
    ]

    report = build_report(queries, searches, seed=1, resamples=50, today=date(2026, 9, 24))

    meta = report.metadata
    assert (meta.benchmark_commit, meta.run_date) == (COMMIT, "2026-09-24")
    assert (meta.category, meta.num_results, meta.search_type) == (
        CATEGORY,
        NUM_RESULTS,
        SEARCH_TYPE,
    )
    assert (meta.queries, meta.calls_from_cache) == (3, 2)
    assert meta.cost_usd_reported_total == pytest.approx(0.014)
    assert meta.calls_with_cost == 2
    assert (meta.latency_ms_median, meta.latency_ms_max) == (200.0, 900.0)  # median, not mean
    assert meta.employee_tolerances == {"strict": 0.0, "tolerant": 0.2}
    # 120 employees vs lte 100: violates strictly, satisfies within 20%.
    assert report.strict.verdicts["violates"] == 1
    assert report.tolerant.verdicts["violates"] == 0
    assert report.strict.queries == report.tolerant.queries == 3
    json.dumps(asdict(report))  # must be serializable for the results file


def test_build_report_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="queries but"):
        build_report([query("a", {"employees": {"lte": 1}})], [], seed=1, resamples=50)


def scripted_client(bodies: list[dict[str, object]]) -> httpx.Client:
    pending = list(bodies)
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=pending.pop(0)))
    )


def test_load_shallow_reads_every_querys_cached_top_10(tmp_path: Path) -> None:
    queries = [query("a", {"employees": {"lte": 30}}), query("b", {"employees": {"lte": 30}})]
    body: dict[str, object] = {"results": [company(5)]}
    with scripted_client([body]) as client:
        cached_search(
            tmp_path / "exa",
            client,
            "test-key",
            "a",
            category=CATEGORY,
            num_results=NUM_RESULTS,
            search_type=SEARCH_TYPE,
        )

    with pytest.raises(FileNotFoundError, match="query b"):
        load_shallow(tmp_path, queries)
    shallow = load_shallow(tmp_path, queries[:1])
    assert list(shallow) == ["a"]
    assert shallow["a"].body == body
