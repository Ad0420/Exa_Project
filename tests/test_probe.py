"""Tests for the Day-0 coverage probe logic (pure; no network)."""

from collections import Counter
from dataclasses import asdict

import pytest

from exa_bench.core.benchmark_data import BenchmarkQuery
from exa_bench.coverage import CoverageReport
from exa_bench.probe import (
    GATE_MIN_FILL_RATE,
    decide_gate,
    estimated_cost_usd,
    select_queries,
    summarize,
)
from exa_filters.api import ApiCall


def query(
    query_id: str, bucket: str, constraints: dict[str, object], track: str = "retrieval"
) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        text=query_id,
        track=track,
        split="static",
        bucket=bucket,
        constraints=constraints,
    )


CHECKABLE = [
    *(query(f"geo_{i}", "industry_geo", {"country": {"eq": "Israel"}}) for i in range(5)),
    *(query(f"year_{i}", "founded_year", {"founded_year": {"eq": 2020}}) for i in range(5)),
    *(query(f"emp_{i}", "employee_count", {"employees": {"lte": 30}}) for i in range(5)),
]
NOT_CHECKABLE = [
    query("named_1", "named_lookup", {"company": {"homepage": "https://x.test"}}),
    query("sem_1", "semantic", {"criteria": [{"description": "AI infra"}]}),
    query("rag_1", "founding", {"founded_year": {"eq": 2020}}, track="rag"),
]


def test_selects_only_checkable_retrieval_queries() -> None:
    selected = select_queries(CHECKABLE + NOT_CHECKABLE, count=100, seed=1)

    assert {q.query_id for q in selected} == {q.query_id for q in CHECKABLE}


def test_selection_spreads_across_buckets_and_is_deterministic() -> None:
    first = select_queries(CHECKABLE, count=6, seed=7)
    second = select_queries(reversed(CHECKABLE), count=6, seed=7)

    assert Counter(q.bucket for q in first) == Counter(
        {"industry_geo": 2, "founded_year": 2, "employee_count": 2}
    )
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert select_queries(CHECKABLE, count=6, seed=8) != first


def test_selection_stops_when_queries_run_out() -> None:
    assert len(select_queries(CHECKABLE, count=50, seed=1)) == len(CHECKABLE)


def test_estimated_cost_uses_list_price() -> None:
    assert estimated_cost_usd(20) == pytest.approx(0.14)


def report(results: int, filled: dict[str, int]) -> CoverageReport:
    coverage = CoverageReport(results=results, with_company=results)
    coverage.filled.update(filled)
    coverage.samples["country"].update({"Germany": 2, "US": 1})
    return coverage


def test_gate_passes_only_when_every_field_meets_the_threshold() -> None:
    at_threshold = round(GATE_MIN_FILL_RATE * 10)
    passing = report(
        10, {"founded_year": 10, "country": 10, "employees": at_threshold, "funding": 10}
    )
    failing = report(
        10, {"founded_year": 10, "country": 10, "employees": at_threshold - 1, "funding": 10}
    )

    assert decide_gate(passing).passed
    assert not decide_gate(failing).passed
    assert decide_gate(failing).fill_rates["employees"] == pytest.approx((at_threshold - 1) / 10)


def test_gate_fails_on_empty_report() -> None:
    assert not decide_gate(CoverageReport()).passed


def call(cost: float | None, latency: float) -> ApiCall:
    return ApiCall(
        body={}, latency_ms=latency, attempts=1, request_id=None, queue_ms=None, cost_dollars=cost
    )


def test_summary_is_aggregate_only() -> None:
    selected = CHECKABLE[:2]
    calls = [call(0.007, 300.0), call(None, 500.0), call(None, 1900.0)]
    coverage = report(2, {"founded_year": 2, "country": 1, "employees": 0, "funding": 1})

    summary = summarize(selected, calls, coverage, decide_gate(coverage), seed=3)

    assert summary.query_ids == ["geo_0", "geo_1"]
    assert summary.fill_rates == {
        "country": 0.5,
        "employees": 0.0,
        "founded_year": 1.0,
        "funding": 0.5,
    }
    assert not summary.gate_passed
    assert summary.gate_min_fill_rate == GATE_MIN_FILL_RATE
    assert summary.gate_fill_rates == {
        "founded_year": 1.0,
        "country": 0.5,
        "employees": 0.0,
        "funding": 0.5,
    }
    assert summary.value_samples["country"] == {"Germany": 2, "US": 1}
    assert (summary.cost_usd_reported_total, summary.calls_with_cost) == (0.007, 1)
    assert (summary.latency_ms_median, summary.latency_ms_max) == (500.0, 1900.0)
    assert "body" not in str(asdict(summary))


def test_value_samples_keep_only_the_most_common() -> None:
    coverage = CoverageReport(results=1, with_company=1)
    coverage.samples["country"].update({"Germany": 3, "France": 2, "Spain": 1})

    summary = summarize([], [], coverage, decide_gate(coverage), seed=0, sample_limit=2)

    assert summary.value_samples["country"] == {"Germany": 3, "France": 2}


def test_summary_with_no_calls_has_no_latency() -> None:
    summary = summarize([], [], CoverageReport(), decide_gate(CoverageReport()), seed=0)

    assert (summary.latency_ms_median, summary.latency_ms_max) == (None, None)
    assert (summary.cost_usd_reported_total, summary.calls_with_cost) == (0, 0)
