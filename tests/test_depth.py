"""Tests for depth-dataset logic: non-clean selection, stability, depth needed."""

import pytest

from exa_bench.constraints.grader import ResultVerdict
from exa_bench.core.benchmark_data import BenchmarkQuery
from exa_bench.depth.stability import (
    PrefixStability,
    depth_needed,
    prefix_stability,
    result_urls,
    summarize_stability,
)

V, S, U = ResultVerdict.VIOLATES, ResultVerdict.SATISFIES, ResultVerdict.UNEVALUABLE


def query(query_id: str) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id, query_id, "retrieval", "dynamic", "employee_count", {"employees": {"lte": 100}}
    )


def company(employees: int | None) -> dict[str, object]:
    entity = {"type": "company", "properties": {"workforce": {"total": employees}}}
    return {"url": f"https://{employees}.test", "entities": [entity]}


def test_result_urls_keeps_rank_order_and_skips_malformed_entries() -> None:
    body: dict[str, object] = {
        "results": [
            {"url": "https://a.test"},
            {"title": "no url"},
            "junk",
            {"url": "https://b.test"},
        ]
    }

    assert result_urls(body) == ["https://a.test", "https://b.test"]
    assert result_urls({}) == []


def test_prefix_stability_measures_overlap_order_and_common_prefix() -> None:
    shallow = ["a", "b", "c", "d"]

    assert prefix_stability("q", shallow, ["a", "b", "c", "d", "e"]) == PrefixStability(
        "q", 4, 1.0, True, 4
    )
    assert prefix_stability("q", shallow, ["a", "b", "d", "c"]) == PrefixStability(
        "q", 4, 1.0, False, 2
    )
    assert prefix_stability("q", shallow, ["a", "x", "y", "z"]) == PrefixStability(
        "q", 4, 0.25, False, 1
    )
    assert prefix_stability("q", shallow, ["a", "b"]) == PrefixStability("q", 4, 0.5, False, 2)
    assert prefix_stability("q", [], ["a"]) == PrefixStability("q", 0, 0.0, False, 0)


def test_summary_gate_uses_both_thresholds() -> None:
    stable = [PrefixStability(f"q{i}", 10, 1.0, True, 10) for i in range(8)]
    reordered = [PrefixStability("r1", 10, 1.0, False, 3), PrefixStability("r2", 10, 0.9, False, 0)]

    passing = summarize_stability(stable + reordered)
    assert passing.mean_overlap == pytest.approx(0.99)
    assert passing.share_same_order == pytest.approx(0.8)
    assert passing.share_overlap_at_least_90 == pytest.approx(1.0)
    assert passing.gate_passed

    failing = summarize_stability(
        stable[:7] + reordered + [PrefixStability("r3", 10, 0.5, False, 0)]
    )
    assert failing.share_same_order == pytest.approx(0.7)
    assert not failing.gate_passed
    assert summarize_stability([]).gate_passed is False
    assert summarize_stability([]).mean_overlap is None


def test_depth_needed_strict_and_lenient() -> None:
    verdicts = [S, V, U, S, S, V, S]

    assert depth_needed(verdicts, 3, lenient=False) == 5
    assert depth_needed(verdicts, 3, lenient=True) == 4
    assert depth_needed(verdicts, 4, lenient=False) == 7
    assert depth_needed(verdicts, 5, lenient=False) is None
    assert depth_needed([], 1, lenient=False) is None


def test_depth_needed_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        depth_needed([S], 0, lenient=False)
