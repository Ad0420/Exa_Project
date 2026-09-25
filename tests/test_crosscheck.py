"""Tests for cross-check sampling (pure logic; no network)."""

from collections import Counter

import pytest

from exa_bench.benchmark_data import BenchmarkQuery
from exa_bench.crosscheck import SampleItem, build_items, checkable_subset, select_sample
from exa_bench.grader import ConstraintOutcome, Outcome, ResultVerdict


def query(query_id: str, constraints: dict[str, object]) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id, f"text of {query_id}", "retrieval", "dynamic", "composite", constraints
    )


def result(url: str, employees: int | None, title: str | None = "T") -> dict[str, object]:
    entity = {"type": "company", "properties": {"workforce": {"total": employees}}}
    return {"url": url, "title": title, "entities": [entity]}


def test_checkable_subset_keeps_only_ops_the_grader_can_check() -> None:
    constraints: dict[str, object] = {
        "employees": {"gte": 60, "lte": 100},
        "founded_year": {"not_eq": 2019, "gte": 2015},
        "category": {"contains": "ai"},
        "criteria": [{"description": "x"}],
    }

    assert checkable_subset(constraints) == {
        "employees": {"gte": 60, "lte": 100},
        "founded_year": {"gte": 2015},
    }
    assert checkable_subset({"category": {"contains": "ai"}}) == {}


def test_build_items_grades_each_result_and_records_checkable_outcomes() -> None:
    q = query("q1", {"employees": {"lte": 30}, "category": {"contains": "ai"}})
    body: dict[str, object] = {
        "results": [
            result("https://a.test", 20),
            result("https://b.test", 200, title=None),
            {"no": "url"},
        ]
    }

    items = build_items([q], [body])

    assert [(i.url, i.verdict, i.title) for i in items] == [
        ("https://a.test", ResultVerdict.SATISFIES, "T"),
        ("https://b.test", ResultVerdict.VIOLATES, ""),
    ]
    assert items[0].constraints == {"employees": {"lte": 30}}
    assert items[1].outcomes == (ConstraintOutcome("employees", "lte", Outcome.FAIL),)
    assert items[0].query_text == "text of q1"


def test_build_items_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="queries but"):
        build_items([query("q", {"employees": {"lte": 1}})], [])


def item(n: int, verdict: ResultVerdict) -> SampleItem:
    return SampleItem(f"q{n // 10}", "t", {}, f"https://{n}.test", "", verdict, ())


ITEMS = (
    [item(n, ResultVerdict.VIOLATES) for n in range(50)]
    + [item(n, ResultVerdict.SATISFIES) for n in range(50, 100)]
    + [item(n, ResultVerdict.UNEVALUABLE) for n in range(100, 105)]
)


def test_sample_is_stratified_capped_and_without_replacement() -> None:
    strata = {
        ResultVerdict.VIOLATES: 10,
        ResultVerdict.SATISFIES: 20,
        ResultVerdict.UNEVALUABLE: 60,
    }

    sample = select_sample(ITEMS, seed=1, strata=strata)

    assert Counter(i.verdict for i in sample) == Counter(
        {ResultVerdict.VIOLATES: 10, ResultVerdict.SATISFIES: 20, ResultVerdict.UNEVALUABLE: 5}
    )
    assert len({i.url for i in sample}) == len(sample)


def test_sample_is_deterministic_and_order_independent() -> None:
    first = select_sample(ITEMS, seed=7)
    second = select_sample(list(reversed(ITEMS)), seed=7)

    assert first == second
    assert select_sample(ITEMS, seed=8) != first
