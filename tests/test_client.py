"""Tests for the filtered search loop (no network)."""

from collections.abc import Sequence

import pytest

from exa_filters.api import ApiCall
from exa_filters.client import (
    CallTrace,
    PlanTrace,
    SearchResponse,
    Stop,
    filtered_search,
)
from exa_filters.constraints import NumberField, NumberOp
from exa_filters.evaluate import Filter, NumberFilter
from exa_filters.planner import AdaptivePlanner, Budget, CallRecord, FixedPlanner
from exa_filters.results import NullPolicy

SMALL: list[Filter] = [NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100)]


def company(url: str, employees: int | None) -> dict[str, object]:
    properties = {"workforce": {"total": employees}}
    entity = {"type": "company", "id": f"id-{url}", "properties": properties}
    return {"url": f"https://{url}", "title": url, "entities": [entity]}


def passing(prefix: str, count: int) -> list[dict[str, object]]:
    return [company(f"{prefix}{i}.test", 10) for i in range(1, count + 1)]


def failing(prefix: str, count: int) -> list[dict[str, object]]:
    return [company(f"{prefix}{i}.test", 500) for i in range(1, count + 1)]


def reply(
    results: list[dict[str, object]], *, cost: float | None = 0.007, latency_ms: float = 100.0
) -> ApiCall:
    return ApiCall(
        body={"results": results},
        latency_ms=latency_ms,
        attempts=1,
        request_id="req",
        queue_ms=None,
        cost_dollars=cost,
    )


class ScriptedSearch:
    """A search function that answers each call with the next scripted reply."""

    def __init__(self, replies: list[ApiCall]) -> None:
        self.pending = list(replies)
        self.seen: list[tuple[str, int]] = []

    def __call__(self, query: str, num_results: int) -> ApiCall:
        self.seen.append((query, num_results))
        return self.pending.pop(0)


class RecordingPlanner:
    """Asks for 10, then 25, then stops, recording what it was told each time."""

    def __init__(self) -> None:
        self.asked: list[tuple[int, list[CallRecord], int]] = []

    def next_num_results(self, k: int, calls: Sequence[CallRecord], accepted: int) -> int | None:
        self.asked.append((k, [CallRecord(c.requested, c.returned) for c in calls], accepted))
        return (10, 25, None)[len(calls)]


def urls(response: SearchResponse) -> list[str]:
    return [result.url for result in response.results]


def test_one_call_fills_k_and_traces_it() -> None:
    search = ScriptedSearch(
        [reply(passing("a", 10) + failing("f", 2), cost=0.022, latency_ms=120.0)]
    )

    response = filtered_search(search, "q", SMALL, k=10, planner=FixedPlanner(25))

    assert search.seen == [("q", 25)]
    assert urls(response) == [f"https://a{i}.test" for i in range(1, 11)]
    assert response.selection.violating == 2
    assert response.trace == PlanTrace((CallTrace(25, 12, 120.0, 0.022, 1, "req"),), Stop.FILLED)
    assert response.trace.cost_usd == pytest.approx(0.022)
    assert response.trace.latency_ms == 120.0


def test_escalation_merges_calls_and_keeps_results_seen_only_earlier() -> None:
    first = passing("a", 7) + failing("f", 3)  # a7 is not re-fetched by the second call
    second = passing("a", 6) + failing("f", 3) + passing("b", 16)
    search = ScriptedSearch([reply(first), reply(second, cost=0.022, latency_ms=250.0)])

    response = filtered_search(search, "q", SMALL, k=10, planner=AdaptivePlanner((10, 25, 100)))

    assert search.seen == [("q", 10), ("q", 25)]
    expected = [f"https://a{i}.test" for i in range(1, 8)] + [
        f"https://b{i}.test" for i in (1, 2, 3)
    ]
    assert urls(response) == expected
    assert response.selection.seen == 35
    assert response.selection.duplicates == 9
    assert response.trace.stopped_by is Stop.FILLED
    assert [call.requested for call in response.trace.calls] == [10, 25]
    assert response.trace.cost_usd == pytest.approx(0.029)
    assert response.trace.latency_ms == 350.0


def test_one_result_short_still_escalates() -> None:
    first = passing("a", 9) + failing("f", 1)
    second = passing("a", 9) + failing("f", 1) + passing("b", 15)
    search = ScriptedSearch([reply(first), reply(second)])

    response = filtered_search(search, "q", SMALL, k=10, planner=AdaptivePlanner((10, 25)))

    assert search.seen == [("q", 10), ("q", 25)]
    assert len(response.results) == 10
    assert response.trace.stopped_by is Stop.FILLED


def test_stops_when_exa_returns_fewer_than_asked() -> None:
    search = ScriptedSearch([reply(passing("a", 5) + failing("f", 3))])  # 8 of the 10 asked

    response = filtered_search(search, "q", SMALL, k=10, planner=AdaptivePlanner((10, 25)))

    assert search.seen == [("q", 10)]
    assert response.trace.stopped_by is Stop.EXHAUSTED
    assert response.selection.short_by == 5


def test_stops_when_the_planner_has_no_next_call() -> None:
    search = ScriptedSearch([reply(passing("a", 5) + failing("f", 5))])

    response = filtered_search(search, "q", SMALL, k=10, planner=FixedPlanner(10))

    assert response.trace.stopped_by is Stop.PLANNER
    assert len(response.results) == 5


def test_budget_stops_by_calls_or_dollars() -> None:
    short = passing("a", 5) + failing("f", 5)
    escalate = AdaptivePlanner((10, 25))

    by_calls = filtered_search(
        ScriptedSearch([reply(short)]), "q", SMALL, planner=escalate, budget=Budget(max_calls=1)
    )
    assert by_calls.trace.stopped_by is Stop.BUDGET
    assert len(by_calls.trace.calls) == 1

    # Calls made count at Exa's reported cost, the next at list price: 0.05 + 0.022 > 0.06.
    by_dollars = filtered_search(
        ScriptedSearch([reply(short, cost=0.05)]),
        "q",
        SMALL,
        planner=escalate,
        budget=Budget(max_cost_usd=0.06),
    )
    assert by_dollars.trace.stopped_by is Stop.BUDGET

    within = filtered_search(
        ScriptedSearch([reply(short), reply(short)]),
        "q",
        SMALL,
        planner=escalate,
        budget=Budget(max_cost_usd=0.03),
    )
    assert [call.requested for call in within.trace.calls] == [10, 25]

    none = filtered_search(ScriptedSearch([]), "q", SMALL, budget=Budget(max_cost_usd=0.001))
    assert none.trace == PlanTrace((), Stop.BUDGET)
    assert none.results == []


def test_unreported_cost_falls_back_to_the_list_price_of_returned_results() -> None:
    assert CallTrace(25, 12, 1.0, None, 1, None).cost_usd == pytest.approx(0.009)
    assert CallTrace(25, 12, 1.0, 0.03, 1, None).cost_usd == 0.03


def test_passes_exclusions_and_null_policy_to_the_selection() -> None:
    results = [company("a.test", 10), company("u.test", None), company("b.test", 20)]

    strict = filtered_search(
        ScriptedSearch([reply(results)]),
        "q",
        SMALL,
        k=2,
        planner=FixedPlanner(10),
        exclude_entities=["id-a.test"],
    )
    assert urls(strict) == ["https://b.test"]
    assert strict.selection.excluded == 1
    assert strict.selection.unevaluable == 1

    lenient = filtered_search(
        ScriptedSearch([reply(results)]),
        "q",
        SMALL,
        k=2,
        planner=FixedPlanner(10),
        exclude_entities=["id-a.test"],
        null_policy=NullPolicy.LENIENT,
    )
    assert urls(lenient) == ["https://u.test", "https://b.test"]
    assert lenient.trace.stopped_by is Stop.FILLED


def test_planner_sees_k_the_calls_so_far_and_the_accepted_count() -> None:
    planner = RecordingPlanner()
    first = passing("a", 7) + failing("f", 3)
    second = passing("a", 7) + failing("f", 18)
    search = ScriptedSearch([reply(first), reply(second)])

    response = filtered_search(search, "q", SMALL, k=10, planner=planner)

    assert planner.asked == [
        (10, [], 0),
        (10, [CallRecord(10, 10)], 7),
        (10, [CallRecord(10, 10), CallRecord(25, 25)], 7),
    ]
    assert response.trace.stopped_by is Stop.PLANNER


def test_rejects_bad_k_before_calling_exa() -> None:
    search = ScriptedSearch([])

    with pytest.raises(ValueError, match="k must be"):
        filtered_search(search, "q", SMALL, k=0)
    assert search.seen == []
