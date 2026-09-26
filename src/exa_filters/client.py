"""Exa company search with hard filters, run as a policy over /search calls.

`filtered_search` is the loop: ask the planner how many results to request, check the budget,
call Exa, merge every call's results, select the k that satisfy the filters, and stop with a
stated reason. `FilteredExa` wraps it around an httpx client and an API key.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import httpx

from exa_filters import api
from exa_filters.api import ApiCall
from exa_filters.evaluate import Filter
from exa_filters.planner import AdaptivePlanner, Budget, CallRecord, Planner, exhausted, list_price
from exa_filters.response import search_results
from exa_filters.results import FilteredResult, NullPolicy, Selection, select
from exa_filters.spec import Filters

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_PLANNER: Planner = AdaptivePlanner()
DEFAULT_BUDGET = Budget()

type SearchFn = Callable[[str, int], ApiCall]  # (query, numResults) -> one /search response


class Stop(StrEnum):
    FILLED = "filled"  # k results accepted
    EXHAUSTED = "exhausted"  # Exa returned fewer than asked, so a deeper call cannot help
    PLANNER = "planner"  # the planner's schedule ended
    BUDGET = "budget"  # the budget refused the planner's next call


@dataclass(frozen=True)
class CallTrace(CallRecord):
    latency_ms: float
    cost_dollars: float | None  # Exa's costDollars.total, when reported
    attempts: int
    request_id: str | None

    @property
    def cost_usd(self) -> float:
        """Exa's reported cost, or the list price of the returned results when unreported."""
        return self.cost_dollars if self.cost_dollars is not None else list_price(self.returned)


@dataclass(frozen=True)
class PlanTrace:
    calls: tuple[CallTrace, ...]
    stopped_by: Stop

    @property
    def cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    @property
    def latency_ms(self) -> float:
        return sum(call.latency_ms for call in self.calls)


@dataclass(frozen=True)
class SearchResponse:
    selection: Selection  # counts over the results of every call, concatenated
    trace: PlanTrace

    @property
    def results(self) -> list[FilteredResult]:
        return self.selection.results


def filtered_search(
    search_fn: SearchFn,
    query: str,
    filters: Sequence[Filter],
    *,
    k: int = 10,
    null_policy: NullPolicy = NullPolicy.STRICT,
    exclude_entities: Iterable[str] = (),
    planner: Planner = DEFAULT_PLANNER,
    budget: Budget = DEFAULT_BUDGET,
) -> SearchResponse:
    """Call Exa as the planner directs until k results are accepted or a stop reason applies.

    A later call re-fetches most earlier results, since the calls overlap and Exa's ranking
    is not deterministic. The calls' results are concatenated in order and deduplicated, so
    a result's rank is its position in that stream and results seen only by an earlier call
    are kept. The budget judges each next call at list price and the calls made at Exa's
    reported cost.
    """
    exclusions = tuple(exclude_entities)
    stream: list[object] = []
    calls: list[CallTrace] = []
    while True:
        selection = select(
            {"results": stream}, filters, k=k, null_policy=null_policy, exclude_entities=exclusions
        )
        if selection.short_by == 0:
            stop = Stop.FILLED
            break
        if exhausted(calls):
            stop = Stop.EXHAUSTED
            break
        num_results = planner.next_num_results(k, calls, len(selection.results))
        if num_results is None:
            stop = Stop.PLANNER
            break
        cost_so_far = sum(call.cost_usd for call in calls)
        if not budget.allows(len(calls), cost_so_far, list_price(num_results)):
            stop = Stop.BUDGET
            break
        call = search_fn(query, num_results)
        results = search_results(call.body)
        stream.extend(results)
        calls.append(_call_trace(call, num_results, len(results)))
    return SearchResponse(selection, PlanTrace(tuple(calls), stop))


def _call_trace(call: ApiCall, requested: int, returned: int) -> CallTrace:
    return CallTrace(
        requested=requested,
        returned=returned,
        latency_ms=call.latency_ms,
        cost_dollars=call.cost_dollars,
        attempts=call.attempts,
        request_id=call.request_id,
    )


class FilteredExa:
    """Exa company search with hard filters, over httpx. Closes only a client it created."""

    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client(timeout=DEFAULT_TIMEOUT_S)

    def search(
        self,
        query: str,
        filters: Filters,
        *,
        k: int = 10,
        null_policy: NullPolicy = NullPolicy.STRICT,
        exclude_entities: Iterable[str] = (),
        planner: Planner = DEFAULT_PLANNER,
        budget: Budget = DEFAULT_BUDGET,
    ) -> SearchResponse:
        return filtered_search(
            self._search,
            query,
            filters.items,
            k=k,
            null_policy=null_policy,
            exclude_entities=exclude_entities,
            planner=planner,
            budget=budget,
        )

    def _search(self, query: str, num_results: int) -> ApiCall:
        return api.search(
            self.client, self._api_key, query, category="company", num_results=num_results
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
