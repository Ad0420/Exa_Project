"""Depth dataset: is Exa's ranking a stable prefix, and how deep do k satisfying results lie?

Every depth policy in Part 2 is simulated by taking prefixes of one deep response per
query. That is only valid if the top of a deep response matches a shallow one, so the
stability of that prefix is measured first and gates the simulation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from exa_bench.constraints.grader import ResultVerdict
from exa_filters.response import search_results

DEPTH = 100
STABILITY_COUNT = 20
STABILITY_SEED = 20260925
# H-S gate, fixed before any fetch: mean top-10 URL overlap and share of identical orderings.
GATE_MIN_MEAN_OVERLAP = 0.90
GATE_MIN_SAME_ORDER_SHARE = 0.80


def result_urls(body: Mapping[str, object]) -> list[str]:
    """The URLs of a /search response's results, in rank order, skipping malformed entries."""
    urls: list[str] = []
    for result in search_results(body):
        if isinstance(result, Mapping) and isinstance(result.get("url"), str):
            urls.append(str(result["url"]))
    return urls


@dataclass(frozen=True)
class PrefixStability:
    query_id: str
    shallow: int  # results in the shallow response
    overlap: float  # share of shallow URLs present in the deep prefix of the same length
    same_order: bool  # the deep prefix is exactly the shallow list
    common_prefix: int  # how many leading positions agree before the first difference


def prefix_stability(
    query_id: str, shallow_urls: Sequence[str], deep_urls: Sequence[str]
) -> PrefixStability:
    prefix = list(deep_urls[: len(shallow_urls)])
    if not shallow_urls:
        return PrefixStability(query_id, 0, 0.0, False, 0)
    overlap = len(set(shallow_urls) & set(prefix)) / len(shallow_urls)
    common = 0
    for mine, theirs in zip(shallow_urls, prefix, strict=False):
        if mine != theirs:
            break
        common += 1
    return PrefixStability(
        query_id, len(shallow_urls), overlap, list(shallow_urls) == prefix, common
    )


@dataclass(frozen=True)
class StabilitySummary:
    queries: int
    mean_overlap: float | None
    share_same_order: float | None
    share_overlap_at_least_90: float | None
    gate_min_mean_overlap: float
    gate_min_same_order_share: float
    gate_passed: bool
    per_query: tuple[PrefixStability, ...]


def summarize_stability(items: Sequence[PrefixStability]) -> StabilitySummary:
    if not items:
        return StabilitySummary(
            0, None, None, None, GATE_MIN_MEAN_OVERLAP, GATE_MIN_SAME_ORDER_SHARE, False, ()
        )
    mean_overlap = sum(i.overlap for i in items) / len(items)
    same_order = sum(i.same_order for i in items) / len(items)
    at_least_90 = sum(i.overlap >= 0.9 for i in items) / len(items)
    return StabilitySummary(
        queries=len(items),
        mean_overlap=mean_overlap,
        share_same_order=same_order,
        share_overlap_at_least_90=at_least_90,
        gate_min_mean_overlap=GATE_MIN_MEAN_OVERLAP,
        gate_min_same_order_share=GATE_MIN_SAME_ORDER_SHARE,
        gate_passed=mean_overlap >= GATE_MIN_MEAN_OVERLAP
        and same_order >= GATE_MIN_SAME_ORDER_SHARE,
        per_query=tuple(items),
    )


def depth_needed(verdicts: Sequence[ResultVerdict], k: int, *, lenient: bool) -> int | None:
    """Smallest prefix length holding k acceptable results, or None if the list never does.

    Strict accepts only satisfying results; lenient also accepts unevaluable ones (kept,
    flagged), matching the two null policies of the filters feature.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    accepted = {ResultVerdict.SATISFIES}
    if lenient:
        accepted.add(ResultVerdict.UNEVALUABLE)
    found = 0
    for position, verdict in enumerate(verdicts, start=1):
        found += verdict in accepted
        if found >= k:
            return position
    return None
