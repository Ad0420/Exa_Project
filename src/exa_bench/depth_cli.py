"""The stability command: is Exa's top-10 a stable prefix of its top-100, and is it repeatable?

Dry run unless `yes`. `repeat` re-issues the shallow searches unchanged (kept apart by a cache
tag) to measure run-to-run variation, which separates nondeterminism from depth dependence.
"""

import json
import os
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from exa_bench.benchmark import CATEGORY, NUM_RESULTS, SEARCH_TYPE, select_gradable
from exa_bench.benchmark_data import COMMIT, BenchmarkQuery, load_company_queries
from exa_bench.depth import (
    DEPTH,
    STABILITY_COUNT,
    STABILITY_SEED,
    StabilitySummary,
    non_clean_queries,
    prefix_stability,
    result_urls,
    seeded_subset,
    summarize_stability,
)
from exa_bench.exa_api import ApiCall
from exa_bench.response_cache import cached_search, is_cached, read_cached

CACHE_DIR = Path("cache")
STABILITY_PATH = Path("results/depth_stability.json")
REPEAT_PATH = Path("results/depth_repeatability.json")
REPEAT_TAG = "repeat-1"
PRICE_BASE_USD = 0.007  # Exa list price per request, up to 10 results
PRICE_EXTRA_RESULT_USD = 0.001  # each result beyond 10


def estimated_cost_usd(calls: int, depth: int) -> float:
    return calls * (PRICE_BASE_USD + max(0, depth - NUM_RESULTS) * PRICE_EXTRA_RESULT_USD)


@dataclass(frozen=True)
class RunMetadata:
    benchmark_commit: str
    run_date: str  # UTC, YYYY-MM-DD
    seed: int
    comparison: str  # what the shallow top-10 was compared with
    depth: int
    queries: int
    calls_from_cache: int
    cost_usd_reported_this_run: float
    latency_ms_median: float | None  # of the compared fetches, cached or not
    latency_ms_max: float | None
    results_returned_min: int | None
    results_returned_max: int | None


def run_stability(*, yes: bool, repeat: bool) -> int:
    """Compare each shallow top-10 with a deep fetch (default) or with a repeated shallow one."""
    depth = NUM_RESULTS if repeat else DEPTH
    tag = REPEAT_TAG if repeat else None
    with httpx.Client(timeout=90.0) as client:
        loaded = _load_targets(client)
        if loaded is None:
            return 2
        shallow, targets = loaded
        cached = sum(_is_cached(q, depth, tag) for q in targets)
        what = f"repeated at depth {depth}" if repeat else f"fetched at depth {depth}"
        print(
            f"Stability subset of {len(targets)} (seed {STABILITY_SEED}); {cached} already {what}"
        )
        uncached = len(targets) - cached
        print(f"Estimated cost at list price: ${estimated_cost_usd(uncached, depth):.2f}")
        if not yes:
            print("Dry run. Re-run with --yes to send the searches to Exa.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2
        compared: dict[str, ApiCall] = {}
        spent = 0.0
        for query in targets:
            item = cached_search(
                CACHE_DIR / "exa",
                client,
                api_key,
                query.text,
                category=CATEGORY,
                num_results=depth,
                search_type=SEARCH_TYPE,
                cache_tag=tag,
            )
            compared[query.query_id] = item.call
            if not item.from_cache:
                spent += item.call.cost_dollars or 0.0
            source = "cache" if item.from_cache else f"{item.call.latency_ms:.0f} ms"
            print(f"  {query.query_id}: {len(result_urls(item.call.body))} results ({source})")

    summary = summarize_stability(
        [
            prefix_stability(
                q.query_id,
                result_urls(shallow[q.query_id].body),
                result_urls(compared[q.query_id].body),
            )
            for q in targets
        ]
    )
    metadata = _metadata(targets, compared, cached, spent, depth, what)
    path = REPEAT_PATH if repeat else STABILITY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": asdict(metadata), "stability": asdict(summary)}, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary, what, spent, path)
    return 0


def _load_targets(
    client: httpx.Client,
) -> tuple[dict[str, ApiCall], list[BenchmarkQuery]] | None:
    queries = select_gradable(load_company_queries(CACHE_DIR / "benchmarks", client))
    shallow: dict[str, ApiCall] = {}
    for query in queries:
        call = read_cached(
            CACHE_DIR / "exa",
            query.text,
            category=CATEGORY,
            num_results=NUM_RESULTS,
            search_type=SEARCH_TYPE,
        )
        if call is None:
            print(f"No cached search for {query.query_id}; run `grade --yes` first.")
            return None
        shallow[query.query_id] = call
    non_clean = non_clean_queries(queries, [shallow[q.query_id].body for q in queries])
    print(f"{len(non_clean)} non-clean queries")
    return shallow, seeded_subset(non_clean, count=STABILITY_COUNT, seed=STABILITY_SEED)


def _is_cached(query: BenchmarkQuery, depth: int, tag: str | None) -> bool:
    return is_cached(
        CACHE_DIR / "exa",
        query.text,
        category=CATEGORY,
        num_results=depth,
        search_type=SEARCH_TYPE,
        cache_tag=tag,
    )


def _metadata(
    targets: Sequence[BenchmarkQuery],
    compared: dict[str, ApiCall],
    cached: int,
    spent: float,
    depth: int,
    comparison: str,
) -> RunMetadata:
    latencies = [compared[q.query_id].latency_ms for q in targets]
    counts = [len(result_urls(compared[q.query_id].body)) for q in targets]
    return RunMetadata(
        benchmark_commit=COMMIT,
        run_date=datetime.now(UTC).date().isoformat(),
        seed=STABILITY_SEED,
        comparison=comparison,
        depth=depth,
        queries=len(targets),
        calls_from_cache=cached,
        cost_usd_reported_this_run=spent,
        latency_ms_median=statistics.median(latencies) if latencies else None,
        latency_ms_max=max(latencies, default=None),
        results_returned_min=min(counts, default=None),
        results_returned_max=max(counts, default=None),
    )


def _print_summary(summary: StabilitySummary, what: str, spent: float, path: Path) -> None:
    print(
        f"\nShallow top-{NUM_RESULTS} vs the same queries {what}, {summary.queries} queries: "
        f"mean URL overlap {_pct(summary.mean_overlap)}, identical order "
        f"{_pct(summary.share_same_order)}, "
        f"overlap >= 90% {_pct(summary.share_overlap_at_least_90)}"
    )
    print(
        f"H-S thresholds (mean overlap >= {summary.gate_min_mean_overlap:.0%} and identical "
        f"order >= {summary.gate_min_same_order_share:.0%}): "
        f"{'PASS' if summary.gate_passed else 'FAIL'}"
    )
    print(f"Exa-reported cost this run: ${spent:.3f}; wrote {path}")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"
