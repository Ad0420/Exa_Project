"""The stability command: is Exa's top-10 a stable prefix of its top-100? Dry run unless `yes`."""

import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from exa_bench.benchmark import CATEGORY, NUM_RESULTS, SEARCH_TYPE, select_gradable
from exa_bench.benchmark_data import COMMIT, BenchmarkQuery, load_company_queries
from exa_bench.depth import (
    DEPTH,
    STABILITY_SEED,
    non_clean_queries,
    prefix_stability,
    result_urls,
    stability_subset,
    summarize_stability,
)
from exa_bench.exa_api import ApiCall
from exa_bench.response_cache import cached_search, is_cached, read_cached

CACHE_DIR = Path("cache")
RESULTS_PATH = Path("results/depth_stability.json")
PRICE_BASE_USD = 0.007  # Exa list price per request, up to 10 results
PRICE_EXTRA_RESULT_USD = 0.001  # each result beyond 10


def estimated_cost_usd(calls: int, depth: int) -> float:
    return calls * (PRICE_BASE_USD + max(0, depth - NUM_RESULTS) * PRICE_EXTRA_RESULT_USD)


@dataclass(frozen=True)
class RunMetadata:
    benchmark_commit: str
    run_date: str  # UTC, YYYY-MM-DD
    seed: int
    depth: int
    queries: int
    calls_from_cache: int
    cost_usd_reported_this_run: float
    latency_ms_median: float | None  # of the original deep fetches, cached or not
    latency_ms_max: float | None
    results_returned_min: int | None
    results_returned_max: int | None


def run_stability(*, yes: bool) -> int:
    with httpx.Client(timeout=90.0) as client:
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
                return 2
            shallow[query.query_id] = call
        non_clean = non_clean_queries(queries, [shallow[q.query_id].body for q in queries])
        targets = stability_subset(non_clean)
        cached = sum(_is_deep_cached(q) for q in targets)
        print(
            f"{len(non_clean)} non-clean queries; stability subset of {len(targets)} "
            f"(seed {STABILITY_SEED}); {cached} already fetched at depth {DEPTH}"
        )
        print(
            f"Estimated cost at list price: ${estimated_cost_usd(len(targets) - cached, DEPTH):.2f}"
        )
        if not yes:
            print("Dry run. Re-run with --yes to fetch the deep responses from Exa.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2
        deep: dict[str, ApiCall] = {}
        spent = 0.0
        for query in targets:
            item = cached_search(
                CACHE_DIR / "exa",
                client,
                api_key,
                query.text,
                category=CATEGORY,
                num_results=DEPTH,
                search_type=SEARCH_TYPE,
            )
            deep[query.query_id] = item.call
            if not item.from_cache:
                spent += item.call.cost_dollars or 0.0
            print(
                f"  {query.query_id}: {len(result_urls(item.call.body))} results "
                f"({'cache' if item.from_cache else f'{item.call.latency_ms:.0f} ms'})"
            )

    summary = summarize_stability(
        [
            prefix_stability(
                q.query_id,
                result_urls(shallow[q.query_id].body),
                result_urls(deep[q.query_id].body),
            )
            for q in targets
        ]
    )
    metadata = _metadata(targets, deep, cached, spent)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"metadata": asdict(metadata), "stability": asdict(summary)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\nTop-{NUM_RESULTS} vs the first {NUM_RESULTS} of {DEPTH} on {summary.queries} queries: "
        f"mean URL overlap {_pct(summary.mean_overlap)}, identical order "
        f"{_pct(summary.share_same_order)}, "
        f"overlap >= 90% {_pct(summary.share_overlap_at_least_90)}"
    )
    print(
        f"H-S gate (mean overlap >= {summary.gate_min_mean_overlap:.0%} and identical order "
        f">= {summary.gate_min_same_order_share:.0%}): {'PASS' if summary.gate_passed else 'FAIL'}"
    )
    print(f"Exa-reported cost this run: ${spent:.3f}; wrote {RESULTS_PATH}")
    return 0


def _is_deep_cached(query: BenchmarkQuery) -> bool:
    return is_cached(
        CACHE_DIR / "exa",
        query.text,
        category=CATEGORY,
        num_results=DEPTH,
        search_type=SEARCH_TYPE,
    )


def _metadata(
    targets: list[BenchmarkQuery], deep: dict[str, ApiCall], cached: int, spent: float
) -> RunMetadata:
    latencies = [deep[q.query_id].latency_ms for q in targets]
    counts = [len(result_urls(deep[q.query_id].body)) for q in targets]
    return RunMetadata(
        benchmark_commit=COMMIT,
        run_date=datetime.now(UTC).date().isoformat(),
        seed=STABILITY_SEED,
        depth=DEPTH,
        queries=len(targets),
        calls_from_cache=cached,
        cost_usd_reported_this_run=spent,
        latency_ms_median=statistics.median(latencies) if latencies else None,
        latency_ms_max=max(latencies, default=None),
        results_returned_min=min(counts, default=None),
        results_returned_max=max(counts, default=None),
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"
