"""The benchmark command: grade every gradable benchmark query. Dry run unless `yes`."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx

from exa_bench.constraints.analysis import Analysis
from exa_bench.constraints.benchmark import (
    CATEGORY,
    NUM_RESULTS,
    SEARCH_TYPE,
    BenchmarkReport,
    build_report,
    select_gradable,
)
from exa_bench.constraints.probe import estimated_cost_usd
from exa_bench.core.benchmark_data import load_company_queries
from exa_bench.core.response_cache import cached_search, is_cached

DEFAULT_SEED = 20260924
CACHE_DIR = Path("cache")
RESULTS_PATH = Path("results/company_constraint_benchmark.json")
PROGRESS_EVERY = 25


def run_grade(*, seed: int, yes: bool) -> int:
    with httpx.Client(timeout=60.0) as client:
        queries = select_gradable(load_company_queries(CACHE_DIR / "benchmarks", client))
        cached = sum(
            is_cached(
                CACHE_DIR / "exa",
                q.text,
                category=CATEGORY,
                num_results=NUM_RESULTS,
                search_type=SEARCH_TYPE,
            )
            for q in queries
        )
        uncached = len(queries) - cached
        print(f"{len(queries)} gradable queries; {cached} already cached, {uncached} to fetch")
        print(f"Estimated cost at list price: ${estimated_cost_usd(uncached):.2f}")
        if not yes:
            print("Dry run. Re-run with --yes to send the uncached searches to Exa.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2

        searches = []
        spent = 0.0
        for number, query in enumerate(queries, start=1):
            item = cached_search(
                CACHE_DIR / "exa",
                client,
                api_key,
                query.text,
                category=CATEGORY,
                num_results=NUM_RESULTS,
                search_type=SEARCH_TYPE,
            )
            searches.append(item)
            if not item.from_cache:
                spent += item.call.cost_dollars or 0.0
            if number % PROGRESS_EVERY == 0 or number == len(queries):
                print(f"  {number}/{len(queries)} done, ${spent:.2f} spent this run")

    report = build_report(queries, searches, seed=seed)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"Wrote {RESULTS_PATH}")
    return 0


def _print_summary(report: BenchmarkReport) -> None:
    meta = report.metadata
    print(f"\n{meta.queries} queries, {report.strict.results} results, ", end="")
    print(f"Exa-reported cost ${meta.cost_usd_reported_total:.2f} over all calls")
    for label, analysis in (("strict", report.strict), ("tolerant", report.tolerant)):
        print(f"\n[{label} headcount] " + _rate_line(analysis))
    print("\nStrict, by query bucket:")
    for bucket, rate in report.strict.by_bucket.items():
        print(f"  {bucket:16} {_pct(rate.value):>6}  ({rate.numerator}/{rate.denominator})")
    print("\nStrict, by constraint (share of checks that fail):")
    for key, stats in report.strict.by_constraint.items():
        if stats.passed + stats.failed == 0:
            continue
        print(
            f"  {key:14} {_pct(stats.fail_rate.value):>6}  "
            f"fail={stats.failed} pass={stats.passed} unknown={stats.unknown}"
        )


def _rate_line(analysis: Analysis) -> str:
    rate, bounds = analysis.violation_rate, analysis.violation_bounds
    return (
        f"violation rate {_pct(rate.value)} (95% CI {_pct(rate.ci_low)}-{_pct(rate.ci_high)}); "
        f"{rate.denominator} evaluable; bounds {_pct(bounds.lower)}-{_pct(bounds.upper)}; "
        f"verdicts {analysis.verdicts}"
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"
