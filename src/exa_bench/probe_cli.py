"""The Day-0 coverage probe command. Dry run unless `yes`, which spends money."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx

from exa_bench.benchmark_data import load_company_queries
from exa_bench.coverage import measure_coverage, search_results
from exa_bench.probe import decide_gate, estimated_cost_usd, select_queries, summarize
from exa_bench.response_cache import cached_search

DEFAULT_COUNT = 20
DEFAULT_SEED = 20260924
CACHE_DIR = Path("cache")
RESULTS_PATH = Path("results/probe_company_coverage.json")


def run_probe(*, count: int, seed: int, yes: bool) -> int:
    with httpx.Client(timeout=60.0) as client:
        queries = load_company_queries(CACHE_DIR / "benchmarks", client)
        selected = select_queries(queries, count=count, seed=seed)
        print(f"Selected {len(selected)} queries (seed {seed}):")
        for query in selected:
            print(f"  {query.query_id}: {query.text}")
        print(f"Estimated cost at list price: ${estimated_cost_usd(len(selected)):.3f}")
        if not yes:
            print("Dry run. Re-run with --yes to send these searches to Exa.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2

        calls = []
        for query in selected:
            cached = cached_search(CACHE_DIR / "exa", client, api_key, query.text)
            calls.append(cached.call)
            source = "cache" if cached.from_cache else f"{cached.call.latency_ms:.0f} ms"
            print(f"  {query.query_id}: {len(search_results(cached.call.body))} results ({source})")

    report = measure_coverage(call.body for call in calls)
    summary = summarize(selected, calls, report, decide_gate(report), seed=seed)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")

    print(f"\nResults: {summary.results} ({summary.results_with_company} with a company entity)")
    for name, rate in summary.gate_fill_rates.items():
        print(f"  {name:>13}: {rate:6.1%}")
    verdict = "PASS" if summary.gate_passed else "FAIL"
    print(f"Gate (each >= {summary.gate_min_fill_rate:.0%}): {verdict}")
    print(f"Cost reported by Exa: ${summary.cost_usd_reported_total:.3f}")
    print(f"Wrote {RESULTS_PATH}")
    return 0
