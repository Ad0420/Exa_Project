"""Run the Day-0 coverage probe. Dry run by default; `--yes` spends money."""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import httpx

from exa_bench.benchmark_data import load_company_queries
from exa_bench.coverage import measure_coverage
from exa_bench.probe import decide_gate, estimated_cost_usd, select_queries, summarize
from exa_bench.response_cache import cached_search

DEFAULT_COUNT = 20
DEFAULT_SEED = 20260924
CACHE_DIR = Path("cache")
RESULTS_PATH = Path("results/probe_company_coverage.json")


def main(argv: Sequence[str]) -> int:
    args = _parse(argv)
    with httpx.Client(timeout=60.0) as client:
        queries = load_company_queries(CACHE_DIR / "benchmarks", client)
        selected = select_queries(queries, count=args.count, seed=args.seed)
        print(f"Selected {len(selected)} queries (seed {args.seed}):")
        for query in selected:
            print(f"  {query.query_id}: {query.text}")
        print(f"Estimated cost at list price: ${estimated_cost_usd(len(selected)):.3f}")
        if not args.yes:
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
            print(f"  {query.query_id}: {len(_results(cached.call.body))} results ({source})")

    report = measure_coverage(call.body for call in calls)
    summary = summarize(selected, calls, report, decide_gate(report), seed=args.seed)
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


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m exa_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe", help="measure typed-field coverage on real searches")
    probe.add_argument("--count", type=int, default=DEFAULT_COUNT)
    probe.add_argument("--seed", type=int, default=DEFAULT_SEED)
    probe.add_argument("--yes", action="store_true", help="actually call Exa (spends credits)")
    return parser.parse_args(argv)


def _results(body: dict[str, object]) -> list[object]:
    results = body.get("results")
    return results if isinstance(results, list) else []
