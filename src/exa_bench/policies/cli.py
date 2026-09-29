"""The policy command: run one fetch policy over the workload. Dry run unless `yes`.

The dry run replays every query against the cache and lists the calls it would send, so
the cost statement is exact for one-call policies and a lower bound for adaptive.
"""

import json
import os
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import httpx

from exa_bench.constraints.benchmark import CATEGORY
from exa_bench.core.response_cache import CachedSearch, cached_search, read_cached
from exa_bench.policies.run import (
    PlannedCall,
    Run,
    RunRecord,
    build_record,
    plan_calls,
    record_name,
    run_policy,
    summarize_plan,
)
from exa_bench.policies.workload import Workload, load_workload
from exa_filters.api import ApiCall
from exa_filters.results import NullPolicy

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results/policy")
PROGRESS_EVERY = 25


def run_policy_command(*, policy: str, null_policy: str, tolerance: float, yes: bool) -> int:
    run = Run(policy, NullPolicy(null_policy), tolerance)
    with httpx.Client(timeout=90.0) as client:
        try:
            workload = load_workload(CACHE_DIR, client)
        except FileNotFoundError as missing:
            print(f"{missing}; run `grade --yes` first.")
            return 2
        planned = plan_calls(_read, workload, run)
        _print_plan(run, workload, planned)
        if not yes:
            print("Dry run. Re-run with --yes to send these searches to Exa.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2
        outcomes = run_policy(_Fetcher(client, api_key), workload, run)

    record = build_record(run, workload, planned, outcomes)
    path = RESULTS_DIR / record_name(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    _print_summary(record, path)
    return 0


def _read(query: str, num_results: int, search_type: str) -> ApiCall | None:
    return read_cached(
        CACHE_DIR / "exa",
        query,
        category=CATEGORY,
        num_results=num_results,
        search_type=search_type,
    )


class _Fetcher:
    """Serve calls from the cache or Exa, reporting progress on the ones that are sent."""

    def __init__(self, client: httpx.Client, api_key: str) -> None:
        self._client = client
        self._api_key = api_key
        self.sent = 0
        self.spent = 0.0

    def __call__(self, query: str, num_results: int, search_type: str) -> CachedSearch:
        item = cached_search(
            CACHE_DIR / "exa",
            self._client,
            self._api_key,
            query,
            category=CATEGORY,
            num_results=num_results,
            search_type=search_type,
        )
        if not item.from_cache:
            self.sent += 1
            self.spent += item.call.cost_dollars or 0.0
            if self.sent % PROGRESS_EVERY == 0:
                print(f"  {self.sent} calls sent, ${self.spent:.2f} spent this run")
        return item


def _print_plan(run: Run, workload: Workload, planned: list[PlannedCall]) -> None:
    plan = summarize_plan(planned)
    served = len(workload.queries) - len({call.query_id for call in planned})
    sizes = ", ".join(f"{count} at numResults={n}" for n, count in plan.by_num_results.items())
    print(
        f"Workload: {len(workload.queries)} queries ({len(workload.clean_ids)} from the clean "
        f"sample, seed {workload.seed}); policy {run.policy}, null policy "
        f"{run.null_policy.value}, headcount tolerance {run.employee_tolerance:g}"
    )
    print(
        f"{served} queries fully served from cache; {plan.calls} calls to send: {sizes or 'none'}"
    )
    print(
        f"Estimated cost at list price: ${plan.list_price_usd:.2f} (each query's first uncached "
        "call; a policy that escalates may need more once these are cached)"
    )


def _print_summary(record: RunRecord, path: Path) -> None:
    meta, outcomes = record.metadata, record.outcomes
    filled = sum(outcome.filled for outcome in outcomes)
    one_call = sum(len(outcome.requested) == 1 for outcome in outcomes)
    calls = statistics.mean(len(outcome.requested) for outcome in outcomes)
    cost = statistics.mean(outcome.cost_usd for outcome in outcomes)
    latency = statistics.median(outcome.latency_ms for outcome in outcomes)
    stops = dict(Counter(outcome.stopped_by for outcome in outcomes))
    print(
        f"\n{meta.policy}/{meta.null_policy}: {filled}/{meta.queries} filled to {meta.k}; "
        f"{one_call} in one call; mean calls {calls:.2f}; mean cost ${cost:.4f}; "
        f"median latency {latency:.0f} ms; stopped by {stops}"
    )
    print(
        f"Calls sent this run: {meta.calls_sent}, Exa-reported ${meta.spent_usd:.3f}; wrote {path}"
    )
