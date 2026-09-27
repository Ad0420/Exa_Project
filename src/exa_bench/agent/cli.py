"""The agent command: run Exa Agent on a seeded subset of the workload. Dry run unless `yes`.

Runs are sequential and cached under cache/exa_agent, so a repeated command never creates a
run twice. The flat effort price is known in advance; the per-search charge is not.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx

from exa_bench.agent.runs import (
    AGENT_SUBSET_SEED,
    AgentRunsRecord,
    agent_outcome,
    agent_request,
    build_runs_record,
    cached_run,
    run_key,
    select_agent_subset,
)
from exa_bench.constraints.benchmark import load_shallow, select_gradable
from exa_bench.core.benchmark_data import load_company_queries
from exa_bench.core.json_cache import is_cached
from exa_bench.policies.run import select_workload
from exa_filters.pricing import AGENT_EFFORT_USD, AGENT_SEARCH_USD

CACHE_DIR = Path("cache")
AGENT_CACHE_DIR = CACHE_DIR / "exa_agent"
RESULTS_DIR = Path("results/agent")


def run_agent(*, effort: str, yes: bool) -> int:
    with httpx.Client(timeout=60.0) as client:
        queries = select_gradable(load_company_queries(CACHE_DIR / "benchmarks", client))
        try:
            shallow = load_shallow(CACHE_DIR, queries)
        except FileNotFoundError as missing:
            print(f"{missing}; run `grade --yes` first.")
            return 2
        workload = select_workload(queries, [shallow[q.query_id].body for q in queries])
        subset = select_agent_subset(workload)
        requests = {query.query_id: agent_request(query.text, effort=effort) for query in subset}
        cached = sum(is_cached(AGENT_CACHE_DIR, run_key(body)) for body in requests.values())
        to_create = len(subset) - cached
        print(
            f"Agent subset: {len(subset)} of the {len(workload.queries)} workload queries "
            f"(seed {AGENT_SUBSET_SEED}); effort {effort}; {cached} already run, {to_create} to run"
        )
        print(
            f"Estimated cost at list price: ${to_create * AGENT_EFFORT_USD[effort]:.2f} flat, plus "
            f"${AGENT_SEARCH_USD:.3f} per search the agent chooses to make (unknown in advance)"
        )
        if not yes:
            print("Dry run. Re-run with --yes to create the runs.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2
        outcomes = []
        spent = 0.0
        for number, query in enumerate(subset, start=1):
            result = cached_run(AGENT_CACHE_DIR, client, api_key, requests[query.query_id])
            outcome = agent_outcome(query, query.query_id in workload.clean_ids, effort, result)
            outcomes.append(outcome)
            if not outcome.from_cache:
                spent += outcome.cost_usd or 0.0
            print(
                f"  {number}/{len(subset)} {query.query_id}: {outcome.status}, "
                f"{outcome.companies} companies, ${outcome.cost_usd or 0.0:.3f}, "
                f"{(outcome.server_ms or 0.0) / 1000:.0f} s "
                f"({'cache' if outcome.from_cache else 'run'}); ${spent:.2f} spent this run",
                flush=True,
            )

    record = build_runs_record(effort, workload, outcomes)
    path = RESULTS_DIR / f"runs.{effort}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    _print_summary(record, path)
    return 0


def _print_summary(record: AgentRunsRecord, path: Path) -> None:
    meta, outcomes = record.metadata, record.outcomes
    completed = [o for o in outcomes if o.status == "completed"]
    costs = [o.cost_usd for o in outcomes if o.cost_usd is not None]
    times = sorted(o.server_ms for o in outcomes if o.server_ms is not None)
    print(
        f"\nAgent {meta.effort}: {len(completed)}/{meta.queries} completed; "
        f"mean companies {sum(o.companies for o in outcomes) / max(1, len(outcomes)):.1f}; "
        f"mean cost ${sum(costs) / max(1, len(costs)):.4f}; "
        f"median time {times[len(times) // 2] / 1000 if times else 0:.0f} s"
    )
    print(
        f"Runs created this time: {meta.runs_created}, Exa-reported ${meta.spent_usd:.3f}; "
        f"wrote {path}"
    )
