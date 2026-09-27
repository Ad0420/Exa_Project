"""The agent-grade command: judge the stored Agent runs by entity lookups. Dry run unless `yes`.

It never creates an Agent run. Each distinct domain a run returned costs one cached /search
(list price $0.007) the first time; the dry run counts exactly those.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx

from exa_bench.agent.cli import AGENT_CACHE_DIR, CACHE_DIR, RESULTS_DIR
from exa_bench.agent.grade import GradedRecord, build_graded_record, grade_run, lookup_plan
from exa_bench.agent.runs import (
    AGENT_SUBSET_SEED,
    agent_request,
    read_cached_run,
    select_agent_subset,
)
from exa_bench.constraints.benchmark import CATEGORY, SEARCH_TYPE, load_shallow, select_gradable
from exa_bench.core.benchmark_data import load_company_queries
from exa_bench.core.response_cache import CachedSearch, cached_search, is_cached
from exa_bench.policies.run import select_workload
from exa_filters.planner import list_price

LOOKUP_RESULTS = 1


def run_agent_grade(*, effort: str, tolerance: float, yes: bool) -> int:
    with httpx.Client(timeout=60.0) as client:
        queries = select_gradable(load_company_queries(CACHE_DIR / "benchmarks", client))
        try:
            shallow = load_shallow(CACHE_DIR, queries)
        except FileNotFoundError as missing:
            print(f"{missing}; run `grade --yes` first.")
            return 2
        workload = select_workload(queries, [shallow[q.query_id].body for q in queries])
        subset = select_agent_subset(workload)
        runs = {}
        for query in subset:
            run = read_cached_run(AGENT_CACHE_DIR, agent_request(query.text, effort=effort))
            if run is None:
                print(f"No stored {effort} run for {query.query_id}; run `agent --yes` first.")
                return 2
            runs[query.query_id] = run
        planned = [item for run in runs.values() for item in lookup_plan(run)[0]]
        uncached = [item for item in planned if not _is_cached(item.query, item.domain)]
        print(
            f"{len(subset)} stored {effort} runs; {len(planned)} distinct companies to look up, "
            f"{len(uncached)} not yet cached"
        )
        print(f"Estimated cost at list price: ${len(uncached) * list_price(LOOKUP_RESULTS):.2f}")
        if not yes:
            print("Dry run. Re-run with --yes to send the lookups to Exa.")
            return 0

        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            print("EXA_API_KEY is not set; refusing to run.")
            return 2

        def lookup(query_text: str, domain: str) -> CachedSearch:
            return cached_search(
                CACHE_DIR / "exa",
                client,
                api_key,
                query_text,
                category=CATEGORY,
                num_results=LOOKUP_RESULTS,
                search_type=SEARCH_TYPE,
                include_domains=[domain],
            )

        outcomes = [
            grade_run(
                query,
                query.query_id in workload.clean_ids,
                effort,
                runs[query.query_id],
                lookup,
                employee_tolerance=tolerance,
            )
            for query in subset
        ]

    record = build_graded_record(
        effort, outcomes, employee_tolerance=tolerance, seed=AGENT_SUBSET_SEED
    )
    suffix = "" if tolerance == 0 else f".tol{tolerance:g}"
    path = RESULTS_DIR / f"graded.{effort}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    _print_summary(record, path)
    return 0


def _is_cached(query_text: str, domain: str) -> bool:
    return is_cached(
        CACHE_DIR / "exa",
        query_text,
        category=CATEGORY,
        num_results=LOOKUP_RESULTS,
        search_type=SEARCH_TYPE,
        include_domains=[domain],
    )


def _print_summary(record: GradedRecord, path: Path) -> None:
    meta, outcomes = record.metadata, record.outcomes
    companies = sum(o.companies for o in outcomes)
    found = sum(o.found for o in outcomes)
    violating = sum(o.violating for o in outcomes)
    print(
        f"\nAgent {meta.effort}, {meta.queries} queries: {companies} companies returned, "
        f"{found} with an entity, {violating} violating "
        f"({violating / found if found else 0:.1%} of those found); filled to {meta.k}: "
        f"{sum(o.filled_strict for o in outcomes)} strict, "
        f"{sum(o.filled_lenient for o in outcomes)} lenient"
    )
    print(
        f"Lookups sent this time: {meta.lookups_sent} of {meta.lookups}, Exa-reported "
        f"${meta.lookup_spent_usd:.3f}; wrote {path}"
    )
