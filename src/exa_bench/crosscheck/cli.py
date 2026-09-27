"""The cross-check command: judge sampled results against their pages. Dry run unless `yes`."""

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from openai import OpenAI

from exa_bench.constraints.benchmark import CATEGORY, NUM_RESULTS, SEARCH_TYPE, select_gradable
from exa_bench.core.benchmark_data import COMMIT, load_company_queries
from exa_bench.core.contents_cache import cached_contents, uncached_urls
from exa_bench.core.response_cache import read_cached
from exa_bench.core.stats import Rate
from exa_bench.crosscheck.agreement import CrossCheck, JudgedItem, cross_check
from exa_bench.crosscheck.llm_grader import EXA_GRADER_MODEL, EXA_GRADER_TEMPERATURE
from exa_bench.crosscheck.openai_judge import OpenAIExtractor, OpenAIJudge, OpenAIStructured
from exa_bench.crosscheck.run import RunTotals, judge_sample
from exa_bench.crosscheck.sample import build_items, select_sample
from exa_filters.api import ApiCall

DEFAULT_SEED = 20260925
CACHE_DIR = Path("cache")
RESULTS_PATH = Path("results/company_crosscheck.json")
WORKSHEET_PATH = CACHE_DIR / "crosscheck_worksheet.jsonl"  # per-item rows; local only
PRICE_PER_PAGE_USD = 0.001  # Exa /contents list price: $1 per 1k pages
PROGRESS_EVERY = 25


@dataclass(frozen=True)
class RunMetadata:
    benchmark_commit: str
    run_date: str  # UTC, YYYY-MM-DD
    sample_seed: int
    strata: dict[str, int]
    model: str
    temperature: float
    pages_distinct: int
    pages_fetched_this_run: int
    pages_from_cache: int
    exa_cost_usd_this_run: float
    extraction_calls_this_run: int
    grader_calls_this_run: int
    extraction_prompt_tokens: int
    extraction_completion_tokens: int
    grader_prompt_tokens: int
    grader_completion_tokens: int
    openai_cost_usd_estimate: float | None  # only when token prices were given


def run_crosscheck(*, seed: int, yes: bool, price_in: float | None, price_out: float | None) -> int:
    with httpx.Client(timeout=60.0) as client:
        queries = select_gradable(load_company_queries(CACHE_DIR / "benchmarks", client))
        calls: list[ApiCall] = []
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
            calls.append(call)
        sample = select_sample(build_items(queries, [call.body for call in calls]), seed=seed)
        urls = [item.url for item in sample]
        to_fetch = uncached_urls(CACHE_DIR / "exa_contents", urls)
        strata = dict(Counter(item.verdict.value for item in sample))
        print(
            f"Sample: {len(sample)} results from {len({i.query_id for i in sample})} queries "
            f"(seed {seed}): {strata}"
        )
        print(
            f"Pages: {len(set(urls))} distinct, {len(to_fetch)} to fetch "
            f"(about ${len(to_fetch) * PRICE_PER_PAGE_USD:.2f} at $1 per 1k)"
        )
        print(
            f"OpenAI: up to {len(sample)} extraction + {len(sample)} grader calls with "
            f"{EXA_GRADER_MODEL} at temperature {EXA_GRADER_TEMPERATURE}; cached calls are free"
        )
        if price_in is None or price_out is None:
            print(
                "Pass --price-in and --price-out (USD per 1M tokens) for an OpenAI dollar figure."
            )
        if not yes:
            print("Dry run. Re-run with --yes to fetch pages and call the judges.")
            return 0

        exa_key = os.environ.get("EXA_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not exa_key or not openai_key:
            print("EXA_API_KEY and OPENAI_API_KEY must both be set; refusing to run.")
            return 2
        fetch = cached_contents(CACHE_DIR / "exa_contents", client, exa_key, urls)
        exa_cost = sum(call.cost_dollars or 0.0 for call in fetch.calls)
        print(
            f"Pages: {fetch.from_cache} from cache, {len(fetch.calls)} batch calls, "
            f"Exa reported ${exa_cost:.3f}"
        )

    structured = OpenAIStructured(
        OpenAI(api_key=openai_key, max_retries=5), EXA_GRADER_MODEL, EXA_GRADER_TEMPERATURE
    )
    judged, totals = judge_sample(
        sample,
        fetch.pages,
        extractor=OpenAIExtractor(structured),
        judge=OpenAIJudge(structured),
        extraction_cache=CACHE_DIR / "llm_extraction",
        grader_cache=CACHE_DIR / "llm_grader",
        model=EXA_GRADER_MODEL,
        temperature=EXA_GRADER_TEMPERATURE,
        progress=_progress,
    )
    result = cross_check(judged, seed=seed)
    metadata = RunMetadata(
        benchmark_commit=COMMIT,
        run_date=datetime.now(UTC).date().isoformat(),
        sample_seed=seed,
        strata=strata,
        model=EXA_GRADER_MODEL,
        temperature=EXA_GRADER_TEMPERATURE,
        pages_distinct=len(set(urls)),
        pages_fetched_this_run=len(set(urls)) - fetch.from_cache,
        pages_from_cache=fetch.from_cache,
        exa_cost_usd_this_run=exa_cost,
        extraction_calls_this_run=totals.extraction_calls,
        grader_calls_this_run=totals.grader_calls,
        extraction_prompt_tokens=totals.extraction_prompt_tokens,
        extraction_completion_tokens=totals.extraction_completion_tokens,
        grader_prompt_tokens=totals.grader_prompt_tokens,
        grader_completion_tokens=totals.grader_completion_tokens,
        openai_cost_usd_estimate=_openai_cost(totals, price_in, price_out),
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"metadata": asdict(metadata), "crosscheck": asdict(result)}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_worksheet(judged)
    _print_summary(result, metadata)
    print(f"Wrote {RESULTS_PATH} (aggregates) and {WORKSHEET_PATH} (per-item, local only)")
    return 0


def _progress(number: int, total: int) -> None:
    if number % PROGRESS_EVERY == 0 or number == total:
        print(f"  {number}/{total} judged")


def _openai_cost(
    totals: RunTotals, price_in: float | None, price_out: float | None
) -> float | None:
    if price_in is None or price_out is None:
        return None
    prompt = totals.extraction_prompt_tokens + totals.grader_prompt_tokens
    completion = totals.extraction_completion_tokens + totals.grader_completion_tokens
    return (prompt * price_in + completion * price_out) / 1_000_000


def _write_worksheet(judged: Sequence[JudgedItem]) -> None:
    """Per-item rows for hand-labelling disagreements. Stays in the gitignored cache."""
    with WORKSHEET_PATH.open("w", encoding="utf-8") as handle:
        for j in judged:
            row = {
                "query_id": j.item.query_id,
                "query": j.item.query_text,
                "constraints": j.item.constraints,
                "url": j.item.url,
                "our_verdict": j.item.verdict.value,
                "our_outcomes": [(o.key, o.op, o.outcome.value) for o in j.item.outcomes],
                "typed": j.item.typed,
                "has_text": j.has_text,
                "page_facts": j.facts.model_dump() if j.facts is not None else None,
                "grader": (
                    {"score": j.grader.score, "explanation": j.grader.explanation}
                    if j.grader is not None
                    else None
                ),
            }
            handle.write(json.dumps(row) + "\n")


def _print_summary(result: CrossCheck, metadata: RunMetadata) -> None:
    print(
        f"\n{result.items} items: {result.with_text} with page text, {result.with_facts} with "
        f"extracted facts, {result.single_company_pages} judged to be single-company pages"
    )
    print("\nExa's typed field vs the page (accuracy = agrees / (agrees + disagrees)):")
    for name, stats in result.fields.items():
        print(
            f"  {name:14} {_rate(stats.accuracy):>24}  agrees={stats.agrees} "
            f"disagrees={stats.disagrees} page_silent={stats.not_stated} "
            f"no_typed={stats.no_typed_value}"
        )
    print("\nOur violations vs the page (confirmed rate = confirmed / (confirmed + contradicted)):")
    for name, checked in result.outcome_verification.items():
        print(
            f"  {name:14} {_rate(checked.confirmed_rate):>24}  confirmed={checked.confirmed} "
            f"contradicted={checked.contradicted} unverifiable={checked.unverifiable}"
        )
    v, s, u = result.violating_items, result.satisfying_items, result.unevaluable_items
    print(
        f"\nItems we called violates: confirmed={v.confirmed} contradicted={v.contradicted} "
        f"mixed={v.mixed} unverifiable={v.unverifiable}"
    )
    print(
        f"Items we called satisfies: undisputed={s.undisputed} disputed={s.disputed} "
        f"unverifiable={s.unverifiable}"
    )
    print(
        f"Items we called unevaluable: page resolves to satisfies={u.resolved_satisfies} "
        f"violates={u.resolved_violates} still_unknown={u.still_unknown}"
    )
    g = result.grader
    kappa = "n/a" if g.kappa is None else f"{g.kappa:.2f}"
    print(
        f"\nExa's grader vs ours on {g.compared} items: agreement {_rate(g.agreement)}, "
        f"kappa {kappa}; {g.cannot_verify} explanations read as 'cannot verify', "
        f"agreement excluding those {_rate(g.agreement_excluding_cannot_verify)}"
    )
    print(
        f"  ours satisfies: grader matches={g.ours_satisfies_grader_matches} "
        f"rejects={g.ours_satisfies_grader_rejects} | ours violates: grader "
        f"matches={g.ours_violates_grader_matches} rejects={g.ours_violates_grader_rejects}"
    )
    print(
        f"  our unevaluable items the grader judged: {g.unevaluable_compared}, "
        f"of which it matched {g.unevaluable_grader_matches}"
    )
    tokens = (
        metadata.extraction_prompt_tokens
        + metadata.extraction_completion_tokens
        + metadata.grader_prompt_tokens
        + metadata.grader_completion_tokens
    )
    cost = (
        "not priced"
        if metadata.openai_cost_usd_estimate is None
        else f"about ${metadata.openai_cost_usd_estimate:.2f}"
    )
    print(
        f"\nOpenAI: {metadata.extraction_calls_this_run + metadata.grader_calls_this_run} calls "
        f"this run, {tokens:,} tokens over all judged items ({cost}); "
        f"Exa this run ${metadata.exa_cost_usd_this_run:.3f}"
    )


def _rate(rate: Rate) -> str:
    if rate.value is None:
        return "n/a"
    if rate.ci_low is None or rate.ci_high is None:
        return f"{rate.value:.1%}"
    return f"{rate.value:.1%} ({rate.ci_low:.0%}-{rate.ci_high:.0%})"
