"""Compare Exa Agent with the fetch policies on the same queries, from the stored records."""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exa_bench.agent_grade import GradedAgentOutcome, GradedMetadata, GradedRecord
from exa_bench.policy_analysis import Spread, spread
from exa_bench.stats import DEFAULT_RESAMPLES, Rate, rate


def load_graded_record(path: Path) -> GradedRecord:
    """Read a graded record written by the agent-grade command."""
    data = json.loads(path.read_text(encoding="utf-8"))
    outcomes = tuple(_outcome(row) for row in data["outcomes"])
    return GradedRecord(GradedMetadata(**data["metadata"]), outcomes)


def _outcome(row: dict[str, Any]) -> GradedAgentOutcome:
    return GradedAgentOutcome(**row)


@dataclass(frozen=True)
class AgentSummary:
    effort: str
    queries: int
    status: dict[str, int]
    filled_strict: int
    filled_lenient: int
    fill_rate_strict: Rate  # queries with K satisfying companies; CI by query
    fill_rate_lenient: Rate  # counting unknown and not-found companies as the lenient policy does
    mean_companies: float
    mean_accepted_strict: float
    lookups: int  # distinct domains looked up
    found_share: Rate  # domains with a company entity, of those looked up
    violating_share: Rate  # of the companies found
    unevaluable_share: Rate
    cost_usd: Spread  # the Agent's own cost per run, as Exa reported it
    latency_ms: Spread  # server-side time per run
    lookup_cost_usd: float  # our grading cost in total; not the user's


def summarize_agent(
    outcomes: Sequence[GradedAgentOutcome], *, seed: int, resamples: int = DEFAULT_RESAMPLES
) -> AgentSummary:
    if not outcomes:
        raise ValueError("no outcomes to summarize")
    efforts = {outcome.effort for outcome in outcomes}
    if len(efforts) != 1:
        raise ValueError(f"outcomes mix efforts: {sorted(efforts)}")
    (effort,) = efforts
    queries = len(outcomes)
    costs = [o.cost_usd for o in outcomes if o.cost_usd is not None]
    times = [o.server_ms for o in outcomes if o.server_ms is not None]
    return AgentSummary(
        effort=effort,
        queries=queries,
        status=dict(sorted(Counter(o.status for o in outcomes).items())),
        filled_strict=sum(o.filled_strict for o in outcomes),
        filled_lenient=sum(o.filled_lenient for o in outcomes),
        fill_rate_strict=_rate([(int(o.filled_strict), 1) for o in outcomes], seed, resamples),
        fill_rate_lenient=_rate([(int(o.filled_lenient), 1) for o in outcomes], seed, resamples),
        mean_companies=sum(o.companies for o in outcomes) / queries,
        mean_accepted_strict=sum(o.accepted_strict for o in outcomes) / queries,
        lookups=sum(o.lookups for o in outcomes),
        found_share=_rate([(o.found, o.lookups) for o in outcomes], seed, resamples),
        violating_share=_rate([(o.violating, o.found) for o in outcomes], seed, resamples),
        unevaluable_share=_rate([(o.unevaluable, o.found) for o in outcomes], seed, resamples),
        cost_usd=spread(costs) if costs else Spread(0.0, 0.0, 0.0),
        latency_ms=spread(times) if times else Spread(0.0, 0.0, 0.0),
        lookup_cost_usd=sum(o.lookup_cost_usd for o in outcomes),
    )


def _rate(clusters: list[tuple[int, int]], seed: int, resamples: int) -> Rate:
    return rate(clusters, seed=seed, resamples=resamples)
