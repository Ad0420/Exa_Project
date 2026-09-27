"""Compare Exa Agent with the fetch policies on the same queries, from the stored records."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from exa_bench.agent_grade import GradedAgentOutcome, GradedMetadata, GradedRecord
from exa_bench.policy import RunRecord
from exa_bench.policy_analysis import Check, RunSummary, Spread, check, spread, summarize_run
from exa_bench.stats import DEFAULT_RESAMPLES, Rate, rate

COST_RATIO_MIN = 10.0  # H-A: Agent costs at least this many times the cheapest one-call policy
LATENCY_RATIO_MIN = 10.0  # H-A: and takes at least this many times the baseline's p50 latency


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


@dataclass(frozen=True)
class AgentComparison:
    """Agent at one effort against one policy run on the same queries."""

    effort: str
    policy: str  # "policy.null_policy"
    fill_gain: int  # Agent's filled queries beyond the policy's, under the policy's null policy
    cost_ratio_mean: float | None  # Agent mean cost over the policy's
    latency_ratio_p50: float | None
    latency_ratio_p95: float | None


def compare_agent(agent: AgentSummary, policy: RunSummary) -> AgentComparison:
    filled = agent.filled_strict if policy.null_policy == "strict" else agent.filled_lenient
    return AgentComparison(
        effort=agent.effort,
        policy=f"{policy.policy}.{policy.null_policy}",
        fill_gain=filled - policy.filled,
        cost_ratio_mean=_ratio(agent.cost_usd.mean, policy.cost_usd.mean),
        latency_ratio_p50=_ratio(agent.latency_ms.p50, policy.latency_ms.p50),
        latency_ratio_p95=_ratio(agent.latency_ms.p95, policy.latency_ms.p95),
    )


def _ratio(value: float, base: float) -> float | None:
    return value / base if base else None


def evaluate_agent_hypotheses(
    agents: Mapping[str, AgentSummary], policies: Mapping[str, RunSummary]
) -> list[Check]:
    """H-A per effort: Agent costs >= 10x the one-call prior policy and takes >= 10x baseline."""
    prior, baseline = policies.get("prior.strict"), policies.get("baseline.strict")
    checks: list[Check] = []
    for effort, agent in agents.items():
        cost = _ratio(agent.cost_usd.mean, prior.cost_usd.mean) if prior is not None else None
        latency = (
            _ratio(agent.latency_ms.p50, baseline.latency_ms.p50) if baseline is not None else None
        )
        checks.append(
            check(
                "H-A",
                f"Agent ({effort}) costs at least 10x the prior policy per query",
                f"agent.{effort} mean cost / prior.strict",
                cost,
                ">=",
                COST_RATIO_MIN,
            )
        )
        checks.append(
            check(
                "H-A",
                f"Agent ({effort}) takes at least 10x the baseline's p50 latency",
                f"agent.{effort} p50 latency / baseline.strict",
                latency,
                ">=",
                LATENCY_RATIO_MIN,
            )
        )
    return checks


@dataclass(frozen=True)
class AgentEvalMetadata:
    benchmark_commit: str
    analysis_date: str  # UTC, YYYY-MM-DD
    queries: int  # the Agent subset
    subset_seed: int
    k: int
    efforts: list[str]
    policy_runs: list[str]
    bootstrap_seed: int
    bootstrap_resamples: int


@dataclass(frozen=True)
class AgentEvaluation:
    metadata: AgentEvalMetadata
    agents: dict[str, AgentSummary]  # by effort
    policies: dict[str, RunSummary]  # the policy runs restricted to the Agent subset
    comparisons: list[AgentComparison]
    checks: list[Check]


def build_agent_evaluation(
    policy_records: Mapping[str, RunRecord],
    graded: Mapping[str, GradedRecord],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    today: date | None = None,
) -> AgentEvaluation:
    """Summarize every graded Agent record and every policy run on the Agent's queries."""
    if not graded:
        raise ValueError("no graded Agent records")
    subsets = {frozenset(o.query_id for o in record.outcomes) for record in graded.values()}
    if len(subsets) != 1:
        raise ValueError("graded records do not share one subset of queries")
    (subset,) = subsets
    agents = {
        effort: summarize_agent(record.outcomes, seed=seed, resamples=resamples)
        for effort, record in sorted(graded.items())
    }
    policies: dict[str, RunSummary] = {}
    for name, record in sorted(policy_records.items()):
        outcomes = [o for o in record.outcomes if o.query_id in subset]
        if len(outcomes) != len(subset):
            raise ValueError(f"policy run {name} does not cover the Agent subset")
        policies[name] = summarize_run(outcomes, seed=seed, resamples=resamples)
    first = next(iter(graded.values())).metadata
    metadata = AgentEvalMetadata(
        benchmark_commit=first.benchmark_commit,
        analysis_date=(today or datetime.now(UTC).date()).isoformat(),
        queries=len(subset),
        subset_seed=first.subset_seed,
        k=first.k,
        efforts=list(agents),
        policy_runs=list(policies),
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )
    comparisons = [
        compare_agent(agent, policy) for agent in agents.values() for policy in policies.values()
    ]
    return AgentEvaluation(
        metadata, agents, policies, comparisons, evaluate_agent_hypotheses(agents, policies)
    )
