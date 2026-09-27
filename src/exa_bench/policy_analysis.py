"""Aggregate the policy run records into the numbers the evaluation reports."""

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exa_bench.policy import QueryOutcome, RunMetadata, RunRecord
from exa_bench.stats import DEFAULT_RESAMPLES, Cluster, Rate, rate


def load_record(path: Path) -> RunRecord:
    """Read a run record written by the policy command."""
    data = json.loads(path.read_text(encoding="utf-8"))
    outcomes = tuple(_outcome(row) for row in data["outcomes"])
    return RunRecord(RunMetadata(**data["metadata"]), outcomes)


def _outcome(row: dict[str, Any]) -> QueryOutcome:
    return QueryOutcome(
        query_id=row["query_id"],
        split=row["split"],
        clean=row["clean"],
        policy=row["policy"],
        null_policy=row["null_policy"],
        filters=row["filters"],
        expected_pass_rate=row["expected_pass_rate"],
        accepted=row["accepted"],
        seen=row["seen"],
        duplicates=row["duplicates"],
        violating=row["violating"],
        unevaluable=row["unevaluable"],
        satisfying=row["satisfying"],
        requested=tuple(row["requested"]),
        returned=tuple(row["returned"]),
        from_cache=tuple(row["from_cache"]),
        cost_usd=row["cost_usd"],
        spent_usd=row["spent_usd"],
        latency_ms=row["latency_ms"],
        stopped_by=row["stopped_by"],
    )


@dataclass(frozen=True)
class Spread:
    mean: float
    p50: float
    p95: float


def spread(values: Sequence[float]) -> Spread:
    if not values:
        raise ValueError("spread needs at least one value")
    if len(values) == 1:
        return Spread(values[0], values[0], values[0])
    cuts = statistics.quantiles(values, n=20, method="inclusive")  # 5%, 10%, ..., 95%
    return Spread(statistics.mean(values), statistics.median(values), cuts[-1])


@dataclass(frozen=True)
class RunSummary:
    policy: str
    null_policy: str
    queries: int
    filled: int
    fill_rate: Rate  # queries filled to k; cluster-bootstrap CI by query
    fill_rate_by_split: dict[str, Rate]  # static / dynamic
    fill_rate_by_sample: dict[str, Rate]  # clean / non_clean
    mean_accepted: float
    one_call_share: float
    mean_calls: float
    cost_usd: Spread  # per query
    cost_per_accepted_usd: float | None  # total cost over total accepted results
    latency_ms: Spread  # per query, calls summed
    violating_share: Rate  # of distinct results fetched; CI by query
    unevaluable_share: Rate
    expected_pass_rate: float  # mean of the prior planner's per-query estimate
    observed_pass_rate: float | None  # satisfying over distinct results fetched
    stopped_by: dict[str, int]


def summarize_run(
    outcomes: Sequence[QueryOutcome], *, seed: int, resamples: int = DEFAULT_RESAMPLES
) -> RunSummary:
    if not outcomes:
        raise ValueError("no outcomes to summarize")
    runs = {(outcome.policy, outcome.null_policy) for outcome in outcomes}
    if len(runs) != 1:
        raise ValueError(f"outcomes mix runs: {sorted(runs)}")
    ((policy, null_policy),) = runs
    queries = len(outcomes)
    accepted = sum(outcome.accepted for outcome in outcomes)
    distinct = sum(_distinct(outcome) for outcome in outcomes)
    return RunSummary(
        policy=policy,
        null_policy=null_policy,
        queries=queries,
        filled=sum(outcome.filled for outcome in outcomes),
        fill_rate=rate(_fill_clusters(outcomes), seed=seed, resamples=resamples),
        fill_rate_by_split=_fill_rate_by(outcomes, lambda o: o.split, seed, resamples),
        fill_rate_by_sample=_fill_rate_by(
            outcomes, lambda o: "clean" if o.clean else "non_clean", seed, resamples
        ),
        mean_accepted=accepted / queries,
        one_call_share=sum(len(outcome.requested) == 1 for outcome in outcomes) / queries,
        mean_calls=sum(len(outcome.requested) for outcome in outcomes) / queries,
        cost_usd=spread([outcome.cost_usd for outcome in outcomes]),
        cost_per_accepted_usd=_ratio(sum(outcome.cost_usd for outcome in outcomes), accepted),
        latency_ms=spread([outcome.latency_ms for outcome in outcomes]),
        violating_share=rate(
            [(o.violating, _distinct(o)) for o in outcomes], seed=seed, resamples=resamples
        ),
        unevaluable_share=rate(
            [(o.unevaluable, _distinct(o)) for o in outcomes], seed=seed, resamples=resamples
        ),
        expected_pass_rate=statistics.mean(outcome.expected_pass_rate for outcome in outcomes),
        observed_pass_rate=_ratio(sum(outcome.satisfying for outcome in outcomes), distinct),
        stopped_by=dict(sorted(Counter(outcome.stopped_by for outcome in outcomes).items())),
    )


def _distinct(outcome: QueryOutcome) -> int:
    return outcome.seen - outcome.duplicates


def _fill_clusters(outcomes: Sequence[QueryOutcome]) -> list[Cluster]:
    return [(int(outcome.filled), 1) for outcome in outcomes]


def _fill_rate_by(
    outcomes: Sequence[QueryOutcome],
    key: Callable[[QueryOutcome], str],
    seed: int,
    resamples: int,
) -> dict[str, Rate]:
    groups: defaultdict[str, list[QueryOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[key(outcome)].append(outcome)
    return {
        name: rate(_fill_clusters(group), seed=seed, resamples=resamples)
        for name, group in sorted(groups.items())
    }


def _ratio(value: float, base: float) -> float | None:
    return value / base if base else None


@dataclass(frozen=True)
class Comparison:
    """A run against the baseline on the same queries under the same null policy."""

    policy: str
    null_policy: str
    against: str
    fill_gain: int  # queries filled beyond the baseline's
    cost_ratio_mean: float | None
    cost_ratio_p50: float | None
    latency_ratio_p50: float | None
    latency_ratio_p95: float | None


def compare(run: RunSummary, baseline: RunSummary) -> Comparison:
    if run.null_policy != baseline.null_policy or run.queries != baseline.queries:
        raise ValueError("compare runs of the same null policy over the same queries")
    return Comparison(
        policy=run.policy,
        null_policy=run.null_policy,
        against=baseline.policy,
        fill_gain=run.filled - baseline.filled,
        cost_ratio_mean=_ratio(run.cost_usd.mean, baseline.cost_usd.mean),
        cost_ratio_p50=_ratio(run.cost_usd.p50, baseline.cost_usd.p50),
        latency_ratio_p50=_ratio(run.latency_ms.p50, baseline.latency_ms.p50),
        latency_ratio_p95=_ratio(run.latency_ms.p95, baseline.latency_ms.p95),
    )
