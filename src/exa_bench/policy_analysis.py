"""Aggregate the policy run records into the numbers the evaluation reports."""

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from exa_bench.policy import QueryOutcome, RunMetadata, RunRecord
from exa_bench.stats import DEFAULT_RESAMPLES, Cluster, Rate, rate

type Direction = Literal[">=", "<="]


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


@dataclass(frozen=True)
class Check:
    """One pre-registered threshold, judged on a measured value; None when its run is absent."""

    hypothesis: str
    claim: str
    metric: str
    value: float | None
    direction: Direction
    threshold: float
    passed: bool | None


def _check(
    hypothesis: str,
    claim: str,
    metric: str,
    value: float | None,
    direction: Direction,
    threshold: float,
) -> Check:
    passed = None
    if value is not None:
        passed = value >= threshold if direction == ">=" else value <= threshold
    return Check(hypothesis, claim, metric, value, direction, threshold, passed)


def evaluate_hypotheses(
    runs: Mapping[str, RunSummary], comparisons: Mapping[str, Comparison]
) -> list[Check]:
    """The Part 2 hypotheses with the thresholds fixed before any run (strict null policy)."""
    fixed, prior, adaptive = (
        runs.get("fixed-25.strict"),
        runs.get("prior.strict"),
        runs.get("adaptive.strict"),
    )

    def ratio(run: str, field: str) -> float | None:
        comparison = comparisons.get(run)
        value = getattr(comparison, field) if comparison is not None else None
        return value if isinstance(value, float) else None

    prior_vs_adaptive = None
    if prior is not None and adaptive is not None:
        prior_vs_adaptive = _ratio(prior.cost_usd.mean, adaptive.cost_usd.mean)
    return [
        _check(
            "H-D",
            "10 satisfying results exist within Exa's top 25 for at least 80% of non-clean queries",
            "fixed-25.strict fill rate on non-clean queries",
            _fill(fixed, "non_clean"),
            ">=",
            0.80,
        ),
        _check(
            "H-P1",
            "fixed-25 fills at least 85% of queries",
            "fixed-25.strict fill rate",
            _fill(fixed),
            ">=",
            0.85,
        ),
        _check(
            "H-P1",
            "fixed-25 costs at most 2.5x the baseline",
            "fixed-25.strict mean cost / baseline",
            ratio("fixed-25.strict", "cost_ratio_mean"),
            "<=",
            2.5,
        ),
        _check(
            "H-P2",
            "adaptive fills at least 95% of queries",
            "adaptive.strict fill rate",
            _fill(adaptive),
            ">=",
            0.95,
        ),
        _check(
            "H-P2",
            "adaptive's median cost is at most 1.5x the baseline",
            "adaptive.strict p50 cost / baseline",
            ratio("adaptive.strict", "cost_ratio_p50"),
            "<=",
            1.5,
        ),
        _check(
            "H-P2",
            "adaptive's p50 latency is at most 1.2x the baseline",
            "adaptive.strict p50 latency / baseline",
            ratio("adaptive.strict", "latency_ratio_p50"),
            "<=",
            1.2,
        ),
        _check(
            "H-P2",
            "adaptive's p95 latency is at most 2x the baseline",
            "adaptive.strict p95 latency / baseline",
            ratio("adaptive.strict", "latency_ratio_p95"),
            "<=",
            2.0,
        ),
        _check(
            "H-P3",
            "prior fills at least 90% of queries",
            "prior.strict fill rate",
            _fill(prior),
            ">=",
            0.90,
        ),
        _check(
            "H-P3",
            "prior answers at least 90% of queries in one call",
            "prior.strict one-call share",
            prior.one_call_share if prior is not None else None,
            ">=",
            0.90,
        ),
        _check(
            "H-P3",
            "prior costs no more than adaptive",
            "prior.strict mean cost / adaptive.strict mean cost",
            prior_vs_adaptive,
            "<=",
            1.0,
        ),
        _check(
            "H-P3",
            "prior's p95 latency is at most 1.3x the baseline",
            "prior.strict p95 latency / baseline",
            ratio("prior.strict", "latency_ratio_p95"),
            "<=",
            1.3,
        ),
    ]


def _fill(summary: RunSummary | None, sample: str | None = None) -> float | None:
    if summary is None:
        return None
    if sample is None:
        return summary.fill_rate.value
    group = summary.fill_rate_by_sample.get(sample)
    return group.value if group is not None else None


@dataclass(frozen=True)
class RecordSummary:
    run_date: str
    calls_sent: int
    spent_usd: float


@dataclass(frozen=True)
class EvalMetadata:
    benchmark_commit: str
    analysis_date: str  # UTC, YYYY-MM-DD
    k: int
    queries: int
    clean_sample: int
    clean_sample_seed: int
    bootstrap_seed: int
    bootstrap_resamples: int
    records: dict[str, RecordSummary]  # run name -> when it ran and what it cost
    spent_usd_total: float


@dataclass(frozen=True)
class Evaluation:
    metadata: EvalMetadata
    runs: dict[str, RunSummary]
    comparisons: dict[str, Comparison]  # every non-baseline run against its baseline
    null_policy_effect: dict[str, int]  # policy -> queries filled under lenient beyond strict
    checks: list[Check]


def build_evaluation(
    records: Mapping[str, RunRecord],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    today: date | None = None,
) -> Evaluation:
    """Summarize every run record (keyed "policy.null_policy") over one shared workload."""
    if not records:
        raise ValueError("no run records")
    _require_one_workload(records)
    runs = {
        name: summarize_run(record.outcomes, seed=seed, resamples=resamples)
        for name, record in sorted(records.items())
    }
    comparisons = {
        name: compare(summary, runs[f"baseline.{summary.null_policy}"])
        for name, summary in runs.items()
        if summary.policy != "baseline" and f"baseline.{summary.null_policy}" in runs
    }
    effect = {
        summary.policy: runs[f"{summary.policy}.lenient"].filled - summary.filled
        for summary in runs.values()
        if summary.null_policy == "strict" and f"{summary.policy}.lenient" in runs
    }
    first = next(iter(records.values())).metadata
    metadata = EvalMetadata(
        benchmark_commit=first.benchmark_commit,
        analysis_date=(today or datetime.now(UTC).date()).isoformat(),
        k=first.k,
        queries=first.queries,
        clean_sample=first.clean_sample,
        clean_sample_seed=first.clean_sample_seed,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
        records={
            name: RecordSummary(r.metadata.run_date, r.metadata.calls_sent, r.metadata.spent_usd)
            for name, r in sorted(records.items())
        },
        spent_usd_total=sum(record.metadata.spent_usd for record in records.values()),
    )
    return Evaluation(metadata, runs, comparisons, effect, evaluate_hypotheses(runs, comparisons))


def _require_one_workload(records: Mapping[str, RunRecord]) -> None:
    """Runs are only comparable over the same queries, k, clean sample, and benchmark."""
    signatures = {
        (
            frozenset(outcome.query_id for outcome in record.outcomes),
            record.metadata.k,
            record.metadata.clean_sample,
            record.metadata.clean_sample_seed,
            record.metadata.benchmark_commit,
        )
        for record in records.values()
    }
    if len(signatures) != 1:
        raise ValueError("run records do not share one workload")
