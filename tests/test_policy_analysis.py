"""Tests for aggregating policy run records, checked against hand-computed answers."""

import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pytest

from exa_bench.policy import QueryOutcome, RunMetadata, RunRecord
from exa_bench.policy_analysis import (
    Spread,
    build_evaluation,
    compare,
    load_record,
    spread,
    summarize_run,
)

RESAMPLES = 200  # keep the tests fast; interval quality is tested in test_stats


def outcome(
    query_id: str,
    accepted: int,
    *,
    seen: int = 10,
    violating: int = 0,
    unevaluable: int = 0,
    duplicates: int = 0,
    requested: tuple[int, ...] = (10,),
    returned: tuple[int, ...] | None = None,
    cost: float = 0.007,
    latency: float = 100.0,
    clean: bool = False,
    split: str = "dynamic",
    stopped_by: str = "filled",
    expected: float = 0.9,
    policy: str = "fixed-25",
    null_policy: str = "strict",
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        split=split,
        clean=clean,
        policy=policy,
        null_policy=null_policy,
        filters=1,
        expected_pass_rate=expected,
        accepted=accepted,
        seen=seen,
        duplicates=duplicates,
        violating=violating,
        unevaluable=unevaluable,
        satisfying=seen - duplicates - violating - unevaluable,
        requested=requested,
        returned=returned or requested,
        from_cache=tuple(False for _ in requested),
        cost_usd=cost,
        spent_usd=cost,
        latency_ms=latency,
        stopped_by=stopped_by,
    )


# Four queries: a and c fill; b stops short; d runs Exa dry. c needed a second call.
OUTCOMES = [
    outcome(
        "a", 10, seen=25, violating=5, unevaluable=2, requested=(25,), cost=0.022, expected=0.8
    ),
    outcome(
        "b",
        4,
        seen=25,
        violating=15,
        unevaluable=6,
        requested=(25,),
        cost=0.022,
        latency=300.0,
        stopped_by="planner",
    ),
    outcome(
        "c",
        10,
        seen=35,
        duplicates=10,
        violating=5,
        requested=(10, 25),
        cost=0.029,
        latency=400.0,
        clean=True,
        split="static",
    ),
    outcome(
        "d",
        7,
        seen=8,
        returned=(8,),
        violating=1,
        latency=200.0,
        split="static",
        stopped_by="exhausted",
    ),
]
BASELINE = [
    outcome("a", 10, policy="baseline"),
    outcome("b", 3, policy="baseline", stopped_by="planner"),
    outcome(
        "c", 8, policy="baseline", latency=200.0, clean=True, split="static", stopped_by="planner"
    ),
    outcome("d", 7, policy="baseline", latency=200.0, split="static", stopped_by="planner"),
]


def test_summary_counts_fill_calls_cost_and_latency() -> None:
    summary = summarize_run(OUTCOMES, seed=1, resamples=RESAMPLES)

    assert (summary.policy, summary.null_policy, summary.queries, summary.filled) == (
        "fixed-25",
        "strict",
        4,
        2,
    )
    assert summary.fill_rate.value == 0.5
    assert summary.fill_rate.ci_low is not None
    assert summary.fill_rate.ci_low < 0.5
    assert {k: v.value for k, v in summary.fill_rate_by_split.items()} == {
        "dynamic": 0.5,
        "static": 0.5,
    }
    assert {k: v.value for k, v in summary.fill_rate_by_sample.items()} == {
        "clean": 1.0,
        "non_clean": pytest.approx(1 / 3),
    }
    assert summary.mean_accepted == 7.75
    assert (summary.one_call_share, summary.mean_calls) == (0.75, 1.25)
    assert summary.cost_usd.mean == pytest.approx(0.02)
    assert summary.cost_usd.p50 == 0.022
    assert summary.cost_usd.p95 == pytest.approx(0.02795)
    assert summary.cost_per_accepted_usd == pytest.approx(0.08 / 31)
    assert summary.latency_ms == Spread(250.0, 250.0, 385.0)
    assert summary.stopped_by == {"exhausted": 1, "filled": 2, "planner": 1}


def test_summary_pools_result_shares_over_distinct_results() -> None:
    summary = summarize_run(OUTCOMES, seed=1, resamples=RESAMPLES)

    assert (summary.violating_share.numerator, summary.violating_share.denominator) == (26, 83)
    assert (summary.unevaluable_share.numerator, summary.unevaluable_share.denominator) == (8, 83)
    assert summary.observed_pass_rate == pytest.approx(49 / 83)
    assert summary.expected_pass_rate == pytest.approx(0.875)


def test_summary_rejects_empty_or_mixed_runs() -> None:
    with pytest.raises(ValueError, match="no outcomes"):
        summarize_run([], seed=1, resamples=RESAMPLES)
    with pytest.raises(ValueError, match="mix runs"):
        summarize_run(OUTCOMES + BASELINE, seed=1, resamples=RESAMPLES)


def test_compare_against_the_baseline() -> None:
    run = summarize_run(OUTCOMES, seed=1, resamples=RESAMPLES)
    baseline = summarize_run(BASELINE, seed=1, resamples=RESAMPLES)

    comparison = compare(run, baseline)

    assert (comparison.policy, comparison.against, comparison.fill_gain) == (
        "fixed-25",
        "baseline",
        1,
    )
    assert comparison.cost_ratio_mean == pytest.approx(0.02 / 0.007)
    assert comparison.cost_ratio_p50 == pytest.approx(0.022 / 0.007)
    assert comparison.latency_ratio_p50 == pytest.approx(250 / 150)
    assert comparison.latency_ratio_p95 == pytest.approx(385 / 200)
    lenient = summarize_run(
        [replace(o, null_policy="lenient") for o in OUTCOMES], seed=1, resamples=RESAMPLES
    )
    with pytest.raises(ValueError, match="same null policy"):
        compare(lenient, baseline)


def test_spread_edges() -> None:
    assert spread([3.0]) == Spread(3.0, 3.0, 3.0)
    with pytest.raises(ValueError, match="at least one"):
        spread([])


def test_load_record_round_trips_the_policy_command_output(tmp_path: Path) -> None:
    metadata = RunMetadata(
        "abc", "2026-09-26", "fixed-25", "strict", 0.0, 10, 2, 1, 7, 1, 0.022, 1, 0.02
    )
    record = RunRecord(metadata, (OUTCOMES[0], OUTCOMES[2]))
    path = tmp_path / "fixed-25.strict.json"
    path.write_text(json.dumps(asdict(record)), encoding="utf-8")

    assert load_record(path) == record


def run_record(
    policy: str,
    null_policy: str,
    accepted: dict[str, int],
    *,
    cost: float = 0.007,
    requested: tuple[int, ...] = (10,),
    latency: float = 100.0,
) -> RunRecord:
    """A record over queries a-d: c is the clean sample; c and d are static."""
    outcomes = tuple(
        outcome(
            query_id,
            count,
            policy=policy,
            null_policy=null_policy,
            cost=cost,
            requested=requested,
            latency=latency,
            clean=query_id == "c",
            split="static" if query_id in ("c", "d") else "dynamic",
        )
        for query_id, count in accepted.items()
    )
    metadata = RunMetadata(
        "abc", "2026-09-26", policy, null_policy, 0.0, 10, len(accepted), 1, 7, 0, 0.0, 0, cost
    )
    return RunRecord(metadata, outcomes)


def prior_record() -> RunRecord:
    """Prior fills every query, but d needed a second call."""
    record = run_record("prior", "strict", dict.fromkeys("abcd", 10), cost=0.012, requested=(13,))
    second_call = replace(
        record.outcomes[3], requested=(13, 100), returned=(13, 100), from_cache=(False, False)
    )
    return RunRecord(record.metadata, (*record.outcomes[:3], second_call))


def adaptive_record() -> RunRecord:
    """Adaptive escalates once for every query, and d is slower than the rest."""
    record = run_record(
        "adaptive",
        "strict",
        {"a": 10, "b": 10, "c": 10, "d": 4},
        cost=0.029,
        requested=(10, 25),
        latency=200.0,
    )
    slow = replace(record.outcomes[3], latency_ms=400.0)
    return RunRecord(record.metadata, (*record.outcomes[:3], slow))


RECORDS = {
    "baseline.strict": run_record("baseline", "strict", {"a": 10, "b": 3, "c": 8, "d": 7}),
    "fixed-25.strict": run_record(
        "fixed-25", "strict", {"a": 10, "b": 4, "c": 10, "d": 10}, cost=0.022, requested=(25,)
    ),
    "prior.strict": prior_record(),
    "adaptive.strict": adaptive_record(),
    "baseline.lenient": run_record("baseline", "lenient", {"a": 10, "b": 10, "c": 8, "d": 7}),
    "fixed-25.lenient": run_record(
        "fixed-25", "lenient", dict.fromkeys("abcd", 10), cost=0.022, requested=(25,)
    ),
}


def test_evaluation_summarizes_compares_and_judges_the_hypotheses() -> None:
    evaluation = build_evaluation(RECORDS, seed=1, resamples=RESAMPLES, today=date(2026, 1, 1))

    assert list(evaluation.runs) == sorted(RECORDS)
    assert set(evaluation.comparisons) == set(RECORDS) - {"baseline.strict", "baseline.lenient"}
    assert evaluation.comparisons["fixed-25.strict"].fill_gain == 2
    assert evaluation.null_policy_effect == {"baseline": 1, "fixed-25": 1}
    meta = evaluation.metadata
    assert (meta.benchmark_commit, meta.analysis_date, meta.k, meta.queries) == (
        "abc",
        "2026-01-01",
        10,
        4,
    )
    assert (meta.clean_sample, meta.clean_sample_seed, meta.bootstrap_seed) == (1, 7, 1)
    assert meta.spent_usd_total == pytest.approx(0.007 + 0.022 + 0.012 + 0.029 + 0.007 + 0.022)
    assert meta.records["prior.strict"].run_date == "2026-09-26"
    checks = {(check.hypothesis, check.metric): check for check in evaluation.checks}
    assert len(checks) == 11
    expected = {
        ("H-D", "fixed-25.strict fill rate on non-clean queries"): (2 / 3, False),
        ("H-P1", "fixed-25.strict fill rate"): (0.75, False),
        ("H-P1", "fixed-25.strict mean cost / baseline"): (22 / 7, False),
        ("H-P2", "adaptive.strict fill rate"): (0.75, False),
        ("H-P2", "adaptive.strict p50 cost / baseline"): (29 / 7, False),
        ("H-P2", "adaptive.strict p50 latency / baseline"): (2.0, False),
        ("H-P2", "adaptive.strict p95 latency / baseline"): (3.7, False),  # 370 ms vs 100
        ("H-P3", "prior.strict fill rate"): (1.0, True),
        ("H-P3", "prior.strict one-call share"): (0.75, False),
        ("H-P3", "prior.strict mean cost / adaptive.strict mean cost"): (12 / 29, True),
        ("H-P3", "prior.strict p95 latency / baseline"): (1.0, True),
    }
    for key, (value, passed) in expected.items():
        assert checks[key].value == pytest.approx(value), key
        assert checks[key].passed is passed, key


def test_checks_are_undecided_without_their_runs() -> None:
    only_baseline = {"baseline.strict": RECORDS["baseline.strict"]}

    evaluation = build_evaluation(only_baseline, seed=1, resamples=RESAMPLES)

    assert evaluation.comparisons == {}
    assert evaluation.null_policy_effect == {}
    assert [check.value for check in evaluation.checks] == [None] * 11
    assert [check.passed for check in evaluation.checks] == [None] * 11


def test_evaluation_requires_one_workload() -> None:
    other = run_record("prior", "strict", {"a": 10, "b": 10, "c": 10})
    mixed = {"baseline.strict": RECORDS["baseline.strict"], "prior.strict": other}

    with pytest.raises(ValueError, match="one workload"):
        build_evaluation(mixed, seed=1, resamples=RESAMPLES)
    with pytest.raises(ValueError, match="no run records"):
        build_evaluation({}, seed=1, resamples=RESAMPLES)
