"""Tests for aggregating policy run records, checked against hand-computed answers."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from exa_bench.policy import QueryOutcome, RunMetadata, RunRecord
from exa_bench.policy_analysis import Spread, compare, load_record, spread, summarize_run

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
