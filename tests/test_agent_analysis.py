"""Tests for comparing Agent with the policies, checked against hand-computed answers."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from exa_bench.agent_analysis import (
    compare_agent,
    load_graded_record,
    summarize_agent,
)
from exa_bench.agent_grade import GradedAgentOutcome, GradedMetadata, GradedRecord
from exa_bench.policy import QueryOutcome
from exa_bench.policy_analysis import summarize_run

RESAMPLES = 200


def graded(
    query_id: str,
    satisfying: int,
    *,
    violating: int = 0,
    unevaluable: int = 0,
    not_found: int = 0,
    duplicates: int = 0,
    cost: float | None = 0.025,
    server_ms: float | None = 20000.0,
    effort: str = "low",
    status: str = "completed",
) -> GradedAgentOutcome:
    found = satisfying + violating + unevaluable
    return GradedAgentOutcome(
        query_id=query_id,
        split="dynamic",
        clean=False,
        effort=effort,
        status=status,
        companies=found + not_found + duplicates,
        duplicates=duplicates,
        lookups=found + not_found,
        found=found,
        not_found=not_found,
        satisfying=satisfying,
        violating=violating,
        unevaluable=unevaluable,
        accepted_strict=min(10, satisfying),
        accepted_lenient=min(10, satisfying + unevaluable + not_found),
        cost_usd=cost,
        lookup_cost_usd=0.007 * (found + not_found),
        lookup_spent_usd=0.0,
        lookups_from_cache=found + not_found,
        server_ms=server_ms,
        client_ms=(server_ms or 0.0) + 500.0,
    )


def policy_outcome(
    query_id: str, accepted: int, policy: str, null_policy: str = "strict"
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        split="dynamic",
        clean=False,
        policy=policy,
        null_policy=null_policy,
        filters=1,
        expected_pass_rate=0.9,
        accepted=accepted,
        seen=10,
        duplicates=0,
        violating=10 - accepted,
        unevaluable=0,
        satisfying=accepted,
        requested=(10,),
        returned=(10,),
        from_cache=(True,),
        cost_usd=0.007 if policy == "baseline" else 0.014,
        spent_usd=0.0,
        latency_ms=1000.0,
        stopped_by="filled" if accepted >= 10 else "planner",
    )


GRADED_LOW = [
    graded("a", 10, violating=1, duplicates=2),  # filled strictly; 11 found, 13 returned
    graded("b", 6, unevaluable=4),  # filled only leniently
    graded("c", 2, violating=3, not_found=2, cost=0.03, server_ms=40000.0),
    graded("d", 0, cost=None, server_ms=None, status="failed"),
]
GRADED_RECORDS = {
    "low": GradedRecord(
        GradedMetadata("abc", "2026-09-26", "low", 10, 0.0, 4, 9, 22, 0, 0.0), tuple(GRADED_LOW)
    )
}


def test_summarize_agent_counts_fill_shares_cost_and_time() -> None:
    summary = summarize_agent(GRADED_LOW, seed=1, resamples=RESAMPLES)

    assert (summary.effort, summary.queries, summary.status) == (
        "low",
        4,
        {"completed": 3, "failed": 1},
    )
    assert (summary.filled_strict, summary.filled_lenient) == (1, 2)
    assert (summary.fill_rate_strict.value, summary.fill_rate_lenient.value) == (0.25, 0.5)
    assert (summary.mean_companies, summary.mean_accepted_strict) == (7.5, 4.5)  # 30 returned
    assert summary.lookups == 28
    assert (summary.found_share.numerator, summary.found_share.denominator) == (26, 28)
    assert (summary.violating_share.numerator, summary.violating_share.denominator) == (4, 26)
    assert (summary.unevaluable_share.numerator, summary.unevaluable_share.denominator) == (4, 26)
    assert summary.cost_usd.mean == pytest.approx((0.025 + 0.025 + 0.03) / 3)
    assert summary.latency_ms.p50 == 20000.0
    assert summary.lookup_cost_usd == pytest.approx(0.007 * 28)


def test_summarize_agent_rejects_empty_or_mixed_efforts() -> None:
    with pytest.raises(ValueError, match="no outcomes"):
        summarize_agent([], seed=1, resamples=RESAMPLES)
    with pytest.raises(ValueError, match="mix efforts"):
        summarize_agent([*GRADED_LOW, graded("z", 1, effort="medium")], seed=1, resamples=RESAMPLES)


def test_compare_agent_uses_the_policys_null_policy() -> None:
    agent = summarize_agent(GRADED_LOW, seed=1, resamples=RESAMPLES)
    strict = summarize_run(
        [policy_outcome(q, n, "prior") for q, n in {"a": 10, "b": 10, "c": 10, "d": 4}.items()],
        seed=1,
        resamples=RESAMPLES,
    )
    lenient = summarize_run(
        [policy_outcome(q, 10, "prior", "lenient") for q in "abcd"], seed=1, resamples=RESAMPLES
    )

    comparison = compare_agent(agent, strict)

    assert (comparison.effort, comparison.policy, comparison.fill_gain) == (
        "low",
        "prior.strict",
        -2,
    )
    assert comparison.cost_ratio_mean == pytest.approx((0.025 + 0.025 + 0.03) / 3 / 0.014)
    assert comparison.latency_ratio_p50 == 20.0
    assert comparison.latency_ratio_p95 == pytest.approx(38.0)  # p95 of 20, 20, 40 s is 38 s
    assert compare_agent(agent, lenient).fill_gain == 2 - 4


def test_load_graded_record_round_trips(tmp_path: Path) -> None:
    record = GRADED_RECORDS["low"]
    path = tmp_path / "graded.low.json"
    path.write_text(json.dumps(asdict(record)), encoding="utf-8")

    assert load_graded_record(path) == record
