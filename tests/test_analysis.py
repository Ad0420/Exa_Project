"""Tests for aggregating graded results, checked against hand-computed answers."""

import pytest

from exa_bench.constraints.analysis import Analysis, Bounds, QueryGrade, QueryRow, analyze
from exa_bench.constraints.grader import ConstraintOutcome, GradedResult, Outcome, ResultVerdict

RESAMPLES = 200  # keep the tests fast; interval quality is tested in test_stats


def graded(verdict: ResultVerdict, **outcomes: Outcome) -> GradedResult:
    """A result whose outcomes are given as key=outcome (op is fixed to "lte")."""
    return GradedResult(
        verdict, tuple(ConstraintOutcome(key, "lte", outcome) for key, outcome in outcomes.items())
    )


def repeated(verdict: ResultVerdict, count: int, **outcomes: Outcome) -> list[GradedResult]:
    return [graded(verdict, **outcomes)] * count


V, S, U, N = (
    ResultVerdict.VIOLATES,
    ResultVerdict.SATISFIES,
    ResultVerdict.UNEVALUABLE,
    ResultVerdict.NOT_CHECKABLE,
)
P, F, UNK, NC = Outcome.PASS, Outcome.FAIL, Outcome.UNKNOWN, Outcome.NOT_CHECKABLE

# Query A: 3 violate, 6 satisfy, 1 unevaluable; every result also has an uncheckable constraint.
QUERY_A = QueryGrade(
    "a",
    "employee_count",
    "dynamic",
    tuple(
        repeated(V, 3, employees=F, category=NC)
        + repeated(S, 6, employees=P, category=NC)
        + repeated(U, 1, employees=UNK, category=NC)
    ),
)
# Query B: 5 violate, 5 satisfy on a funding-stage constraint.
QUERY_B = QueryGrade(
    "b",
    "funding_stage",
    "dynamic",
    tuple(repeated(V, 5, funding_stage=F) + repeated(S, 5, funding_stage=P)),
)
# Query C: 4 satisfy plus one not-checkable result (no checkable constraint outcomes at all).
QUERY_C = QueryGrade(
    "c",
    "employee_count",
    "static",
    tuple(repeated(S, 4, employees=P) + repeated(N, 1, category=NC)),
)
GRADES = [QUERY_B, QUERY_A, QUERY_C]  # unsorted on purpose: output must sort


@pytest.fixture(scope="module")
def analysis() -> Analysis:
    return analyze(GRADES, seed=1, resamples=RESAMPLES)


def test_totals_and_verdict_counts(analysis: Analysis) -> None:
    assert analysis.queries == 3
    assert analysis.results == 25
    assert analysis.verdicts == {
        "violates": 8,
        "satisfies": 15,
        "unevaluable": 1,
        "not_checkable": 1,
    }


def test_violation_rate_excludes_unevaluable_and_not_checkable(analysis: Analysis) -> None:
    assert (analysis.violation_rate.numerator, analysis.violation_rate.denominator) == (8, 23)
    assert analysis.violation_rate.value == pytest.approx(8 / 23)


def test_bounds_treat_unevaluable_as_all_satisfying_or_all_violating(analysis: Analysis) -> None:
    # 24 graded results (not_checkable excluded): 8 violate, 1 unevaluable.
    assert analysis.violation_bounds.lower == pytest.approx(8 / 24)
    assert analysis.violation_bounds.upper == pytest.approx(9 / 24)
    assert analyze([], seed=1, resamples=RESAMPLES).violation_bounds == Bounds(None, None)


def test_rates_by_bucket_and_split(analysis: Analysis) -> None:
    assert list(analysis.by_bucket) == ["employee_count", "funding_stage"]
    assert analysis.by_bucket["employee_count"].value == pytest.approx(3 / 13)
    assert analysis.by_bucket["funding_stage"].value == pytest.approx(5 / 10)
    assert analysis.by_split["dynamic"].value == pytest.approx(8 / 19)
    assert analysis.by_split["static"].value == pytest.approx(0 / 4)


def test_breakdowns_partition_the_overall_counts(analysis: Analysis) -> None:
    for breakdown in (analysis.by_bucket, analysis.by_split):
        assert sum(r.numerator for r in breakdown.values()) == analysis.violation_rate.numerator
        assert sum(r.denominator for r in breakdown.values()) == analysis.violation_rate.denominator


def test_per_constraint_counts_every_outcome(analysis: Analysis) -> None:
    assert list(analysis.by_constraint) == ["category", "employees", "funding_stage"]

    employees = analysis.by_constraint["employees"]
    assert (employees.passed, employees.failed, employees.unknown, employees.not_checkable) == (
        10,
        3,
        1,
        0,
    )
    assert employees.fail_rate.value == pytest.approx(3 / 13)
    assert employees.queries == 2  # queries A and C carry it; B does not

    category = analysis.by_constraint["category"]
    assert (category.passed, category.failed, category.unknown, category.not_checkable) == (
        0,
        0,
        0,
        11,
    )
    assert category.fail_rate.value is None

    stage = analysis.by_constraint["funding_stage"]
    assert stage.queries == 1
    assert (stage.fail_rate.ci_low, stage.fail_rate.ci_high) == (0.5, 0.5)  # single cluster


def test_per_constraint_agrees_with_a_brute_force_count(analysis: Analysis) -> None:
    outcomes = [o for grade in GRADES for result in grade.results for o in result.outcomes]
    for key, stats in analysis.by_constraint.items():
        mine = [o.outcome for o in outcomes if o.key == key]
        assert (stats.passed, stats.failed, stats.unknown, stats.not_checkable) == (
            mine.count(P),
            mine.count(F),
            mine.count(UNK),
            mine.count(NC),
        )


def test_per_query_rows_and_histogram(analysis: Analysis) -> None:
    assert analysis.per_query == (
        QueryRow("b", "funding_stage", "dynamic", 10, 5, 5, 0, 0),
        QueryRow("a", "employee_count", "dynamic", 10, 3, 6, 1, 0),
        QueryRow("c", "employee_count", "static", 5, 0, 4, 0, 1),
    )
    assert analysis.satisfying_histogram == {4: 1, 5: 1, 6: 1}
    assert sum(analysis.satisfying_histogram.values()) == analysis.queries


def test_sections_are_consistent_with_the_rows(analysis: Analysis) -> None:
    rows = analysis.per_query
    assert analysis.violation_rate.numerator == sum(r.violates for r in rows)
    assert analysis.violation_rate.denominator == sum(r.evaluable for r in rows)
    assert analysis.results == sum(r.results for r in rows)


def test_records_bootstrap_settings(analysis: Analysis) -> None:
    assert (analysis.bootstrap_seed, analysis.bootstrap_resamples) == (1, RESAMPLES)


def test_query_with_no_results_is_kept() -> None:
    analysis = analyze(
        [QueryGrade("empty", "composite", "dynamic", ())], seed=1, resamples=RESAMPLES
    )

    assert analysis.per_query == (QueryRow("empty", "composite", "dynamic", 0, 0, 0, 0, 0),)
    assert analysis.violation_rate.value is None
    assert analysis.satisfying_histogram == {0: 1}
