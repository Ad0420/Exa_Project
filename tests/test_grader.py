"""Tests for grading a company result against benchmark constraints."""

import pytest

from exa_bench.constraints.grader import (
    ConstraintOutcome,
    Outcome,
    ResultVerdict,
    checkable_constraints,
    filters_from_constraints,
    grade,
)
from exa_filters.constraints import NumberField, NumberOp, Verdict
from exa_filters.evaluate import CountryFilter, NumberFilter, StageFilter, evaluate_filters

# A company as Exa returns it: 120 people, Berlin, Series B in March 2024, $25M raised.
ACME: dict[str, object] = {
    "foundedYear": 2019,
    "workforce": {"total": 120},
    "headquarters": {"city": "Berlin", "country": "Germany"},
    "financials": {
        "fundingTotal": 25_000_000,
        "fundingLatestRound": {"name": "Series b", "date": "2024-03-01"},
    },
}


def outcomes(result_constraints: dict[str, object], company: dict[str, object] = ACME) -> list[str]:
    return [f"{o.key}.{o.op}={o.outcome}" for o in grade(result_constraints, company).outcomes]


def test_all_pass_is_satisfies() -> None:
    result = grade(
        {
            "country": {"eq": "Germany"},
            "employees": {"gte": 100, "lte": 200},
            "funding_stage": {"contains": "Series B"},
        },
        ACME,
    )

    assert result.verdict == ResultVerdict.SATISFIES
    assert result.outcomes == (
        ConstraintOutcome("country", "eq", Outcome.PASS),
        ConstraintOutcome("employees", "gte", Outcome.PASS),
        ConstraintOutcome("employees", "lte", Outcome.PASS),
        ConstraintOutcome("funding_stage", "contains", Outcome.PASS),
    )


def test_any_fail_is_violates_even_with_unknowns() -> None:
    no_funding = {**ACME, "financials": None}
    result = grade({"country": {"eq": "France"}, "funding": {"gte": 1}}, no_funding)

    assert result.verdict == ResultVerdict.VIOLATES
    assert [o.outcome for o in result.outcomes] == [Outcome.FAIL, Outcome.UNKNOWN]


def test_unknown_without_fail_is_unevaluable() -> None:
    no_funding = {**ACME, "financials": None}
    result = grade({"country": {"eq": "Germany"}, "funding": {"gte": 1}}, no_funding)

    assert result.verdict == ResultVerdict.UNEVALUABLE


def test_only_uncheckable_constraints_is_not_checkable() -> None:
    result = grade(
        {
            "category": {"contains": "fintech"},
            "criteria": [{"description": "AI infra"}],
            "company": {"homepage": "https://x.test"},
            "founded_year": {"not_eq": 2019},
            "country": {"not_in": ["Germany"]},
        },
        ACME,
    )

    assert result.verdict == ResultVerdict.NOT_CHECKABLE
    assert {o.outcome for o in result.outcomes} == {Outcome.NOT_CHECKABLE}
    assert [o.op for o in result.outcomes] == ["contains", "", "homepage", "not_eq", "not_in"]


def test_checkable_constraints_counts_ops_the_grader_can_check() -> None:
    assert (
        checkable_constraints(
            {"employees": {"gte": 60, "lte": 100}, "category": {"contains": "ai"}}
        )
        == 2
    )
    assert (
        checkable_constraints(
            {"country": {"in": ["France"]}, "funding_date": {"gte": "2024-01-01"}}
        )
        == 2
    )
    assert (
        checkable_constraints(
            {"category": {"contains": "ai"}, "criteria": [], "founded_year": {"not_eq": 1}}
        )
        == 0
    )
    assert checkable_constraints({}) == 0


def test_uncheckable_constraints_do_not_affect_a_checkable_verdict() -> None:
    assert grade({"category": {"contains": "x"}, "country": {"eq": "Germany"}}, ACME).verdict == (
        ResultVerdict.SATISFIES
    )


def test_number_constraints() -> None:
    assert outcomes({"founded_year": {"eq": 2019}}) == ["founded_year.eq=pass"]
    assert outcomes({"founded_year": {"gte": 2016, "lte": 2018}}) == [
        "founded_year.gte=pass",
        "founded_year.lte=fail",
    ]
    assert outcomes({"funding": {"lte": 30_000_000}}) == ["funding.lte=pass"]
    assert outcomes({"funding": {"gte": 100_000_000}}) == ["funding.gte=fail"]


def test_country_in_list() -> None:
    assert outcomes({"country": {"in": ["France", "germany"]}}) == ["country.in=pass"]
    assert outcomes({"country": {"in": ["France", "Spain"]}}) == ["country.in=fail"]
    no_hq = {**ACME, "headquarters": None}
    assert outcomes({"country": {"in": ["France"]}}, no_hq) == ["country.in=unknown"]


def test_funding_stage_uses_stage_matching() -> None:
    assert outcomes({"funding_stage": {"contains": "Series B"}}) == ["funding_stage.contains=pass"]
    assert outcomes({"funding_stage": {"contains": "Seed"}}) == ["funding_stage.contains=fail"]
    pre_seed = {**ACME, "financials": {"fundingLatestRound": {"name": "Pre seed"}}}
    assert outcomes({"funding_stage": {"contains": "Seed"}}, pre_seed) == [
        "funding_stage.contains=fail"
    ]
    assert outcomes({"funding_stage": {"contains": "Pre-Seed"}}, pre_seed) == [
        "funding_stage.contains=pass"
    ]
    no_round = {**ACME, "financials": {"fundingLatestRound": None}}
    assert outcomes({"funding_stage": {"contains": "Seed"}}, no_round) == [
        "funding_stage.contains=unknown"
    ]


def test_funding_date_bounds() -> None:
    assert outcomes({"funding_date": {"gte": "2024-01-01"}}) == ["funding_date.gte=pass"]
    assert outcomes({"funding_date": {"gte": "2024-06-01"}}) == ["funding_date.gte=fail"]
    assert outcomes({"funding_date": {"lte": "2024-03-01"}}) == ["funding_date.lte=pass"]


@pytest.mark.parametrize(
    ("bound", "op", "strict", "tolerant"),
    [
        (100, "lte", Outcome.FAIL, Outcome.PASS),  # 120 <= 100 * 1.2
        (99, "lte", Outcome.FAIL, Outcome.FAIL),  # 120 > 118.8
        (150, "gte", Outcome.FAIL, Outcome.PASS),  # 120 >= 150 * 0.8
        (151, "gte", Outcome.FAIL, Outcome.FAIL),  # 120 < 120.8
    ],
)
def test_employee_tolerance_widens_bounds(
    bound: int, op: str, strict: Outcome, tolerant: Outcome
) -> None:
    constraints = {"employees": {op: bound}}

    assert grade(constraints, ACME).outcomes[0].outcome == strict
    assert grade(constraints, ACME, employee_tolerance=0.2).outcomes[0].outcome == tolerant


def test_tolerance_does_not_apply_to_other_numbers() -> None:
    strict = grade({"funding": {"lte": 24_000_000}}, ACME).outcomes[0].outcome
    tolerant = (
        grade({"funding": {"lte": 24_000_000}}, ACME, employee_tolerance=0.2).outcomes[0].outcome
    )

    assert (strict, tolerant) == (Outcome.FAIL, Outcome.FAIL)


@pytest.mark.parametrize(
    "constraints",
    [
        {"employees": {"lte": "30"}},
        {"employees": {"lte": True}},
        {"country": {"eq": 49}},
        {"country": {"in": "Germany"}},
        {"funding_stage": {"contains": ["Seed"]}},
        {"funding_date": {"gte": "March 2024"}},
        {"funding_date": {"gte": 2024}},
    ],
)
def test_malformed_benchmark_values_raise(constraints: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="needs"):
        grade(constraints, ACME)


@pytest.mark.parametrize(
    ("asked", "actual", "expected"),
    [
        ("Series A", "Series c", Outcome.FAIL),  # a later ladder stage is evidence against
        ("Seed", "Pre seed", Outcome.FAIL),
        ("Seed", "Venture", Outcome.UNKNOWN),  # "series unknown" is not evidence either way
        ("Series B", "Grant", Outcome.UNKNOWN),
        ("Series A", "Private equity", Outcome.UNKNOWN),
        ("Grant", "Grant", Outcome.PASS),  # a non-ladder filter still matches literally
        ("Grant", "Seed", Outcome.FAIL),
    ],
)
def test_non_ladder_rounds_are_unknown_for_stage_constraints(
    asked: str, actual: str, expected: Outcome
) -> None:
    company = {**ACME, "financials": {"fundingLatestRound": {"name": actual}}}

    assert grade({"funding_stage": {"contains": asked}}, company).outcomes[0].outcome == expected


def test_country_checks_tolerate_iso_codes_and_aliases() -> None:
    coded = {**ACME, "headquarters": {"country": "DE"}}
    british = {**ACME, "headquarters": {"country": "United Kingdom"}}

    assert outcomes({"country": {"eq": "Germany"}}, coded) == ["country.eq=pass"]
    assert outcomes({"country": {"eq": "France"}}, coded) == ["country.eq=fail"]
    assert outcomes({"country": {"in": ["Germany", "France", "UK"]}}, british) == [
        "country.in=pass"
    ]


def status(verdicts: list[Verdict]) -> ResultVerdict:
    if Verdict.FAIL in verdicts:
        return ResultVerdict.VIOLATES
    if Verdict.UNKNOWN in verdicts:
        return ResultVerdict.UNEVALUABLE
    return ResultVerdict.SATISFIES


@pytest.mark.parametrize("tolerance", [0.0, 0.2])
def test_filters_from_constraints_accept_exactly_what_grade_marks_satisfying(
    tolerance: float,
) -> None:
    constraints: dict[str, object] = {
        "industry": "fintech",
        "employees": {"gte": 50, "lte": 100},
        "country": {"eq": "Germany"},
        "funding_stage": {"contains": "Series B"},
    }

    filters = filters_from_constraints(constraints, employee_tolerance=tolerance)

    assert filters == [
        NumberFilter(NumberField.EMPLOYEES, NumberOp.GTE, 50.0, tolerance),
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100.0, tolerance),
        CountryFilter(("Germany",)),
        StageFilter("Series B"),
    ]
    companies = [ACME, {**ACME, "workforce": {"total": 80}}, {**ACME, "workforce": {}}]
    verdicts = [status(evaluate_filters(filters, company)) for company in companies]
    assert verdicts == [
        grade(constraints, company, employee_tolerance=tolerance).verdict for company in companies
    ]
    assert verdicts[0] is (ResultVerdict.SATISFIES if tolerance else ResultVerdict.VIOLATES)
    assert verdicts[1:] == [ResultVerdict.SATISFIES, ResultVerdict.UNEVALUABLE]
