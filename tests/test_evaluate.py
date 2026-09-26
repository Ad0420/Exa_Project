"""Tests for the shared filter evaluation engine."""

from datetime import date

import pytest

from exa_filters.constraints import DateField, DateOp, NumberField, NumberOp, Verdict
from exa_filters.evaluate import (
    CountryFilter,
    DateFilter,
    Filter,
    NumberFilter,
    StageFilter,
    evaluate_filter,
    evaluate_filters,
)

P, F, U = Verdict.PASS, Verdict.FAIL, Verdict.UNKNOWN

# 120 people, Berlin, Series B in March 2024, $25M raised, founded 2019.
ACME: dict[str, object] = {
    "foundedYear": 2019,
    "workforce": {"total": 120},
    "headquarters": {"country": "DE"},
    "financials": {
        "fundingTotal": 25_000_000,
        "fundingLatestRound": {"name": "Series b", "date": "2024-03-01"},
    },
}


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100), F),
        (NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100, tolerance=0.2), P),  # 120 <= 120
        (NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 99, tolerance=0.2), F),  # 120 > 118.8
        (NumberFilter(NumberField.EMPLOYEES, NumberOp.GTE, 150, tolerance=0.2), P),  # 120 >= 120
        (NumberFilter(NumberField.EMPLOYEES, NumberOp.GTE, 151, tolerance=0.2), F),
        (NumberFilter(NumberField.EMPLOYEES, NumberOp.EQ, 120, tolerance=0.2), P),  # eq unwidened
        (NumberFilter(NumberField.FOUNDED_YEAR, NumberOp.EQ, 2019), P),
        (NumberFilter(NumberField.FUNDING, NumberOp.GTE, 30_000_000), F),
        (CountryFilter(("Germany",)), P),  # ISO code in the data, name in the filter
        (CountryFilter(("France", "germany")), P),
        (CountryFilter(("France", "Spain")), F),
        (StageFilter("Series B"), P),
        (StageFilter("Seed"), F),
        (DateFilter(DateField.FUNDING_DATE, DateOp.GTE, date(2024, 1, 1)), P),
        (DateFilter(DateField.FUNDING_DATE, DateOp.LTE, date(2024, 2, 29)), F),
    ],
)
def test_evaluate_filter(item: Filter, expected: Verdict) -> None:
    assert evaluate_filter(item, ACME) is expected


def test_missing_fields_are_unknown() -> None:
    assert evaluate_filter(NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 1), {}) is U
    assert evaluate_filter(CountryFilter(("Germany",)), {}) is U
    assert evaluate_filter(StageFilter("Seed"), {}) is U
    assert (
        evaluate_filter(DateFilter(DateField.FUNDING_DATE, DateOp.GTE, date(2024, 1, 1)), {}) is U
    )


def test_non_ladder_rounds_are_unknown_for_stage_filters() -> None:
    venture = {"financials": {"fundingLatestRound": {"name": "Venture"}}}

    assert evaluate_filter(StageFilter("Seed"), venture) is U
    assert (
        evaluate_filter(StageFilter("Venture"), venture) is P
    )  # a non-ladder filter matches literally
    assert evaluate_filter(StageFilter("Grant"), venture) is F


def test_seed_filter_does_not_match_a_pre_seed_round() -> None:
    pre_seed = {"financials": {"fundingLatestRound": {"name": "Pre seed"}}}

    assert evaluate_filter(StageFilter("Seed"), pre_seed) is F
    assert evaluate_filter(StageFilter("Pre-Seed"), pre_seed) is P


def test_evaluate_filters_keeps_order() -> None:
    items: list[Filter] = [
        CountryFilter(("Germany",)),
        StageFilter("Seed"),
        StageFilter("Series B"),
    ]

    assert evaluate_filters(items, ACME) == [P, F, P]
    assert evaluate_filters([], ACME) == []


def test_invalid_number_filters_are_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100, tolerance=1.0)
    with pytest.raises(ValueError, match="tolerance"):
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100, tolerance=-0.1)
    with pytest.raises(ValueError, match="finite number"):
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, float("nan"))


def test_invalid_country_and_stage_filters_are_rejected() -> None:
    with pytest.raises(ValueError, match="countries"):
        CountryFilter(())
    with pytest.raises(ValueError, match="countries"):
        CountryFilter(("Germany", " "))
    with pytest.raises(ValueError, match="stage"):
        StageFilter("  ")


def test_date_filter_requires_a_plain_date() -> None:
    with pytest.raises(ValueError, match=r"datetime\.date"):
        DateFilter(DateField.FUNDING_DATE, DateOp.GTE, "2024-01-01")  # type: ignore[arg-type]
