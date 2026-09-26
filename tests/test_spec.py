"""Tests for the Filters builder."""

from datetime import date, datetime

import pytest

from exa_filters.constraints import DateField, DateOp, NumberField, NumberOp, Verdict
from exa_filters.evaluate import (
    CountryFilter,
    DateFilter,
    NumberFilter,
    StageFilter,
    evaluate_filters,
)
from exa_filters.spec import Contains, Eq, Filters, In, Range


def test_builds_typed_filters_in_a_fixed_order() -> None:
    spec = Filters(
        funding_date=Range(gte=date(2024, 1, 1)),
        funding_stage=Contains("Series A"),
        country=In(["SG", "Singapore"]),
        funding=Range(gte=1_000_000),
        employees=Range(gte=10, lte=30, tolerance=0.2),
        founded_year=Range(eq=2021),
    )

    assert spec.items == (
        NumberFilter(NumberField.FOUNDED_YEAR, NumberOp.EQ, 2021.0),
        NumberFilter(NumberField.EMPLOYEES, NumberOp.GTE, 10.0, 0.2),
        NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 30.0, 0.2),
        NumberFilter(NumberField.FUNDING, NumberOp.GTE, 1_000_000.0),
        CountryFilter(("SG", "Singapore")),
        StageFilter("Series A"),
        DateFilter(DateField.FUNDING_DATE, DateOp.GTE, date(2024, 1, 1)),
    )


def test_country_eq_and_funding_range() -> None:
    spec = Filters(country=Eq("Germany"), funding=Range(gte=20_000_000, lte=100_000_000))

    assert spec.items == (
        NumberFilter(NumberField.FUNDING, NumberOp.GTE, 20_000_000.0),
        NumberFilter(NumberField.FUNDING, NumberOp.LTE, 100_000_000.0),
        CountryFilter(("Germany",)),
    )
    assert Filters().items == ()


def test_built_filters_evaluate_like_the_engine() -> None:
    company = {"workforce": {"total": 25}, "headquarters": {"country": "Singapore"}}
    spec = Filters(employees=Range(lte=30), country=In(["SG"]))

    assert evaluate_filters(spec.items, company) == [Verdict.PASS, Verdict.PASS]


def test_equality_and_repr() -> None:
    assert Filters(employees=Range(lte=30)) == Filters(employees=Range(lte=30))
    assert Filters(employees=Range(lte=30)) != Filters(employees=Range(lte=31))
    assert hash(Filters(country=Eq("DE"))) == hash(Filters(country=Eq("DE")))
    assert repr(Filters(country=Eq("DE"))) == "Filters(CountryFilter(countries=('DE',)))"


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Range(), "at least one"),
        (lambda: Range(eq=5, lte=10), "cannot be combined"),
        (lambda: Filters(employees=Range(lte="30")), "must be numbers"),  # type: ignore[arg-type]
        (lambda: Filters(employees=Range(lte=True)), "must be numbers"),
        (lambda: Filters(employees=Range(lte=date(2024, 1, 1))), "must be numbers"),
        (lambda: Filters(funding_date=Range(gte="2024-01-01")), "datetime.date"),  # type: ignore[arg-type]
        (lambda: Filters(funding_date=Range(gte=datetime(2024, 1, 1))), "datetime.date"),
        (lambda: Filters(funding_date=Range(eq=2024)), "does not support eq"),
        (lambda: Filters(funding_date=Range(gte=date(2024, 1, 1), tolerance=0.1)), "tolerance"),
        (lambda: Filters(employees=Range(lte=30, tolerance=1.5)), "tolerance"),
        (lambda: Filters(country=In([])), "countries"),
        (lambda: Filters(funding_stage=Contains(" ")), "stage"),
    ],
)
def test_invalid_specs_are_rejected(build: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()  # type: ignore[operator]
