"""Tests for evaluating constraints against Exa company entity properties."""

import math

import pytest

from exa_filters.constraints import (
    Constraint,
    NumberConstraint,
    NumberField,
    NumberOp,
    TextConstraint,
    TextField,
    TextOp,
    Verdict,
    evaluate,
    lookup,
    usable_number,
    usable_text,
)

# Shaped like `results[].entities[].properties` for a company in Exa's /search response.
ACME: dict[str, object] = {
    "name": "Acme",
    "foundedYear": 2019,
    "workforce": {"total": 120},
    "headquarters": {"address": None, "city": "Berlin", "postalCode": None, "country": "Germany"},
    "financials": {
        "revenueAnnual": None,
        "fundingTotal": 25_000_000,
        "fundingLatestRound": {"name": "Series B", "date": "2024-03-01", "amount": 20_000_000},
    },
}


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        (NumberConstraint(NumberField.FOUNDED_YEAR, NumberOp.EQ, 2019), Verdict.PASS),
        (NumberConstraint(NumberField.FOUNDED_YEAR, NumberOp.EQ, 2020), Verdict.FAIL),
        (NumberConstraint(NumberField.FOUNDED_YEAR, NumberOp.EQ, 2018), Verdict.FAIL),
        # Bounds are inclusive.
        (NumberConstraint(NumberField.EMPLOYEES, NumberOp.GTE, 120), Verdict.PASS),
        (NumberConstraint(NumberField.EMPLOYEES, NumberOp.GTE, 121), Verdict.FAIL),
        (NumberConstraint(NumberField.EMPLOYEES, NumberOp.LTE, 120), Verdict.PASS),
        (NumberConstraint(NumberField.EMPLOYEES, NumberOp.LTE, 119), Verdict.FAIL),
        (NumberConstraint(NumberField.FUNDING, NumberOp.LTE, 30_000_000), Verdict.PASS),
        (NumberConstraint(NumberField.FUNDING, NumberOp.GTE, 30_000_000), Verdict.FAIL),
        # Text matching ignores case and surrounding whitespace.
        (TextConstraint(TextField.COUNTRY, TextOp.EQ, " germany "), Verdict.PASS),
        (TextConstraint(TextField.COUNTRY, TextOp.EQ, "France"), Verdict.FAIL),
        (TextConstraint(TextField.FUNDING_STAGE, TextOp.CONTAINS, "series b"), Verdict.PASS),
        (TextConstraint(TextField.FUNDING_STAGE, TextOp.CONTAINS, "series"), Verdict.PASS),
        (TextConstraint(TextField.FUNDING_STAGE, TextOp.CONTAINS, "Seed"), Verdict.FAIL),
    ],
)
def test_evaluate_known_values(constraint: Constraint, expected: Verdict) -> None:
    assert evaluate(constraint, ACME) == expected


@pytest.mark.parametrize(
    "properties",
    [
        {},  # field absent
        {"workforce": None},  # parent object is null
        {"workforce": {"total": None}},  # value is null
        {"workforce": {"total": "120"}},  # wrong type
        {"workforce": {"total": True}},  # a bool is not a count
        {"workforce": {"total": math.nan}},
    ],
)
def test_missing_or_malformed_number_is_unknown(properties: dict[str, object]) -> None:
    constraint = NumberConstraint(NumberField.EMPLOYEES, NumberOp.GTE, 10)
    assert evaluate(constraint, properties) == Verdict.UNKNOWN


@pytest.mark.parametrize("country", [None, "", "   ", 49])
def test_missing_or_blank_text_is_unknown(country: object) -> None:
    constraint = TextConstraint(TextField.COUNTRY, TextOp.EQ, "Germany")
    assert evaluate(constraint, {"headquarters": {"country": country}}) == Verdict.UNKNOWN


def test_usable_values_are_returned_unchanged() -> None:
    assert usable_number(NumberField.EMPLOYEES, ACME) == 120
    assert usable_text(TextField.FUNDING_STAGE, ACME) == "Series B"


def test_unusable_values_are_none() -> None:
    assert usable_number(NumberField.EMPLOYEES, {"workforce": {"total": "120"}}) is None
    assert usable_text(TextField.COUNTRY, {"headquarters": {"country": "  "}}) is None


def test_lookup_follows_nested_path_and_stops_at_non_objects() -> None:
    assert lookup(ACME, ("headquarters", "city")) == "Berlin"
    assert lookup({"headquarters": "Berlin"}, ("headquarters", "city")) is None
    assert lookup(ACME, ("missing", "city")) is None


@pytest.mark.parametrize("value", [True, math.nan, math.inf])
def test_number_constraint_rejects_bool_and_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        NumberConstraint(NumberField.EMPLOYEES, NumberOp.GTE, value)
