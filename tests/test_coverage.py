"""Tests for measuring typed-field coverage in Exa company results."""

import math
from collections import Counter
from datetime import date

import pytest

from exa_bench.coverage import measure_coverage
from exa_filters.constraints import (
    DateConstraint,
    DateField,
    DateOp,
    NumberConstraint,
    NumberField,
    NumberOp,
    TextConstraint,
    TextField,
    TextOp,
    Verdict,
    evaluate,
)

FULL: dict[str, object] = {
    "foundedYear": 2019,
    "workforce": {"total": 120},
    "headquarters": {"city": "Berlin", "country": "Germany"},
    "financials": {
        "fundingTotal": 25_000_000,
        "fundingLatestRound": {"name": "Series B", "date": "2024-03-01"},
    },
}
PARTIAL: dict[str, object] = {
    "foundedYear": 2021,
    "workforce": None,
    "headquarters": {"country": "US"},
    "financials": {"fundingTotal": None, "fundingLatestRound": {"name": "Seed", "date": None}},
}
MALFORMED: dict[str, object] = {
    "foundedYear": True,
    "workforce": {"total": "120"},
    "headquarters": {"country": " "},
    "financials": {"fundingTotal": math.nan, "fundingLatestRound": {"name": 7, "date": " "}},
}


def company(properties: dict[str, object]) -> dict[str, object]:
    entity = {"type": "company", "properties": properties}
    return {"url": "https://example.test", "entities": [entity]}


RESPONSES: list[dict[str, object]] = [
    {
        "results": [
            company(FULL),
            company(PARTIAL),
            {"url": "https://news.example.test"},  # no entities, e.g. a news article
            {"entities": [{"type": "person", "properties": {"name": "Ada"}}]},
            {"entities": "not-a-list"},
            "not-a-result",
        ]
    },
    {"results": [company(FULL)]},
    {},  # a response without results
]


def test_counts_results_and_company_entities() -> None:
    report = measure_coverage(RESPONSES)

    assert report.results == 7
    assert report.with_company == 3


def test_counts_filled_fields() -> None:
    report = measure_coverage(RESPONSES)

    assert report.filled == Counter(
        {
            "founded_year": 3,
            "employees": 2,
            "funding": 2,
            "country": 3,
            "funding_stage": 3,
            "funding_date": 2,
        }
    )


def test_fill_rate_uses_all_results_as_denominator() -> None:
    assert measure_coverage(RESPONSES).fill_rate("employees") == pytest.approx(2 / 7)
    assert measure_coverage([]).fill_rate("employees") == 0.0


def test_records_raw_value_samples() -> None:
    report = measure_coverage(RESPONSES)

    assert report.samples["country"] == Counter({"Germany": 2, "US": 1})
    assert report.samples["funding_stage"] == Counter({"Series B": 2, "Seed": 1})
    assert report.samples["funding_date"] == Counter({"2024-03-01": 2})


def test_malformed_values_are_sampled_but_never_filled() -> None:
    report = measure_coverage([{"results": [company(MALFORMED)]}])

    assert report.samples == {
        "country": Counter({" ": 1}),
        "funding_stage": Counter({"7": 1}),
        "funding_date": Counter({" ": 1}),
    }
    assert sum(report.filled.values()) == 0


@pytest.mark.parametrize("properties", [FULL, PARTIAL, MALFORMED])
def test_filled_matches_what_the_evaluator_can_use(properties: dict[str, object]) -> None:
    filled = measure_coverage([{"results": [company(properties)]}]).filled

    for number_field in NumberField:
        constraint = NumberConstraint(number_field, NumberOp.GTE, 0)
        usable = evaluate(constraint, properties) != Verdict.UNKNOWN
        assert (filled[number_field.value] == 1) == usable
    for text_field in TextField:
        text_constraint = TextConstraint(text_field, TextOp.CONTAINS, "")
        usable = evaluate(text_constraint, properties) != Verdict.UNKNOWN
        assert (filled[text_field.value] == 1) == usable
    for date_field in DateField:
        date_constraint = DateConstraint(date_field, DateOp.GTE, date(1900, 1, 1))
        usable = evaluate(date_constraint, properties) != Verdict.UNKNOWN
        assert (filled[date_field.value] == 1) == usable
