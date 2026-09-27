"""Tests for comparing typed values with page facts."""

import pytest

from exa_bench.constraints.grader import Outcome, grade
from exa_bench.crosscheck.fact_extraction import PageFacts
from exa_bench.crosscheck.field_check import (
    FIELDS,
    FieldCheck,
    compare_all,
    compare_field,
    page_properties,
    page_typed_fields,
)

A, D, NS, NT = (
    FieldCheck.AGREES,
    FieldCheck.DISAGREES,
    FieldCheck.NOT_STATED,
    FieldCheck.NO_TYPED_VALUE,
)


def facts(**overrides: object) -> PageFacts:
    base: dict[str, object] = {
        "founded_year": None,
        "employees": None,
        "employees_range": None,
        "hq_country": None,
        "funding_total_usd": None,
        "latest_round_name": None,
        "latest_round_date": None,
        "is_single_company_page": True,
    }
    return PageFacts.model_validate(base | overrides)


@pytest.mark.parametrize(
    ("key", "typed", "page", "expected"),
    [
        ("founded_year", 2019, facts(founded_year=2019), A),
        ("founded_year", 2019, facts(founded_year=2018), D),
        ("founded_year", 2019, facts(), NS),
        # Exact headcount: within 20% of the page's number, inclusive.
        ("employees", 120, facts(employees=100), A),
        ("employees", 121, facts(employees=100), D),
        ("employees", 80, facts(employees=100), A),
        ("employees", 79, facts(employees=100), D),
        # Range headcount: inside the range, inclusive; open-ended "1,000+".
        ("employees", 51, facts(employees_range="51-200"), A),
        ("employees", 200, facts(employees_range="51" + chr(0x2013) + "200"), A),  # en dash
        ("employees", 201, facts(employees_range="51 to 200"), D),
        ("employees", 1500, facts(employees_range="1,000+"), A),
        ("employees", 999, facts(employees_range="1,000+"), D),
        ("employees", 50, facts(employees_range="lots"), NS),
        ("employees", 50, facts(employees_range="200-51"), NS),  # inverted range is unusable
        ("employees", 50, facts(), NS),
        ("country", "DE", facts(hq_country="Germany"), A),
        ("country", "Germany", facts(hq_country="Austria"), D),
        ("country", "Germany", facts(), NS),
        ("funding", 25_839_097, facts(funding_total_usd=25_000_000), A),
        ("funding", 31_000_000, facts(funding_total_usd=25_000_000), D),
        ("funding", 25_000_000, facts(), NS),
        # Stages match as whole tokens in either direction; pre-seed is not seed.
        ("funding_stage", "Series b", facts(latest_round_name="Series B"), A),
        ("funding_stage", "Seed", facts(latest_round_name="Seed round"), A),
        ("funding_stage", "Seed", facts(latest_round_name="Pre-seed"), D),
        ("funding_stage", "Series b", facts(latest_round_name="Series C"), D),
        ("funding_stage", "Series b", facts(), NS),
        ("funding_stage", "Venture", facts(latest_round_name="Series B"), NT),  # uninformative
        # Dates compare at the page's precision.
        ("funding_date", "2024-03-01", facts(latest_round_date="2024"), A),
        ("funding_date", "2024-03-01", facts(latest_round_date="2024-03"), A),
        ("funding_date", "2024-03-01", facts(latest_round_date="2024-03-15"), A),
        ("funding_date", "2024-03-01", facts(latest_round_date="2024-04"), D),
        ("funding_date", "2024-03-01", facts(latest_round_date="2023"), D),
        ("funding_date", "2024-03-01", facts(latest_round_date="March 2024"), NS),
        ("funding_date", "2024-03-01", facts(), NS),
    ],
)
def test_compare_field(key: str, typed: object, page: PageFacts, expected: FieldCheck) -> None:
    assert compare_field(key, typed, page) is expected


@pytest.mark.parametrize("key", FIELDS)
def test_missing_typed_value_or_missing_page(key: str) -> None:
    assert compare_field(key, None, facts(founded_year=2019)) is NT
    typed = {"country": "Germany", "funding_stage": "Seed", "funding_date": "2024-01-01"}.get(
        key, 1
    )
    assert compare_field(key, typed, None) is NS


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        compare_field("revenue", 1, facts())


def test_compare_all_covers_every_field() -> None:
    typed = {"founded_year": 2019, "employees": 120, "country": "Germany"}
    checks = compare_all(typed, facts(founded_year=2019, hq_country="France"))

    assert checks == {
        "founded_year": A,
        "employees": NS,
        "country": D,
        "funding": NT,
        "funding_stage": NT,
        "funding_date": NT,
    }


def test_page_properties_let_the_grader_judge_the_pages_version() -> None:
    page = facts(
        founded_year=2019,
        employees=80,
        hq_country="Germany",
        funding_total_usd=25_000_000,
        latest_round_name="Series B",
        latest_round_date="2024-03",
    )
    constraints: dict[str, object] = {
        "employees": {"lte": 100},
        "country": {"eq": "DE"},
        "funding_stage": {"contains": "Series B"},
        "funding_date": {"gte": "2024-01-01"},
    }

    outcomes = [o.outcome for o in grade(constraints, page_properties(page)).outcomes]

    assert outcomes == [Outcome.PASS] * 4
    assert page_typed_fields(page)["funding_date"] == "2024-03-01"
    assert page_typed_fields(facts(latest_round_date="2024"))["funding_date"] == "2024-01-01"
    assert page_properties(None) == {}
    assert page_typed_fields(facts(employees_range="51-200"))["employees"] is None
