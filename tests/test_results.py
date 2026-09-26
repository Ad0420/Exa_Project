"""Tests for the result pipeline: parsing, verdicts, exclusions, dedup, null policy, selection."""

import pytest

from exa_filters.constraints import NumberField, NumberOp, Verdict
from exa_filters.evaluate import CountryFilter, Filter, NumberFilter
from exa_filters.response import CompanyEntity, company_entity
from exa_filters.results import (
    NullPolicy,
    Status,
    canonical_host,
    dedupe,
    exclude,
    parse_results,
    select,
)

SMALL = NumberFilter(NumberField.EMPLOYEES, NumberOp.LTE, 100)
GERMANY = CountryFilter(("Germany",))
FILTERS: list[Filter] = [SMALL, GERMANY]


def result(
    url: str, employees: int | None, country: str | None = "Germany", entity_id: str | None = "e1"
) -> dict[str, object]:
    properties = {"workforce": {"total": employees}, "headquarters": {"country": country}}
    entity: dict[str, object] = {"type": "company", "properties": properties}
    if entity_id is not None:
        entity["id"] = entity_id
    return {"url": url, "title": f"T {url}", "entities": [entity]}


def test_company_entity_reads_id_and_properties() -> None:
    assert company_entity(result("https://a.test", 5, entity_id="abc")) == CompanyEntity(
        "abc", {"workforce": {"total": 5}, "headquarters": {"country": "Germany"}}
    )
    assert company_entity(result("https://a.test", 5, entity_id=None)) is not None
    assert company_entity({"url": "https://news.test"}) is None
    assert company_entity("junk") is None
    person = {
        "url": "https://p.test",
        "entities": [{"type": "person", "properties": {"name": "A"}}],
    }
    assert company_entity(person) is None
    numeric_id = {
        "url": "https://n.test",
        "entities": [{"type": "company", "id": 7, "properties": {}}],
    }
    assert company_entity(numeric_id) == CompanyEntity(None, {})


def test_parse_results_judges_each_result_in_rank_order() -> None:
    body: dict[str, object] = {
        "results": [
            result("https://a.test", 50, entity_id="a"),
            result("https://b.test", 500, entity_id="b"),
            result("https://c.test", None, entity_id="c"),
            {"url": "https://news.test", "title": "no entity"},
            {"title": "no url"},
        ]
    }

    parsed = parse_results(body, FILTERS)

    assert [(r.rank, r.url, r.entity_id) for r in parsed] == [
        (1, "https://a.test", "a"),
        (2, "https://b.test", "b"),
        (3, "https://c.test", "c"),
        (4, "https://news.test", None),
    ]
    assert [r.status for r in parsed] == [
        Status.SATISFIES,
        Status.VIOLATES,
        Status.UNEVALUABLE,
        Status.UNEVALUABLE,
    ]
    assert parsed[1].verdicts == (Verdict.FAIL, Verdict.PASS)
    assert parsed[3].unknown_fields == 2
    assert parsed[3].title == "no entity"


@pytest.mark.parametrize(
    ("text", "host"),
    [
        ("acme.com", "acme.com"),
        ("https://www.acme.com/about?x=1", "acme.com"),
        ("HTTP://ACME.COM", "acme.com"),
        ("www.acme.co.uk", "acme.co.uk"),
        ("  ", None),
        ("https://", None),
    ],
)
def test_canonical_host(text: str, host: str | None) -> None:
    assert canonical_host(text) == host


def test_exclude_by_entity_id_or_domain_in_any_spelling() -> None:
    parsed = parse_results(
        {
            "results": [
                result("https://www.acme.com/", 10, entity_id="acme-id"),
                result("https://beta.test/", 10, entity_id="beta-id"),
                result("https://gamma.test/", 10, entity_id="gamma-id"),
            ]
        },
        FILTERS,
    )

    kept, dropped = exclude(parsed, ["acme-id", "https://Gamma.test/x"])
    assert [r.entity_id for r in kept] == ["beta-id"]
    assert dropped == 2

    kept, dropped = exclude(parsed, ["acme.com", "  "])
    assert [r.entity_id for r in kept] == ["beta-id", "gamma-id"]
    assert dropped == 1


def test_dedupe_by_entity_id_then_host() -> None:
    parsed = parse_results(
        {
            "results": [
                result("https://acme.com/", 10, entity_id="acme"),
                result("https://acme.com/about", 10, entity_id="acme"),  # same entity
                result("https://www.beta.test/", 10, entity_id=None),
                result("https://beta.test/careers", 10, entity_id=None),  # same host, no ids
                result("https://beta.test/other", 10, entity_id="beta-id"),  # id wins over host
            ]
        },
        FILTERS,
    )

    kept, dropped = dedupe(parsed)

    assert [r.url for r in kept] == [
        "https://acme.com/",
        "https://www.beta.test/",
        "https://beta.test/other",
    ]
    assert dropped == 2


def test_select_applies_the_whole_pipeline_in_rank_order() -> None:
    body: dict[str, object] = {
        "results": [
            result("https://a.test", 50, entity_id="a"),  # accepted
            result("https://x.test", 50, entity_id="x"),  # excluded
            result("https://b.test", 500, entity_id="b"),  # violates
            result("https://c.test", None, entity_id="c"),  # unevaluable
            result("https://a.test/2", 50, entity_id="a"),  # duplicate of a
            result("https://d.test", 60, entity_id="d"),  # accepted
            result("https://e.test", 70, entity_id="e"),  # accepted, beyond k
        ]
    }

    strict = select(body, FILTERS, k=2, exclude_entities=["x"])
    assert [r.entity_id for r in strict.results] == ["a", "d"]
    assert (strict.seen, strict.excluded, strict.duplicates) == (7, 1, 1)
    assert (strict.violating, strict.unevaluable, strict.short_by) == (1, 1, 0)

    lenient = select(body, FILTERS, k=10, null_policy=NullPolicy.LENIENT, exclude_entities=["x"])
    assert [r.entity_id for r in lenient.results] == ["a", "c", "d", "e"]
    assert lenient.short_by == 6


def test_select_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        select({"results": []}, FILTERS, k=0)
