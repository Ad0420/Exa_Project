"""Tests for fact extraction prompts and caching (no network)."""

from pathlib import Path

import pytest

from exa_bench.crosscheck.fact_extraction import (
    EXTRACTION_SYSTEM,
    Extraction,
    PageFacts,
    build_extraction_prompt,
    cached_extraction,
)
from exa_bench.crosscheck.llm_grader import MAX_TEXT_CHARS

FACTS = PageFacts(
    founded_year=2019,
    employees=120,
    employees_range=None,
    hq_country="Germany",
    funding_total_usd=25_000_000,
    latest_round_name="Series B",
    latest_round_date="2024-03-01",
    is_single_company_page=True,
)


def test_prompt_includes_url_and_truncated_text() -> None:
    system, user = build_extraction_prompt("https://a.test", "x" * (MAX_TEXT_CHARS + 10))

    assert system == EXTRACTION_SYSTEM
    assert user.startswith("Page URL: https://a.test\n\n")
    assert user.endswith("x" * MAX_TEXT_CHARS)
    assert len(user) == len("Page URL: https://a.test\n\n") + MAX_TEXT_CHARS


@pytest.mark.parametrize("text", [None, ""])
def test_missing_text_becomes_no_content(text: str | None) -> None:
    assert build_extraction_prompt("https://a.test", text)[1].endswith("(no content)")


def test_schema_allows_nulls_everywhere_except_the_page_kind() -> None:
    facts = PageFacts.model_validate(
        {
            "founded_year": None,
            "employees": None,
            "employees_range": None,
            "hq_country": None,
            "funding_total_usd": None,
            "latest_round_name": None,
            "latest_round_date": None,
            "is_single_company_page": False,
        }
    )

    assert facts.employees is None
    assert facts.is_single_company_page is False


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, user: str) -> Extraction:
        self.calls += 1
        return Extraction(FACTS, 900, 40)


def test_extraction_is_cached_per_prompt(tmp_path: Path) -> None:
    extractor = FakeExtractor()

    first = cached_extraction(tmp_path, extractor, "sys", "page one")
    again = cached_extraction(tmp_path, extractor, "sys", "page one")
    other = cached_extraction(tmp_path, extractor, "sys", "page two")

    assert (first.from_cache, again.from_cache, other.from_cache) == (False, True, False)
    assert again.extraction == first.extraction == Extraction(FACTS, 900, 40)
    assert extractor.calls == 2
