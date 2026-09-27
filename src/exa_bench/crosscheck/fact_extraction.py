"""Extract the five typed facts from a page's text, so they can be compared with Exa's fields.

This is our own prompt (not Exa's). It asks only for facts the page states explicitly and
uses null otherwise, so "the page does not say" stays distinct from "the page disagrees".
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from exa_bench.core.json_cache import cache_key, cached_record
from exa_bench.crosscheck.llm_grader import EXA_GRADER_MODEL, EXA_GRADER_TEMPERATURE, MAX_TEXT_CHARS

EXTRACTION_SYSTEM = """You extract facts about a company from the text of one web page.

Rules:
- Report only what the text states explicitly. Never infer, estimate, or use outside knowledge.
- Use null for anything the text does not state.
- founded_year: the year the company was founded, as an integer.
- employees: the total number of employees, as an integer, only if the text gives one number.
- employees_range: if the text gives headcount as a range (e.g. "51-200"), the range verbatim.
- hq_country: the country where the company is headquartered (not every country it operates in).
- funding_total_usd: total funding raised, in US dollars as a number (e.g. "$25M" -> 25000000).
  Null if the total is not stated or is not in US dollars.
- latest_round_name: the most recent funding round's name as written (e.g. "Seed", "Series B").
- latest_round_date: when that round happened, as YYYY-MM-DD, YYYY-MM, or YYYY as stated.
- is_single_company_page: true if the page is about one company (its own site or profile),
  false if it is a news article, a list, a job posting, or something else."""

EXTRACTION_USER = """Page URL: {url}

{text}"""


class PageFacts(BaseModel):
    founded_year: int | None
    employees: int | None
    employees_range: str | None
    hq_country: str | None
    funding_total_usd: float | None
    latest_round_name: str | None
    latest_round_date: str | None
    is_single_company_page: bool


@dataclass(frozen=True)
class Extraction:
    facts: PageFacts
    prompt_tokens: int | None
    completion_tokens: int | None


class Extractor(Protocol):
    """Runs one extraction prompt through a model. Implemented with OpenAI and by fakes."""

    def __call__(self, system: str, user: str) -> Extraction: ...


def build_extraction_prompt(url: str, text: str | None) -> tuple[str, str]:
    body = text[:MAX_TEXT_CHARS] if text else "(no content)"
    return EXTRACTION_SYSTEM, EXTRACTION_USER.format(url=url, text=body)


@dataclass(frozen=True)
class CachedExtraction:
    extraction: Extraction
    from_cache: bool


def cached_extraction(
    cache_dir: Path,
    extractor: Extractor,
    system: str,
    user: str,
    *,
    model: str = EXA_GRADER_MODEL,
    temperature: float = EXA_GRADER_TEMPERATURE,
) -> CachedExtraction:
    key = cache_key({"model": model, "temperature": temperature, "system": system, "user": user})

    def compute() -> dict[str, object]:
        extraction = extractor(system, user)
        return {
            "model": model,
            "temperature": temperature,
            "facts": extraction.facts.model_dump(),
            "prompt_tokens": extraction.prompt_tokens,
            "completion_tokens": extraction.completion_tokens,
        }

    record, from_cache = cached_record(cache_dir, key, compute)
    facts = record["facts"]
    assert isinstance(facts, Mapping)
    extraction = Extraction(
        facts=PageFacts.model_validate(dict(facts)),
        prompt_tokens=_int_or_none(record.get("prompt_tokens")),
        completion_tokens=_int_or_none(record.get("completion_tokens")),
    )
    return CachedExtraction(extraction, from_cache)


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
