"""Compare Exa's typed values with the facts a page states, field by field."""

import math
import re
from collections.abc import Mapping
from enum import StrEnum

from exa_bench.crosscheck.fact_extraction import PageFacts
from exa_filters.country import same_country
from exa_filters.funding_stage import is_stage, stage_matches

RELATIVE_TOLERANCE = 0.2  # Exa's own headcount tolerance; applied to funding totals too
FIELDS = ("founded_year", "employees", "country", "funding", "funding_stage", "funding_date")

_DASHES = "-" + chr(0x2013) + chr(0x2014)  # hyphen, en dash, em dash
_RANGE = re.compile(rf"^\s*(\d[\d,]*)\s*(?:[{_DASHES}]|to)\s*(\d[\d,]*)\s*$")
_OPEN_RANGE = re.compile(r"^\s*(\d[\d,]*)\s*\+\s*$")
_ISO_PREFIX = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


class FieldCheck(StrEnum):
    AGREES = "agrees"
    DISAGREES = "disagrees"
    NOT_STATED = "not_stated"  # the page does not state this fact
    NO_TYPED_VALUE = "no_typed_value"  # Exa has no usable (or no informative) value


def compare_all(typed: Mapping[str, object], facts: PageFacts | None) -> dict[str, FieldCheck]:
    return {key: compare_field(key, typed.get(key), facts) for key in FIELDS}


def compare_field(key: str, typed: object, facts: PageFacts | None) -> FieldCheck:
    if typed is None or (key == "funding_stage" and isinstance(typed, str) and not is_stage(typed)):
        return FieldCheck.NO_TYPED_VALUE
    if facts is None:
        return FieldCheck.NOT_STATED
    match key:
        case "founded_year":
            if facts.founded_year is None:
                return FieldCheck.NOT_STATED
            return _verdict(_number(typed) == facts.founded_year)
        case "employees":
            return _compare_employees(_number(typed), facts)
        case "country":
            if facts.hq_country is None:
                return FieldCheck.NOT_STATED
            return _verdict(same_country(str(typed), facts.hq_country))
        case "funding":
            if facts.funding_total_usd is None:
                return FieldCheck.NOT_STATED
            return _verdict(_within_tolerance(_number(typed), facts.funding_total_usd))
        case "funding_stage":
            if facts.latest_round_name is None:
                return FieldCheck.NOT_STATED
            return _verdict(stage_matches(facts.latest_round_name, str(typed)))
        case "funding_date":
            return _compare_date(str(typed), facts.latest_round_date)
        case _:
            raise ValueError(f"unknown field {key!r}")


def page_properties(facts: PageFacts | None) -> dict[str, object]:
    """A company record built from page facts, so the grader can judge the page's version."""
    if facts is None:
        return {}
    date = _iso_date(facts.latest_round_date)
    return {
        "foundedYear": facts.founded_year,
        "workforce": {"total": facts.employees},
        "headquarters": {"country": facts.hq_country},
        "financials": {
            "fundingTotal": facts.funding_total_usd,
            "fundingLatestRound": {"name": facts.latest_round_name, "date": date},
        },
    }


def _compare_employees(typed: float, facts: PageFacts) -> FieldCheck:
    if facts.employees is not None:
        return _verdict(_within_tolerance(typed, facts.employees))
    if facts.employees_range is not None:
        bounds = _parse_range(facts.employees_range)
        if bounds is not None:
            low, high = bounds
            return _verdict(low <= typed <= high)
    return FieldCheck.NOT_STATED


def _compare_date(typed_iso: str, page: str | None) -> FieldCheck:
    if page is None:
        return FieldCheck.NOT_STATED
    parts = _ISO_PREFIX.match(page.strip())
    if parts is None:
        return FieldCheck.NOT_STATED
    precision = 4 if parts.group(2) is None else 7  # compare by year, or by year and month
    return _verdict(typed_iso[:precision] == page.strip()[:precision])


def _iso_date(page: str | None) -> str | None:
    if page is None:
        return None
    parts = _ISO_PREFIX.match(page.strip())
    if parts is None:
        return None
    year, month, day = parts.group(1), parts.group(2) or "01", parts.group(3) or "01"
    return f"{year}-{month}-{day}"


def _parse_range(text: str) -> tuple[float, float] | None:
    if (closed := _RANGE.match(text)) is not None:
        low, high = (float(part.replace(",", "")) for part in closed.groups())
        return (low, high) if low <= high else None
    if (open_ended := _OPEN_RANGE.match(text)) is not None:
        return float(open_ended.group(1).replace(",", "")), math.inf
    return None


def _within_tolerance(typed: float, page: float) -> bool:
    return abs(typed - page) <= RELATIVE_TOLERANCE * abs(page)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a typed number, got {value!r}")
    return float(value)


def _verdict(agrees: bool) -> FieldCheck:
    return FieldCheck.AGREES if agrees else FieldCheck.DISAGREES
