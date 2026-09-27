"""Measure how often Exa company results carry the typed fields our constraints need."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from exa_filters.constraints import (
    FIELD_PATHS,
    DateField,
    NumberField,
    TextField,
    lookup,
    usable_date,
    usable_number,
    usable_text,
)
from exa_filters.response import company_properties, search_results

# Raw values to inspect so we learn Exa's formats (e.g. "US" vs "United States").
FORMAT_SAMPLE_PATHS: dict[str, tuple[str, ...]] = {
    "country": FIELD_PATHS[TextField.COUNTRY],
    "funding_stage": FIELD_PATHS[TextField.FUNDING_STAGE],
    "funding_date": FIELD_PATHS[DateField.FUNDING_DATE],
}


@dataclass
class CoverageReport:
    results: int = 0  # every element of every response's `results`
    with_company: int = 0  # results carrying a company entity with `properties`
    filled: Counter[str] = field(default_factory=Counter)
    samples: dict[str, Counter[str]] = field(
        default_factory=lambda: {name: Counter() for name in FORMAT_SAMPLE_PATHS}
    )

    def fill_rate(self, name: str) -> float:
        """Share of all results whose company entity has a usable value for `name`."""
        return self.filled[name] / self.results if self.results else 0.0


def measure_coverage(responses: Iterable[Mapping[str, object]]) -> CoverageReport:
    """Count usable company fields across raw Exa /search responses."""
    report = CoverageReport()
    for response in responses:
        for result in search_results(response):
            report.results += 1
            company = company_properties(result)
            if company is None:
                continue
            report.with_company += 1
            _count_filled(report, company)
            _record_samples(report, company)
    return report


def _count_filled(report: CoverageReport, company: Mapping[str, object]) -> None:
    for number_field in NumberField:
        if usable_number(number_field, company) is not None:
            report.filled[number_field.value] += 1
    for text_field in TextField:
        if usable_text(text_field, company) is not None:
            report.filled[text_field.value] += 1
    for date_field in DateField:
        if usable_date(date_field, company) is not None:
            report.filled[date_field.value] += 1


def _record_samples(report: CoverageReport, company: Mapping[str, object]) -> None:
    for name, path in FORMAT_SAMPLE_PATHS.items():
        value = lookup(company, path)
        if value is not None:
            report.samples[name][value if isinstance(value, str) else repr(value)] += 1
