"""Typed filters over Exa company entities, and their evaluation: the one shared engine.

Both the benchmark grader (translating benchmark constraint syntax) and the filters feature
(taking user filters) build these objects and call `evaluate_filter`, so a company is judged
by identical logic in both places.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from exa_filters.constraints import (
    DateConstraint,
    DateField,
    DateOp,
    NumberConstraint,
    NumberField,
    NumberOp,
    TextField,
    Verdict,
    evaluate,
    usable_text,
)
from exa_filters.country import same_country
from exa_filters.funding_stage import is_stage, stage_matches


@dataclass(frozen=True)
class NumberFilter:
    """A numeric bound. `tolerance` widens gte/lte bounds by that fraction (0.2 = 20%)."""

    field: NumberField
    op: NumberOp
    value: float
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError(f"tolerance must be in [0, 1), got {self.tolerance!r}")
        NumberConstraint(self.field, self.op, self.value)  # validates the value


@dataclass(frozen=True)
class CountryFilter:
    """Headquarters in any of these countries (names, ISO codes, or common aliases)."""

    countries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.countries or not all(c.strip() for c in self.countries):
            raise ValueError("countries must be a non-empty tuple of non-empty strings")


@dataclass(frozen=True)
class StageFilter:
    """Latest funding round contains this stage name, e.g. "Series B" or "Seed"."""

    stage: str

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must be a non-empty string")


@dataclass(frozen=True)
class DateFilter:
    field: DateField
    op: DateOp
    value: date

    def __post_init__(self) -> None:
        DateConstraint(self.field, self.op, self.value)  # validates the value


type Filter = NumberFilter | CountryFilter | StageFilter | DateFilter


def evaluate_filter(item: Filter, properties: Mapping[str, object]) -> Verdict:
    """PASS or FAIL against a company's typed fields; UNKNOWN when the field cannot decide."""
    match item:
        case NumberFilter():
            bound = item.value
            if item.op is NumberOp.LTE:
                bound *= 1 + item.tolerance
            elif item.op is NumberOp.GTE:
                bound *= 1 - item.tolerance
            return evaluate(NumberConstraint(item.field, item.op, bound), properties)
        case CountryFilter():
            actual = usable_text(TextField.COUNTRY, properties)
            if actual is None:
                return Verdict.UNKNOWN
            matched = any(same_country(actual, wanted) for wanted in item.countries)
            return Verdict.PASS if matched else Verdict.FAIL
        case StageFilter():
            actual = usable_text(TextField.FUNDING_STAGE, properties)
            if actual is None:
                return Verdict.UNKNOWN
            if is_stage(item.stage) and not is_stage(actual):
                return Verdict.UNKNOWN  # "Venture", grants, debt: no evidence about the stage
            return Verdict.PASS if stage_matches(actual, item.stage) else Verdict.FAIL
        case DateFilter():
            return evaluate(DateConstraint(item.field, item.op, item.value), properties)


def evaluate_filters(items: Sequence[Filter], properties: Mapping[str, object]) -> list[Verdict]:
    return [evaluate_filter(item, properties) for item in items]
