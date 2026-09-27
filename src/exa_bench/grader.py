"""Grade one Exa company result against a benchmark query's constraints, using typed fields."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from exa_filters.constraints import DateField, DateOp, NumberField, NumberOp
from exa_filters.evaluate import (
    CountryFilter,
    DateFilter,
    Filter,
    NumberFilter,
    StageFilter,
    evaluate_filter,
)


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"  # the result has no usable value for the constrained field
    NOT_CHECKABLE = "not_checkable"  # no typed field covers this constraint


class ResultVerdict(StrEnum):
    VIOLATES = "violates"  # at least one checkable constraint fails
    SATISFIES = "satisfies"  # every checkable constraint passes
    UNEVALUABLE = "unevaluable"  # nothing fails, but a needed field is missing
    NOT_CHECKABLE = "not_checkable"  # the query has no checkable constraint


@dataclass(frozen=True)
class ConstraintOutcome:
    key: str  # benchmark constraint key, e.g. "employees"
    op: str  # benchmark operator, e.g. "lte"; "" when the constraint is not an op mapping
    outcome: Outcome


@dataclass(frozen=True)
class GradedResult:
    verdict: ResultVerdict
    outcomes: tuple[ConstraintOutcome, ...]


NUMBER_FIELDS = {
    "founded_year": NumberField.FOUNDED_YEAR,
    "employees": NumberField.EMPLOYEES,
    "funding": NumberField.FUNDING,
}
NUMBER_OPS = {"eq": NumberOp.EQ, "gte": NumberOp.GTE, "lte": NumberOp.LTE}
DATE_OPS = {"gte": DateOp.GTE, "lte": DateOp.LTE}


def grade(
    constraints: Mapping[str, object],
    properties: Mapping[str, object],
    *,
    employee_tolerance: float = 0.0,
) -> GradedResult:
    """Check every benchmark constraint against a company's typed fields.

    `employee_tolerance` widens headcount bounds by that fraction (0.2 = Exa's own
    grader's "within 20%"), so lte 100 accepts up to 120 and gte 100 accepts 80.
    """
    outcomes = [
        ConstraintOutcome(key, op, _outcome(item, properties))
        for key, op, item in _translate(constraints, employee_tolerance)
    ]
    return GradedResult(_verdict(outcomes), tuple(outcomes))


def filters_from_constraints(
    constraints: Mapping[str, object], *, employee_tolerance: float = 0.0
) -> list[Filter]:
    """The typed filters for a query's checkable constraints, in the benchmark's order.

    Built by the same translation `grade` uses, so the filters feature accepts exactly the
    results the grader marks as satisfying.
    """
    return [item for _, _, item in _translate(constraints, employee_tolerance) if item is not None]


def checkable_constraints(constraints: Mapping[str, object]) -> int:
    """How many of a query's constraints the typed fields can check at all."""
    outcomes = grade(constraints, {}).outcomes
    return sum(outcome.outcome is not Outcome.NOT_CHECKABLE for outcome in outcomes)


def _verdict(outcomes: list[ConstraintOutcome]) -> ResultVerdict:
    seen = {item.outcome for item in outcomes}
    if Outcome.FAIL in seen:
        return ResultVerdict.VIOLATES
    if Outcome.UNKNOWN in seen:
        return ResultVerdict.UNEVALUABLE
    if Outcome.PASS in seen:
        return ResultVerdict.SATISFIES
    return ResultVerdict.NOT_CHECKABLE


def _translate(
    constraints: Mapping[str, object], employee_tolerance: float
) -> list[tuple[str, str, Filter | None]]:
    """Every constraint as (key, op, filter); the filter is None when no typed field covers it."""
    translated: list[tuple[str, str, Filter | None]] = []
    for key, spec in constraints.items():
        if not isinstance(spec, Mapping):
            translated.append((key, "", None))
            continue
        for op, value in spec.items():
            translated.append((key, op, filter_from_benchmark(key, op, value, employee_tolerance)))
    return translated


def _outcome(item: Filter | None, properties: Mapping[str, object]) -> Outcome:
    if item is None:
        return Outcome.NOT_CHECKABLE
    return Outcome(evaluate_filter(item, properties).value)


def filter_from_benchmark(
    key: str, op: str, value: object, employee_tolerance: float = 0.0
) -> Filter | None:
    """Translate one benchmark constraint into a typed filter; None when no typed field covers it.

    Raises ValueError for a checkable constraint whose value is malformed, since the benchmark
    file is pinned and such a value would be a bug worth knowing about.
    """
    if key in NUMBER_FIELDS and op in NUMBER_OPS:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key} {op} needs a number, got {value!r}")
        tolerance = employee_tolerance if key == "employees" else 0.0
        return NumberFilter(NUMBER_FIELDS[key], NUMBER_OPS[op], float(value), tolerance)
    if key == "country" and op == "eq":
        if not isinstance(value, str):
            raise ValueError(f"country eq needs a string, got {value!r}")
        return CountryFilter((value,))
    if key == "country" and op == "in":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"country in needs a list of strings, got {value!r}")
        return CountryFilter(tuple(value))
    if key == "funding_stage" and op == "contains":
        if not isinstance(value, str):
            raise ValueError(f"funding_stage contains needs a string, got {value!r}")
        return StageFilter(value)
    if key == "funding_date" and op in DATE_OPS:
        if not isinstance(value, str):
            raise ValueError(f"funding_date {op} needs an ISO date string, got {value!r}")
        try:
            bound = date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"funding_date {op} needs an ISO date string, got {value!r}") from None
        return DateFilter(DateField.FUNDING_DATE, DATE_OPS[op], bound)
    return None
