"""Grade one Exa company result against a benchmark query's constraints, using typed fields."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

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
    usable_text,
)
from exa_filters.funding_stage import stage_matches


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
    outcomes: list[ConstraintOutcome] = []
    for key, spec in constraints.items():
        if not isinstance(spec, Mapping):
            outcomes.append(ConstraintOutcome(key, "", Outcome.NOT_CHECKABLE))
            continue
        for op, value in spec.items():
            outcome = _check(key, op, value, properties, employee_tolerance)
            outcomes.append(ConstraintOutcome(key, op, outcome))
    return GradedResult(_verdict(outcomes), tuple(outcomes))


def _verdict(outcomes: list[ConstraintOutcome]) -> ResultVerdict:
    seen = {item.outcome for item in outcomes}
    if Outcome.FAIL in seen:
        return ResultVerdict.VIOLATES
    if Outcome.UNKNOWN in seen:
        return ResultVerdict.UNEVALUABLE
    if Outcome.PASS in seen:
        return ResultVerdict.SATISFIES
    return ResultVerdict.NOT_CHECKABLE


def _check(
    key: str, op: str, value: object, properties: Mapping[str, object], employee_tolerance: float
) -> Outcome:
    if key in NUMBER_FIELDS and op in NUMBER_OPS:
        return _check_number(
            NUMBER_FIELDS[key], NUMBER_OPS[op], value, properties, employee_tolerance
        )
    if key == "country" and op == "eq":
        return _check_country(value, properties)
    if key == "country" and op == "in":
        return _check_country_in(value, properties)
    if key == "funding_stage" and op == "contains":
        return _check_stage(value, properties)
    if key == "funding_date" and op in DATE_OPS:
        return _check_date(DATE_OPS[op], value, properties)
    return Outcome.NOT_CHECKABLE


def _check_number(
    field: NumberField,
    op: NumberOp,
    value: object,
    properties: Mapping[str, object],
    employee_tolerance: float,
) -> Outcome:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} {op} needs a number, got {value!r}")
    bound = float(value)
    if field is NumberField.EMPLOYEES and op is NumberOp.LTE:
        bound *= 1 + employee_tolerance
    elif field is NumberField.EMPLOYEES and op is NumberOp.GTE:
        bound *= 1 - employee_tolerance
    return _outcome(evaluate(NumberConstraint(field, op, bound), properties))


def _check_country(value: object, properties: Mapping[str, object]) -> Outcome:
    if not isinstance(value, str):
        raise ValueError(f"country eq needs a string, got {value!r}")
    return _outcome(evaluate(TextConstraint(TextField.COUNTRY, TextOp.EQ, value), properties))


def _check_country_in(value: object, properties: Mapping[str, object]) -> Outcome:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"country in needs a list of strings, got {value!r}")
    results = {_check_country(item, properties) for item in value}
    if Outcome.PASS in results:
        return Outcome.PASS
    return Outcome.UNKNOWN if Outcome.UNKNOWN in results else Outcome.FAIL


def _check_stage(value: object, properties: Mapping[str, object]) -> Outcome:
    if not isinstance(value, str):
        raise ValueError(f"funding_stage contains needs a string, got {value!r}")
    actual = usable_text(TextField.FUNDING_STAGE, properties)
    if actual is None:
        return Outcome.UNKNOWN
    return Outcome.PASS if stage_matches(actual, value) else Outcome.FAIL


def _check_date(op: DateOp, value: object, properties: Mapping[str, object]) -> Outcome:
    if not isinstance(value, str):
        raise ValueError(f"funding_date {op} needs an ISO date string, got {value!r}")
    try:
        bound = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"funding_date {op} needs an ISO date string, got {value!r}") from None
    return _outcome(evaluate(DateConstraint(DateField.FUNDING_DATE, op, bound), properties))


def _outcome(verdict: Verdict) -> Outcome:
    return Outcome(verdict.value)
