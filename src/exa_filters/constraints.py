"""Evaluate typed constraints against the `properties` of an Exa company entity."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeGuard, assert_never


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"  # the entity has no usable value for the constrained field


class NumberField(StrEnum):
    FOUNDED_YEAR = "founded_year"
    EMPLOYEES = "employees"
    FUNDING = "funding"


class TextField(StrEnum):
    COUNTRY = "country"
    FUNDING_STAGE = "funding_stage"


class NumberOp(StrEnum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"


class TextOp(StrEnum):
    EQ = "eq"
    CONTAINS = "contains"


# Where each field lives inside an Exa company entity's `properties` object.
FIELD_PATHS: dict[NumberField | TextField, tuple[str, ...]] = {
    NumberField.FOUNDED_YEAR: ("foundedYear",),
    NumberField.EMPLOYEES: ("workforce", "total"),
    NumberField.FUNDING: ("financials", "fundingTotal"),
    TextField.COUNTRY: ("headquarters", "country"),
    TextField.FUNDING_STAGE: ("financials", "fundingLatestRound", "name"),
}


@dataclass(frozen=True)
class NumberConstraint:
    field: NumberField
    op: NumberOp
    value: float

    def __post_init__(self) -> None:
        if not _is_number(self.value):
            raise ValueError(f"{self.field} needs a finite number, got {self.value!r}")


@dataclass(frozen=True)
class TextConstraint:
    field: TextField
    op: TextOp
    value: str


type Constraint = NumberConstraint | TextConstraint


def evaluate(constraint: Constraint, properties: Mapping[str, object]) -> Verdict:
    """Check one constraint against a company entity's `properties`."""
    if isinstance(constraint, NumberConstraint):
        number = usable_number(constraint.field, properties)
        if number is None:
            return Verdict.UNKNOWN
        return _verdict(_compare_number(constraint.op, number, constraint.value))
    text = usable_text(constraint.field, properties)
    if text is None:
        return Verdict.UNKNOWN
    return _verdict(_compare_text(constraint.op, text, constraint.value))


def usable_number(field: NumberField, properties: Mapping[str, object]) -> float | None:
    """Return the field's value if it is a finite number (not a bool), else None."""
    value = lookup(properties, FIELD_PATHS[field])
    return value if _is_number(value) else None


def usable_text(field: TextField, properties: Mapping[str, object]) -> str | None:
    """Return the field's value if it is a non-blank string, else None."""
    value = lookup(properties, FIELD_PATHS[field])
    return value if isinstance(value, str) and value.strip() else None


def lookup(properties: Mapping[str, object], path: tuple[str, ...]) -> object:
    """Follow `path` through nested objects; None if a step is missing or not an object."""
    node: object = properties
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _is_number(value: object) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _compare_number(op: NumberOp, actual: float, expected: float) -> bool:
    match op:
        case NumberOp.EQ:
            return actual == expected
        case NumberOp.GTE:
            return actual >= expected
        case NumberOp.LTE:
            return actual <= expected
        case _:
            assert_never(op)


def _compare_text(op: TextOp, actual: str, expected: str) -> bool:
    actual_norm, expected_norm = actual.strip().casefold(), expected.strip().casefold()
    match op:
        case TextOp.EQ:
            return actual_norm == expected_norm
        case TextOp.CONTAINS:
            return expected_norm in actual_norm
        case _:
            assert_never(op)


def _verdict(passed: bool) -> Verdict:
    return Verdict.PASS if passed else Verdict.FAIL
