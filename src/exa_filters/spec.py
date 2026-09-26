"""Ergonomic filter specs, e.g. Filters(employees=Range(lte=30), country=In(["SG", "Singapore"])).

These build the typed filters in `exa_filters.evaluate`; they add no semantics of their own.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from exa_filters.constraints import DateField, DateOp, NumberField, NumberOp
from exa_filters.evaluate import CountryFilter, DateFilter, Filter, NumberFilter, StageFilter


@dataclass(frozen=True)
class Range:
    """Bounds on a number or date field. `tolerance` widens gte/lte by that fraction."""

    gte: float | date | None = None
    lte: float | date | None = None
    eq: float | None = None
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.gte is None and self.lte is None and self.eq is None:
            raise ValueError("Range needs at least one of gte, lte, eq")
        if self.eq is not None and (self.gte is not None or self.lte is not None):
            raise ValueError("Range eq cannot be combined with gte or lte")


@dataclass(frozen=True)
class Eq:
    value: str


@dataclass(frozen=True)
class In:
    values: tuple[str, ...]

    def __init__(self, values: Sequence[str]) -> None:
        object.__setattr__(self, "values", tuple(values))


@dataclass(frozen=True)
class Contains:
    text: str


class Filters:
    """The filters for one search, kept in a fixed field order."""

    def __init__(
        self,
        *,
        founded_year: Range | None = None,
        employees: Range | None = None,
        funding: Range | None = None,
        country: Eq | In | None = None,
        funding_stage: Contains | None = None,
        funding_date: Range | None = None,
    ) -> None:
        items: list[Filter] = []
        items += _number_filters(NumberField.FOUNDED_YEAR, founded_year)
        items += _number_filters(NumberField.EMPLOYEES, employees)
        items += _number_filters(NumberField.FUNDING, funding)
        if country is not None:
            items.append(CountryFilter(_countries(country)))
        if funding_stage is not None:
            items.append(StageFilter(funding_stage.text))
        items += _date_filters(DateField.FUNDING_DATE, funding_date)
        self.items: tuple[Filter, ...] = tuple(items)

    def __repr__(self) -> str:
        return f"Filters({', '.join(repr(item) for item in self.items)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Filters) and other.items == self.items

    def __hash__(self) -> int:
        return hash(self.items)


def _number_filters(field: NumberField, bounds: Range | None) -> list[Filter]:
    if bounds is None:
        return []
    filters: list[Filter] = []
    for op, value in (
        (NumberOp.EQ, bounds.eq),
        (NumberOp.GTE, bounds.gte),
        (NumberOp.LTE, bounds.lte),
    ):
        if value is not None:
            filters.append(NumberFilter(field, op, _as_number(field, value), bounds.tolerance))
    return filters


def _as_number(field: NumberField, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field.value} bounds must be numbers, got {value!r}")
    return float(value)


def _date_filters(field: DateField, bounds: Range | None) -> list[Filter]:
    if bounds is None:
        return []
    if bounds.eq is not None:
        raise ValueError(f"{field.value} does not support eq; give gte and lte")
    if bounds.tolerance:
        raise ValueError(f"{field.value} does not support a tolerance")
    filters: list[Filter] = []
    for op, value in ((DateOp.GTE, bounds.gte), (DateOp.LTE, bounds.lte)):
        if value is None:
            continue
        if not isinstance(value, date):
            raise ValueError(f"{field.value} bounds must be datetime.date, got {value!r}")
        filters.append(DateFilter(field, op, value))  # DateFilter rejects datetime itself
    return filters


def _countries(spec: Eq | In) -> tuple[str, ...]:
    return (spec.value,) if isinstance(spec, Eq) else spec.values
