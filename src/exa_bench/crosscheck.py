"""Cross-check our typed-field verdicts against Exa's own LLM grader on a sample."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from exa_bench.benchmark_data import BenchmarkQuery
from exa_bench.coverage import company_properties, search_results
from exa_bench.grader import ConstraintOutcome, Outcome, ResultVerdict, grade

# How many results to sample per verdict of ours. Unevaluable ones test whether the
# page text carries facts the typed fields lack.
DEFAULT_STRATA = {
    ResultVerdict.VIOLATES: 120,
    ResultVerdict.SATISFIES: 120,
    ResultVerdict.UNEVALUABLE: 60,
}


@dataclass(frozen=True)
class SampleItem:
    query_id: str
    query_text: str
    constraints: Mapping[str, object]  # only the constraints our grader can check
    url: str
    title: str
    verdict: ResultVerdict
    outcomes: tuple[ConstraintOutcome, ...]


def checkable_subset(constraints: Mapping[str, object]) -> dict[str, object]:
    """The part of a query's constraints that typed fields can check, in benchmark syntax."""
    checkable = {
        (outcome.key, outcome.op)
        for outcome in grade(constraints, {}).outcomes
        if outcome.outcome is not Outcome.NOT_CHECKABLE
    }
    subset: dict[str, object] = {}
    for key, spec in constraints.items():
        if not isinstance(spec, Mapping):
            continue
        ops = {op: value for op, value in spec.items() if (key, op) in checkable}
        if ops:
            subset[key] = ops
    return subset


def build_items(
    queries: Sequence[BenchmarkQuery], bodies: Sequence[Mapping[str, object]]
) -> list[SampleItem]:
    """One item per graded result, in query order; results without a URL are skipped."""
    if len(queries) != len(bodies):
        raise ValueError(f"{len(queries)} queries but {len(bodies)} bodies")
    items: list[SampleItem] = []
    for query, body in zip(queries, bodies, strict=True):
        subset = checkable_subset(query.constraints)
        for result in search_results(body):
            if not isinstance(result, Mapping) or not isinstance(result.get("url"), str):
                continue
            properties = company_properties(result) or {}
            graded = grade(query.constraints, properties)
            title = result.get("title")
            items.append(
                SampleItem(
                    query_id=query.query_id,
                    query_text=query.text,
                    constraints=subset,
                    url=str(result["url"]),
                    title=title if isinstance(title, str) else "",
                    verdict=graded.verdict,
                    outcomes=tuple(
                        o for o in graded.outcomes if o.outcome is not Outcome.NOT_CHECKABLE
                    ),
                )
            )
    return items


def select_sample(
    items: Sequence[SampleItem],
    *,
    seed: int,
    strata: Mapping[ResultVerdict, int] = DEFAULT_STRATA,
) -> list[SampleItem]:
    """Seeded sample without replacement, `strata[verdict]` items per verdict (or all, if fewer)."""
    rng = random.Random(seed)
    sample: list[SampleItem] = []
    for verdict, count in strata.items():
        candidates = sorted(
            (item for item in items if item.verdict is verdict), key=lambda i: (i.query_id, i.url)
        )
        sample.extend(rng.sample(candidates, min(count, len(candidates))))
    return sample
