"""Aggregate graded results into the numbers the benchmark reports."""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from exa_bench.grader import GradedResult, Outcome, ResultVerdict
from exa_bench.stats import DEFAULT_RESAMPLES, Cluster, Rate, rate


@dataclass(frozen=True)
class QueryGrade:
    query_id: str
    bucket: str
    split: str  # "static" or "dynamic"
    results: tuple[GradedResult, ...]


@dataclass(frozen=True)
class QueryRow:
    query_id: str
    bucket: str
    split: str
    results: int
    violates: int
    satisfies: int
    unevaluable: int
    not_checkable: int

    @property
    def evaluable(self) -> int:
        return self.violates + self.satisfies


@dataclass(frozen=True)
class Bounds:
    """Violation rate if every unevaluable result satisfied (lower) or violated (upper)."""

    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class ConstraintStats:
    queries: int  # queries carrying this constraint; the bootstrap clusters
    passed: int
    failed: int
    unknown: int
    not_checkable: int
    fail_rate: Rate  # failed / (passed + failed)


@dataclass(frozen=True)
class Analysis:
    queries: int
    results: int
    verdicts: dict[str, int]  # every ResultVerdict, in declaration order
    violation_rate: Rate  # violates / (violates + satisfies)
    violation_bounds: Bounds
    by_bucket: dict[str, Rate]
    by_split: dict[str, Rate]
    by_constraint: dict[str, ConstraintStats]
    satisfying_histogram: dict[int, int]  # satisfying results per query -> number of queries
    per_query: tuple[QueryRow, ...]
    bootstrap_seed: int
    bootstrap_resamples: int


def analyze(
    grades: Sequence[QueryGrade], *, seed: int, resamples: int = DEFAULT_RESAMPLES
) -> Analysis:
    rows = tuple(_row(grade) for grade in grades)
    verdicts = Counter(result.verdict for grade in grades for result in grade.results)
    return Analysis(
        queries=len(rows),
        results=sum(row.results for row in rows),
        verdicts={verdict.value: verdicts[verdict] for verdict in ResultVerdict},
        violation_rate=rate(_violation_clusters(rows), seed=seed, resamples=resamples),
        violation_bounds=_bounds(rows),
        by_bucket=_rate_by(rows, lambda row: row.bucket, seed, resamples),
        by_split=_rate_by(rows, lambda row: row.split, seed, resamples),
        by_constraint=_by_constraint(grades, seed, resamples),
        satisfying_histogram=dict(sorted(Counter(row.satisfies for row in rows).items())),
        per_query=rows,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )


def _row(grade: QueryGrade) -> QueryRow:
    counts = Counter(result.verdict for result in grade.results)
    return QueryRow(
        query_id=grade.query_id,
        bucket=grade.bucket,
        split=grade.split,
        results=len(grade.results),
        violates=counts[ResultVerdict.VIOLATES],
        satisfies=counts[ResultVerdict.SATISFIES],
        unevaluable=counts[ResultVerdict.UNEVALUABLE],
        not_checkable=counts[ResultVerdict.NOT_CHECKABLE],
    )


def _violation_clusters(rows: Sequence[QueryRow]) -> list[Cluster]:
    return [(row.violates, row.evaluable) for row in rows]


def _bounds(rows: Sequence[QueryRow]) -> Bounds:
    violates = sum(row.violates for row in rows)
    unevaluable = sum(row.unevaluable for row in rows)
    graded = violates + sum(row.satisfies for row in rows) + unevaluable
    if graded == 0:
        return Bounds(None, None)
    return Bounds(violates / graded, (violates + unevaluable) / graded)


def _rate_by(
    rows: Sequence[QueryRow], key: Callable[[QueryRow], str], seed: int, resamples: int
) -> dict[str, Rate]:
    groups: defaultdict[str, list[QueryRow]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {
        name: rate(_violation_clusters(group), seed=seed, resamples=resamples)
        for name, group in sorted(groups.items())
    }


def _by_constraint(
    grades: Sequence[QueryGrade], seed: int, resamples: int
) -> dict[str, ConstraintStats]:
    counts: defaultdict[str, Counter[Outcome]] = defaultdict(Counter)
    clusters: defaultdict[str, list[Cluster]] = defaultdict(list)
    for grade in grades:
        per_query: defaultdict[str, Counter[Outcome]] = defaultdict(Counter)
        for result in grade.results:
            for outcome in result.outcomes:
                per_query[outcome.key][outcome.outcome] += 1
        for key, query_counts in per_query.items():
            counts[key].update(query_counts)
            failed, passed = query_counts[Outcome.FAIL], query_counts[Outcome.PASS]
            clusters[key].append((failed, failed + passed))
    return {
        key: ConstraintStats(
            queries=len(clusters[key]),
            passed=counts[key][Outcome.PASS],
            failed=counts[key][Outcome.FAIL],
            unknown=counts[key][Outcome.UNKNOWN],
            not_checkable=counts[key][Outcome.NOT_CHECKABLE],
            fail_rate=rate(clusters[key], seed=seed, resamples=resamples),
        )
        for key in sorted(counts)
    }
