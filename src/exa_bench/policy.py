"""Policy evaluation: run each fetch policy on the same queries through the filters feature."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from exa_bench.benchmark_data import BenchmarkQuery
from exa_bench.depth import non_clean_queries, seeded_subset

CLEAN_SAMPLE_COUNT = 30
CLEAN_SAMPLE_SEED = 20260926


@dataclass(frozen=True)
class Workload:
    queries: tuple[BenchmarkQuery, ...]  # by query id
    clean_ids: frozenset[str]  # the sampled queries whose cached top-10 already satisfied


def select_workload(
    queries: Sequence[BenchmarkQuery],
    bodies: Sequence[Mapping[str, object]],
    *,
    clean_count: int = CLEAN_SAMPLE_COUNT,
    seed: int = CLEAN_SAMPLE_SEED,
) -> Workload:
    """Every query whose cached top-10 is not clean, plus a seeded sample of clean ones.

    The clean sample answers whether over-fetching disturbs a top-10 that needed no help.
    """
    non_clean = non_clean_queries(queries, bodies)
    non_clean_ids = {query.query_id for query in non_clean}
    clean = [query for query in queries if query.query_id not in non_clean_ids]
    sample = seeded_subset(clean, count=clean_count, seed=seed)
    selected = sorted(non_clean + sample, key=lambda query: query.query_id)
    return Workload(tuple(selected), frozenset(query.query_id for query in sample))
