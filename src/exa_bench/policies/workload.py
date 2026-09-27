"""The evaluation workload: the benchmark queries whose top 10 needed help, plus a clean sample."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from exa_bench.constraints.benchmark import load_shallow, non_clean_queries, select_gradable
from exa_bench.core.benchmark_data import BenchmarkQuery, load_company_queries
from exa_bench.core.sampling import seeded_subset

CLEAN_SAMPLE_COUNT = 30
CLEAN_SAMPLE_SEED = 20260926


@dataclass(frozen=True)
class Workload:
    queries: tuple[BenchmarkQuery, ...]  # by query id
    clean_ids: frozenset[str]  # the sampled queries whose cached top-10 already satisfied
    seed: int  # of the clean sample


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
    return Workload(tuple(selected), frozenset(query.query_id for query in sample), seed)


def load_workload(cache_dir: Path, client: httpx.Client) -> Workload:
    """The workload from the pinned benchmark and its cached top 10s; never sends a search.

    Raises FileNotFoundError when a query's top 10 is not in the cache.
    """
    queries = select_gradable(load_company_queries(cache_dir / "benchmarks", client))
    shallow = load_shallow(cache_dir, queries)
    return select_workload(queries, [shallow[q.query_id].body for q in queries])
