"""Tests for seeded query sampling."""

from exa_bench.core.benchmark_data import BenchmarkQuery
from exa_bench.core.sampling import seeded_subset


def query(query_id: str) -> BenchmarkQuery:
    return BenchmarkQuery(query_id, query_id, "retrieval", "dynamic", "bucket", {})


def test_seeded_subset_is_seeded_capped_and_order_independent() -> None:
    queries = [query(f"q{i:02}") for i in range(30)]

    first = seeded_subset(queries, count=5, seed=1)
    second = seeded_subset(list(reversed(queries)), count=5, seed=1)

    assert first == second
    assert len(first) == 5
    assert seeded_subset(queries, count=5, seed=2) != first
    assert len(seeded_subset(queries[:3], count=5, seed=1)) == 3
