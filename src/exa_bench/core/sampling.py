"""Seeded, order-independent sampling of benchmark queries."""

import random
from collections.abc import Iterable

from exa_bench.core.benchmark_data import BenchmarkQuery


def seeded_subset(
    queries: Iterable[BenchmarkQuery], *, count: int, seed: int
) -> list[BenchmarkQuery]:
    """A seeded, order-independent sample of `count` queries (all of them if fewer)."""
    ordered = sorted(queries, key=lambda q: q.query_id)
    return random.Random(seed).sample(ordered, min(count, len(ordered)))
