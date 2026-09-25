"""Exa's public company-search benchmark (exa-labs/benchmarks), pinned to one commit."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

COMMIT = "c096f1aa3d4eb452568c588ee3ad4887e25060fa"
COMPANY_QUERIES_URL = (
    f"https://raw.githubusercontent.com/exa-labs/benchmarks/{COMMIT}"
    "/simple-company-benchmark/data/company/simple_company_search.jsonl"
)
COMPANY_QUERIES_SHA256 = "3f4c54433d72d8abfdbb11f578a75acb735c2538b545740f5188848066379529"


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str
    track: str  # "retrieval" or "rag"
    split: str  # "static" or "dynamic"
    bucket: str
    constraints: Mapping[str, object]


def parse_queries(lines: Iterable[str]) -> list[BenchmarkQuery]:
    """Parse benchmark JSONL lines, skipping blank ones."""
    queries: list[BenchmarkQuery] = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        queries.append(
            BenchmarkQuery(
                query_id=row["query_id"],
                text=row["text"],
                track=row["track"],
                split=row["split"],
                bucket=row["bucket"],
                constraints=row.get("constraints") or {},
            )
        )
    return queries


def load_company_queries(
    cache_dir: Path,
    client: httpx.Client,
    *,
    url: str = COMPANY_QUERIES_URL,
    sha256: str = COMPANY_QUERIES_SHA256,
) -> list[BenchmarkQuery]:
    """Return the pinned company queries, downloading them into `cache_dir` on first use.

    Both the download and the cached copy are checked against `sha256`.
    """
    path = cache_dir / f"simple_company_search-{sha256[:12]}.jsonl"
    if not path.exists():
        response = client.get(url)
        response.raise_for_status()
        _check_sha256(response.content, sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(response.content)
        partial.replace(path)
    data = path.read_bytes()
    _check_sha256(data, sha256)
    return parse_queries(data.decode("utf-8").splitlines())


def _check_sha256(data: bytes, expected: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"benchmark file hash mismatch: expected {expected}, got {actual}")
