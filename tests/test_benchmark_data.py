"""Tests for loading Exa's pinned company benchmark."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from exa_bench.core.benchmark_data import load_company_queries, parse_queries

FAKE_URL = "https://example.test/simple_company_search.jsonl"
ROWS = [
    {
        "query_id": "employee_count_dynamic_0001",
        "text": "lean fintech startup in Singapore, under 30 people",
        "track": "retrieval",
        "split": "dynamic",
        "bucket": "employee_count",
        "constraints": {"employees": {"lte": 30}, "country": {"eq": "Singapore"}},
    },
    {
        "query_id": "founding_static_0001",
        "text": "When was Sakana AI founded?",
        "track": "rag",
        "split": "static",
        "bucket": "founding",
    },
]
PAYLOAD = ("\n".join(json.dumps(row) for row in ROWS) + "\n\n").encode()
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


def fake_client(content: bytes, seen: list[httpx.Request], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_queries_reads_fields_and_defaults_missing_constraints() -> None:
    queries = parse_queries(PAYLOAD.decode().splitlines())

    assert [q.query_id for q in queries] == ["employee_count_dynamic_0001", "founding_static_0001"]
    assert queries[0].constraints == {"employees": {"lte": 30}, "country": {"eq": "Singapore"}}
    assert queries[1].constraints == {}


def test_downloads_once_then_reads_from_cache(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    with fake_client(PAYLOAD, seen) as client:
        first = load_company_queries(tmp_path, client, url=FAKE_URL, sha256=PAYLOAD_SHA256)
        second = load_company_queries(tmp_path, client, url=FAKE_URL, sha256=PAYLOAD_SHA256)

    assert first == second
    assert len(first) == 2
    assert [str(request.url) for request in seen] == [FAKE_URL]


def test_rejects_download_with_wrong_hash_and_caches_nothing(tmp_path: Path) -> None:
    with fake_client(b"tampered", []) as client, pytest.raises(ValueError, match="hash mismatch"):
        load_company_queries(tmp_path, client, url=FAKE_URL, sha256=PAYLOAD_SHA256)

    assert list(tmp_path.iterdir()) == []


def test_rejects_corrupted_cache(tmp_path: Path) -> None:
    with fake_client(PAYLOAD, []) as client:
        load_company_queries(tmp_path, client, url=FAKE_URL, sha256=PAYLOAD_SHA256)
        (cached,) = tmp_path.iterdir()
        cached.write_bytes(b"corrupted")

        with pytest.raises(ValueError, match="hash mismatch"):
            load_company_queries(tmp_path, client, url=FAKE_URL, sha256=PAYLOAD_SHA256)


def test_http_error_is_raised(tmp_path: Path) -> None:
    with fake_client(b"", [], status=404) as client, pytest.raises(httpx.HTTPStatusError):
        load_company_queries(tmp_path, client, url=FAKE_URL, sha256=PAYLOAD_SHA256)
