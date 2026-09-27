"""Tests for the per-URL page-text cache (no network: scripted fake server)."""

import json
from pathlib import Path

import httpx
import pytest

from exa_bench.contents_cache import PageText, cached_contents, uncached_urls
from exa_filters.api import ExaAPIError

API_KEY = "test-key-do-not-leak"
A, B, C = "https://a.test/", "https://b.test/", "https://c.test/"


def scripted_client(replies: list[httpx.Response], seen: list[httpx.Request]) -> httpx.Client:
    pending = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return pending.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def response(
    results: list[dict[str, object]], statuses: list[dict[str, object]] | None = None
) -> httpx.Response:
    body = {
        "requestId": "r",
        "results": results,
        "statuses": statuses or [],
        "costDollars": {"total": 0.001 * len(results)},
    }
    return httpx.Response(200, json=body)


def test_fetches_then_serves_from_cache(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    reply = response([{"id": A, "url": A, "text": "alpha"}, {"id": B, "url": B, "text": "beta"}])
    with scripted_client([reply], seen) as client:
        first = cached_contents(tmp_path, client, API_KEY, [A, B])
        second = cached_contents(tmp_path, client, API_KEY, [B, A])

    assert {url: page.text for url, page in first.pages.items()} == {A: "alpha", B: "beta"}
    assert (first.from_cache, len(first.calls)) == (0, 1)
    assert second.pages == first.pages
    assert (second.from_cache, second.calls) == (2, [])
    assert len(seen) == 1
    assert json.loads(seen[0].content)["urls"] == [A, B]


def test_fetches_only_missing_urls_in_batches(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    replies = [
        response([{"id": A, "url": A, "text": "alpha"}]),
        response([{"id": B, "url": B, "text": "beta"}]),
        response([{"id": C, "url": C, "text": "gamma"}]),
    ]
    with scripted_client(replies, seen) as client:
        cached_contents(tmp_path, client, API_KEY, [A])
        result = cached_contents(tmp_path, client, API_KEY, [A, B, C, B], batch_size=1)

    assert [json.loads(r.content)["urls"] for r in seen] == [[A], [B], [C]]
    assert result.from_cache == 1
    assert len(result.calls) == 2
    assert [page.text for page in result.pages.values()] == ["alpha", "beta", "gamma"]


def test_missing_page_keeps_its_status_and_is_cached_as_missing(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    status: dict[str, object] = {
        "id": B,
        "status": "error",
        "error": {"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404},
    }
    reply = response([{"id": A, "url": A, "text": "alpha"}], statuses=[status])
    with scripted_client([reply], seen) as client:
        first = cached_contents(tmp_path, client, API_KEY, [A, B])
        second = cached_contents(tmp_path, client, API_KEY, [B])

    assert first.pages[B] == PageText(None, status)
    assert second.pages[B] == PageText(None, status)
    assert len(seen) == 1  # the miss was cached too, so it is not re-fetched


def test_result_matched_by_id_when_exa_normalizes_the_url(tmp_path: Path) -> None:
    reply = response([{"id": A, "url": "https://a.test/index.html", "text": "alpha"}])
    with scripted_client([reply], []) as client:
        result = cached_contents(tmp_path, client, API_KEY, [A])

    assert result.pages[A].text == "alpha"


def test_uncached_urls_deduplicates_and_skips_cached(tmp_path: Path) -> None:
    with scripted_client([response([{"id": A, "url": A, "text": "alpha"}])], []) as client:
        cached_contents(tmp_path, client, API_KEY, [A])

    assert uncached_urls(tmp_path, [B, A, B, C]) == [B, C]


def test_cache_files_never_contain_the_key(tmp_path: Path) -> None:
    with scripted_client([response([{"id": A, "url": A, "text": "alpha"}])], []) as client:
        cached_contents(tmp_path, client, API_KEY, [A])

    (cached,) = tmp_path.glob("*.json")
    assert API_KEY not in cached.read_text()


def test_failed_batch_caches_nothing(tmp_path: Path) -> None:
    reply = httpx.Response(401, json={"tag": "INVALID_API_KEY", "error": "bad key"})
    with scripted_client([reply], []) as client, pytest.raises(ExaAPIError):
        cached_contents(tmp_path, client, API_KEY, [A])

    assert list(tmp_path.iterdir()) == []


def test_rejects_bad_batch_size(tmp_path: Path) -> None:
    with scripted_client([], []) as client, pytest.raises(ValueError, match="batch_size"):
        cached_contents(tmp_path, client, API_KEY, [A], batch_size=0)


def test_empty_text_counts_as_missing(tmp_path: Path) -> None:
    with scripted_client([response([{"id": A, "url": A, "text": ""}])], []) as client:
        result = cached_contents(tmp_path, client, API_KEY, [A])

    assert result.pages[A] == PageText(None, None)


def test_mismatched_cache_file_is_rejected(tmp_path: Path) -> None:
    with scripted_client([response([{"id": A, "url": A, "text": "alpha"}])], []) as client:
        cached_contents(tmp_path, client, API_KEY, [A])
        (cached,) = tmp_path.glob("*.json")
        record = json.loads(cached.read_text())
        record["url"] = B
        cached.write_text(json.dumps(record))

        with pytest.raises(ValueError, match="does not match"):
            cached_contents(tmp_path, client, API_KEY, [A])
