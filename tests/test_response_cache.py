"""Tests for the on-disk Exa response cache (no network: scripted fake server)."""

import json
from pathlib import Path

import httpx
import pytest

from exa_bench.exa_api import ExaAPIError
from exa_bench.response_cache import cached_search, is_cached, read_cached

API_KEY = "test-key-do-not-leak"
OK_BODY = {"requestId": "req-1", "results": [], "costDollars": {"total": 0.007}}


def scripted_client(replies: list[httpx.Response], seen: list[httpx.Request]) -> httpx.Client:
    """A client whose fake server answers each request with the next scripted reply."""
    pending = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return pending.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_second_identical_search_is_served_from_cache(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    with scripted_client([httpx.Response(200, json=OK_BODY)], seen) as client:
        first = cached_search(tmp_path, client, API_KEY, "q")
        second = cached_search(tmp_path, client, API_KEY, "q")

    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.call == first.call
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("query", "category", "num_results", "search_type", "contents"),
    [
        ("other", "company", 10, "auto", None),
        ("q", None, 10, "auto", None),
        ("q", "company", 25, "auto", None),
        ("q", "company", 10, "fast", None),
        ("q", "company", 10, "auto", {"highlights": True}),
    ],
)
def test_any_parameter_change_is_a_separate_entry(
    tmp_path: Path,
    query: str,
    category: str | None,
    num_results: int,
    search_type: str,
    contents: dict[str, object] | None,
) -> None:
    seen: list[httpx.Request] = []
    with scripted_client([httpx.Response(200, json=OK_BODY)] * 2, seen) as client:
        cached_search(tmp_path, client, API_KEY, "q")
        cached_search(
            tmp_path,
            client,
            API_KEY,
            query,
            category=category,
            num_results=num_results,
            search_type=search_type,
            contents=contents,
        )

    assert len(seen) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_is_cached_reflects_exact_parameters(tmp_path: Path) -> None:
    assert not is_cached(tmp_path, "q")
    with scripted_client([httpx.Response(200, json=OK_BODY)], []) as client:
        cached_search(tmp_path, client, API_KEY, "q")

    assert is_cached(tmp_path, "q")
    assert not is_cached(tmp_path, "q", num_results=25)
    assert not is_cached(tmp_path, "other")


def test_cache_tag_separates_entries_without_reaching_exa(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    replies = [httpx.Response(200, json=OK_BODY), httpx.Response(200, json=OK_BODY)]
    with scripted_client(replies, seen) as client:
        untagged = cached_search(tmp_path, client, API_KEY, "q")
        tagged = cached_search(tmp_path, client, API_KEY, "q", cache_tag="repeat-1")
        again = cached_search(tmp_path, client, API_KEY, "q", cache_tag="repeat-1")

    assert (untagged.from_cache, tagged.from_cache, again.from_cache) == (False, False, True)
    assert json.loads(seen[0].content) == json.loads(seen[1].content)  # identical to Exa
    assert "cache_tag" not in json.loads(seen[1].content)
    assert is_cached(tmp_path, "q", cache_tag="repeat-1")
    assert not is_cached(tmp_path, "q", cache_tag="repeat-2")
    assert read_cached(tmp_path, "q", cache_tag="repeat-1") == tagged.call


def test_untagged_keys_are_unchanged_by_the_tag_feature(tmp_path: Path) -> None:
    with scripted_client([httpx.Response(200, json=OK_BODY)], []) as client:
        cached_search(tmp_path, client, API_KEY, "q")
    (cached,) = tmp_path.glob("*.json")

    assert "cache_tag" not in json.loads(cached.read_text())["params"]
    assert is_cached(tmp_path, "q", cache_tag=None)


def test_include_domains_is_part_of_the_key_and_absent_by_default(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    replies = [httpx.Response(200, json=OK_BODY), httpx.Response(200, json=OK_BODY)]
    with scripted_client(replies, seen) as client:
        plain = cached_search(tmp_path, client, API_KEY, "q")
        scoped = cached_search(tmp_path, client, API_KEY, "q", include_domains=["acme.com"])
        again = cached_search(tmp_path, client, API_KEY, "q", include_domains=["acme.com"])

    assert (plain.from_cache, scoped.from_cache, again.from_cache) == (False, False, True)
    assert json.loads(seen[1].content)["includeDomains"] == ["acme.com"]
    assert is_cached(tmp_path, "q", include_domains=["acme.com"])
    assert not is_cached(tmp_path, "q", include_domains=["other.com"])
    assert read_cached(tmp_path, "q", include_domains=["acme.com"]) == scoped.call
    params = [json.loads(path.read_text())["params"] for path in tmp_path.glob("*.json")]
    assert sorted("include_domains" in p for p in params) == [False, True]


def test_read_cached_never_requests(tmp_path: Path) -> None:
    assert read_cached(tmp_path, "q") is None
    seen: list[httpx.Request] = []
    with scripted_client([httpx.Response(200, json=OK_BODY)], seen) as client:
        fetched = cached_search(tmp_path, client, API_KEY, "q")

    assert read_cached(tmp_path, "q") == fetched.call
    assert read_cached(tmp_path, "q", num_results=25) is None
    assert len(seen) == 1


def test_retry_sleep_is_passed_through(tmp_path: Path) -> None:
    sleeps: list[float] = []
    replies = [httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, json=OK_BODY)]
    with scripted_client(replies, []) as client:
        result = cached_search(tmp_path, client, API_KEY, "q", sleep=sleeps.append)

    assert sleeps == [3.0]
    assert result.call.attempts == 2


def test_cache_file_never_contains_the_api_key(tmp_path: Path) -> None:
    with scripted_client([httpx.Response(200, json=OK_BODY)], []) as client:
        cached_search(tmp_path, client, API_KEY, "q")

    (cached,) = tmp_path.glob("*.json")
    assert API_KEY not in cached.read_text()


def test_failed_search_is_not_cached(tmp_path: Path) -> None:
    reply = httpx.Response(401, json={"tag": "INVALID_API_KEY"})
    with scripted_client([reply], []) as client, pytest.raises(ExaAPIError):
        cached_search(tmp_path, client, API_KEY, "q")

    assert list(tmp_path.iterdir()) == []


def test_mismatched_cache_file_is_rejected(tmp_path: Path) -> None:
    with scripted_client([httpx.Response(200, json=OK_BODY)], []) as client:
        cached_search(tmp_path, client, API_KEY, "q")
        (cached,) = tmp_path.glob("*.json")
        record = json.loads(cached.read_text())
        record["params"]["query"] = "something else"
        cached.write_text(json.dumps(record))

        with pytest.raises(ValueError, match="does not match"):
            cached_search(tmp_path, client, API_KEY, "q")
