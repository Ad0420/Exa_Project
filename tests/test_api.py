"""Tests for the minimal Exa /search client (no network: scripted fake server)."""

import json

import httpx
import pytest

from exa_filters.api import (
    CONTENTS_URL,
    MAX_RETRY_DELAY_S,
    SEARCH_URL,
    ExaAPIError,
    contents,
    request,
    search,
)

API_KEY = "test-key-do-not-leak"
OK_BODY = {"requestId": "req-1", "results": [], "costDollars": {"total": 0.007}}


def scripted_client(
    replies: list[httpx.Response | Exception], seen: list[httpx.Request]
) -> httpx.Client:
    """A client whose fake server answers each request with the next scripted reply."""
    pending = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        reply = pending.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sends_query_category_and_key() -> None:
    seen: list[httpx.Request] = []
    with scripted_client([httpx.Response(200, json=OK_BODY)], seen) as client:
        search(client, API_KEY, "fintech startups in Singapore")

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == SEARCH_URL
    assert request.headers["x-api-key"] == API_KEY
    assert json.loads(request.content) == {
        "query": "fintech startups in Singapore",
        "type": "auto",
        "numResults": 10,
        "category": "company",
    }


def test_optional_fields_are_sent_only_when_given() -> None:
    seen: list[httpx.Request] = []
    with scripted_client([httpx.Response(200, json=OK_BODY)], seen) as client:
        search(
            client,
            API_KEY,
            "q",
            category=None,
            contents={"highlights": True},
            include_domains=["acme.com"],
        )

    body = json.loads(seen[0].content)
    assert "category" not in body
    assert body["contents"] == {"highlights": True}
    assert body["includeDomains"] == ["acme.com"]


def test_parses_body_cost_and_headers() -> None:
    headers = {"x-request-id": "req-1", "x-exa-queue-ms": "12"}
    with scripted_client([httpx.Response(200, json=OK_BODY, headers=headers)], []) as client:
        call = search(client, API_KEY, "q")

    assert call.body == OK_BODY
    assert call.cost_dollars == 0.007
    assert call.request_id == "req-1"
    assert call.queue_ms == 12.0
    assert call.attempts == 1
    assert call.latency_ms >= 0


@pytest.mark.parametrize("queue_header", ["soon", "nan", "inf"])
def test_missing_cost_and_bad_queue_header_become_none(queue_header: str) -> None:
    reply = httpx.Response(200, json={"results": []}, headers={"x-exa-queue-ms": queue_header})
    with scripted_client([reply], []) as client:
        call = search(client, API_KEY, "q")

    assert call.cost_dollars is None
    assert call.queue_ms is None


def test_rate_limit_waits_for_retry_after_then_succeeds() -> None:
    sleeps: list[float] = []
    replies: list[httpx.Response | Exception] = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json=OK_BODY),
    ]
    with scripted_client(replies, []) as client:
        call = search(client, API_KEY, "q", sleep=sleeps.append)

    assert call.attempts == 2
    assert sleeps == [2.0]


def test_backs_off_exponentially_without_retry_after() -> None:
    sleeps: list[float] = []
    replies: list[httpx.Response | Exception] = [
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json=OK_BODY),
    ]
    with scripted_client(replies, []) as client:
        call = search(client, API_KEY, "q", sleep=sleeps.append)

    assert call.attempts == 4
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.parametrize(("retry_after", "expected"), [("999", MAX_RETRY_DELAY_S), ("-5", 0.0)])
def test_retry_after_is_clamped(retry_after: str, expected: float) -> None:
    sleeps: list[float] = []
    replies: list[httpx.Response | Exception] = [
        httpx.Response(429, headers={"Retry-After": retry_after}),
        httpx.Response(200, json=OK_BODY),
    ]
    with scripted_client(replies, []) as client:
        search(client, API_KEY, "q", sleep=sleeps.append)

    assert sleeps == [expected]


def test_gives_up_after_max_attempts() -> None:
    seen: list[httpx.Request] = []
    sleeps: list[float] = []
    replies: list[httpx.Response | Exception] = [httpx.Response(429)] * 3
    with scripted_client(replies, seen) as client, pytest.raises(ExaAPIError) as error:
        search(client, API_KEY, "q", max_attempts=3, sleep=sleeps.append)

    assert error.value.status == 429
    assert len(seen) == 3
    assert len(sleeps) == 2


def test_timeout_is_retried_then_reraised_when_exhausted() -> None:
    sleeps: list[float] = []
    replies: list[httpx.Response | Exception] = [
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, json=OK_BODY),
    ]
    with scripted_client(replies, []) as client:
        assert search(client, API_KEY, "q", sleep=sleeps.append).attempts == 2
    assert sleeps == [1.0]

    exhausted: list[httpx.Response | Exception] = [httpx.ReadTimeout("timed out")] * 2
    with scripted_client(exhausted, []) as client, pytest.raises(httpx.ReadTimeout):
        search(client, API_KEY, "q", max_attempts=2, sleep=lambda _: None)


def test_client_error_raises_once_with_tag_and_never_the_key() -> None:
    seen: list[httpx.Request] = []
    body = {"requestId": "r", "error": "Invalid API key", "tag": "INVALID_API_KEY"}
    reply = httpx.Response(401, json=body)
    with scripted_client([reply], seen) as client, pytest.raises(ExaAPIError) as error:
        search(client, API_KEY, "q")

    assert error.value.status == 401
    assert "INVALID_API_KEY" in str(error.value)
    assert API_KEY not in str(error.value)
    assert len(seen) == 1


def test_non_json_error_body_is_summarized() -> None:
    reply = httpx.Response(502, text="<html>\n  <body>Bad   gateway</body>\n</html>")
    with scripted_client([reply], []) as client, pytest.raises(ExaAPIError) as error:
        search(client, API_KEY, "q")

    assert "non-JSON body" in str(error.value)
    assert "<body>Bad gateway</body>" in str(error.value)


def test_success_body_must_be_a_json_object() -> None:
    reply = httpx.Response(200, json=[])
    with scripted_client([reply], []) as client, pytest.raises(ExaAPIError, match="not an object"):
        search(client, API_KEY, "q")


def test_rejects_non_positive_max_attempts() -> None:
    with scripted_client([], []) as client, pytest.raises(ValueError, match="at least 1"):
        search(client, API_KEY, "q", max_attempts=0)


def test_request_supports_get_without_a_body_and_names_the_path() -> None:
    seen: list[httpx.Request] = []
    replies: list[httpx.Response | Exception] = [
        httpx.Response(200, json={"id": "r1"}),
        httpx.Response(404, json={"tag": "RUN_NOT_FOUND", "error": "no such run"}),
    ]
    with scripted_client(replies, seen) as client:
        call = request(client, API_KEY, "GET", "https://api.exa.ai/agent/runs/r1")
        with pytest.raises(ExaAPIError) as error:
            request(client, API_KEY, "GET", "https://api.exa.ai/agent/runs/r2")

    assert call.body == {"id": "r1"}
    assert (seen[0].method, seen[0].content) == ("GET", b"")
    assert "/agent/runs/r2" in str(error.value)
    assert "RUN_NOT_FOUND" in str(error.value)


def test_contents_sends_urls_as_exas_benchmark_does() -> None:
    seen: list[httpx.Request] = []
    body = {"requestId": "req-2", "results": [{"url": "https://a.test", "text": "hello"}]}
    with scripted_client([httpx.Response(200, json=body)], seen) as client:
        call = contents(client, API_KEY, ["https://a.test", "https://b.test"])

    (request,) = seen
    assert str(request.url) == CONTENTS_URL
    assert request.headers["x-api-key"] == API_KEY
    assert json.loads(request.content) == {
        "urls": ["https://a.test", "https://b.test"],
        "text": True,
        "livecrawl": "fallback",
    }
    assert call.body == body


def test_contents_shares_the_retry_path_and_names_its_endpoint() -> None:
    sleeps: list[float] = []
    replies: list[httpx.Response | Exception] = [
        httpx.Response(429, headers={"Retry-After": "1"}),
        httpx.Response(500, json={"tag": "INTERNAL", "error": "boom"}),
    ]
    with scripted_client(replies, []) as client, pytest.raises(ExaAPIError) as error:
        contents(client, API_KEY, ["https://a.test"], sleep=sleeps.append)

    assert sleeps == [1.0]
    assert "/contents" in str(error.value)
    assert "INTERNAL" in str(error.value)


@pytest.mark.parametrize("urls", [[], [""], ["https://a.test", 3]])
def test_contents_rejects_bad_url_lists(urls: list[object]) -> None:
    with scripted_client([], []) as client, pytest.raises(ValueError, match="non-empty"):
        contents(client, API_KEY, urls)  # type: ignore[arg-type]
