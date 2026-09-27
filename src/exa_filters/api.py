"""Minimal client for Exa's /search and /contents endpoints that keeps the full raw response."""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

SEARCH_URL = "https://api.exa.ai/search"
CONTENTS_URL = "https://api.exa.ai/contents"
RETRYABLE_STATUS = frozenset({429, 503})
MAX_RETRY_DELAY_S = 30.0


class ExaAPIError(RuntimeError):
    """A non-retryable or exhausted request failure. Never includes the API key."""

    def __init__(self, status: int, detail: str, endpoint: str = "/search") -> None:
        super().__init__(f"Exa {endpoint} failed with HTTP {status}: {detail}")
        self.status = status


@dataclass(frozen=True)
class ApiCall:
    body: dict[str, object]  # the raw JSON response
    latency_ms: float  # client-measured, for the successful attempt
    attempts: int
    request_id: str | None
    queue_ms: float | None  # time spent in Exa's rate-limit queue
    cost_dollars: float | None


def search(
    client: httpx.Client,
    api_key: str,
    query: str,
    *,
    category: str | None = "company",
    num_results: int = 10,
    search_type: str = "auto",
    contents: Mapping[str, object] | None = None,
    include_domains: Sequence[str] | None = None,
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> ApiCall:
    """Run one search, retrying rate limits, 503s, and timeouts with backoff."""
    payload: dict[str, object] = {"query": query, "type": search_type, "numResults": num_results}
    if category is not None:
        payload["category"] = category
    if contents is not None:
        payload["contents"] = dict(contents)
    if include_domains is not None:
        payload["includeDomains"] = list(include_domains)
    return request(
        client, api_key, "POST", SEARCH_URL, payload, max_attempts=max_attempts, sleep=sleep
    )


def contents(
    client: httpx.Client,
    api_key: str,
    urls: Sequence[str],
    *,
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> ApiCall:
    """Fetch full page text for `urls`, exactly as Exa's own benchmark requests it."""
    if not urls or not all(isinstance(url, str) and url for url in urls):
        raise ValueError("urls must be a non-empty sequence of non-empty strings")
    payload: dict[str, object] = {"urls": list(urls), "text": True, "livecrawl": "fallback"}
    return request(
        client, api_key, "POST", CONTENTS_URL, payload, max_attempts=max_attempts, sleep=sleep
    )


def request(
    client: httpx.Client,
    api_key: str,
    method: str,
    url: str,
    payload: Mapping[str, object] | None = None,
    *,
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> ApiCall:
    """One JSON request, retrying rate limits, 503s, and timeouts with backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    endpoint = urlsplit(url).path
    body_json = dict(payload) if payload is not None else None
    attempt = 1
    while True:
        started = time.perf_counter()
        try:
            response = client.request(method, url, json=body_json, headers={"x-api-key": api_key})
        except httpx.TimeoutException:
            if attempt == max_attempts:
                raise
            sleep(_backoff(attempt))
            attempt += 1
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code not in RETRYABLE_STATUS or attempt == max_attempts:
            break
        sleep(_retry_delay(response, attempt))
        attempt += 1

    if response.is_error:
        raise ExaAPIError(response.status_code, _error_detail(response), endpoint)
    body = _json_object(response, endpoint)
    return ApiCall(
        body=body,
        latency_ms=latency_ms,
        attempts=attempt,
        request_id=response.headers.get("x-request-id"),
        queue_ms=_finite_float(response.headers.get("x-exa-queue-ms")),
        cost_dollars=_cost_total(body),
    )


def _backoff(attempt: int) -> float:
    return min(2.0 ** (attempt - 1), MAX_RETRY_DELAY_S)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = _finite_float(response.headers.get("retry-after"))
    if retry_after is None:
        return _backoff(attempt)
    return min(max(retry_after, 0.0), MAX_RETRY_DELAY_S)


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"non-JSON body {' '.join(response.text.split())[:200]!r}"
    if isinstance(body, dict):
        return f"{body.get('tag', 'NO_TAG')}: {body.get('error', 'no message')}"
    return f"unexpected body {str(body)[:200]!r}"


def _json_object(response: httpx.Response, endpoint: str) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError:
        raise ExaAPIError(response.status_code, "response is not JSON", endpoint) from None
    if not isinstance(body, dict):
        raise ExaAPIError(response.status_code, "response JSON is not an object", endpoint)
    return body


def _cost_total(body: Mapping[str, object]) -> float | None:
    cost = body.get("costDollars")
    total = cost.get("total") if isinstance(cost, Mapping) else None
    if isinstance(total, int | float) and not isinstance(total, bool) and math.isfinite(total):
        return float(total)
    return None


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None
