"""Minimal client for Exa's /search endpoint that keeps the full raw response."""

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

SEARCH_URL = "https://api.exa.ai/search"
RETRYABLE_STATUS = frozenset({429, 503})
MAX_RETRY_DELAY_S = 30.0


class ExaAPIError(RuntimeError):
    """A non-retryable or exhausted /search failure. Never includes the API key."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Exa /search failed with HTTP {status}: {detail}")
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
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> ApiCall:
    """Run one search, retrying rate limits, 503s, and timeouts with backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    payload: dict[str, object] = {"query": query, "type": search_type, "numResults": num_results}
    if category is not None:
        payload["category"] = category
    if contents is not None:
        payload["contents"] = dict(contents)

    attempt = 1
    while True:
        started = time.perf_counter()
        try:
            response = client.post(SEARCH_URL, json=payload, headers={"x-api-key": api_key})
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
        raise ExaAPIError(response.status_code, _error_detail(response))
    body = _json_object(response)
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


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError:
        raise ExaAPIError(response.status_code, "response is not JSON") from None
    if not isinstance(body, dict):
        raise ExaAPIError(response.status_code, "response JSON is not an object")
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
