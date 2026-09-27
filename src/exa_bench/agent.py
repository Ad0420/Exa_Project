"""Exa Agent runs for the comparison: the request, the create-then-poll client, the parsed run."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from exa_bench.json_cache import cache_key, cached_record
from exa_filters.api import request

AGENT_RUNS_URL = "https://api.exa.ai/agent/runs"
MAX_COMPANIES = 10
# Flat list price per run at each fixed effort (docs, 2026-09-26); auto and ultra are metered.
EFFORT_PRICE_USD = {"minimal": 0.012, "low": 0.025, "medium": 0.10, "high": 0.50, "xhigh": 1.00}
PRICE_PER_SEARCH_USD = 0.005  # each search the agent makes, on top of the flat price
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def agent_request(
    query_text: str, *, effort: str = "low", k: int = MAX_COMPANIES
) -> dict[str, object]:
    """A list-building run for a benchmark query: its own text plus the ask for k companies."""
    if effort not in EFFORT_PRICE_USD:
        raise ValueError(f"effort must be one of {', '.join(EFFORT_PRICE_USD)}, got {effort!r}")
    if not 1 <= k <= MAX_COMPANIES:
        raise ValueError(f"k must be in [1, {MAX_COMPANIES}]")
    ask = (
        f"Return up to {k} companies that match every requirement. For each, give the company "
        "name and its homepage domain."
    )
    return {
        "query": f"{query_text.strip()}\n\n{ask}",
        "effort": effort,
        "outputSchema": _output_schema(k),
    }


def _output_schema(k: int) -> dict[str, object]:
    company = {"type": "string", "description": "The company's name"}
    domain = {"type": "string", "description": "The company's homepage domain, such as acme.com"}
    return {
        "type": "object",
        "required": ["companies"],
        "properties": {
            "companies": {
                "type": "array",
                "maxItems": k,
                "items": {
                    "type": "object",
                    "required": ["company", "domain"],
                    "properties": {"company": company, "domain": domain},
                },
            }
        },
    }


@dataclass(frozen=True)
class AgentCompany:
    name: str
    domain: str


@dataclass(frozen=True)
class AgentRun:
    id: str
    status: str
    stop_reason: str | None
    companies: tuple[AgentCompany, ...]  # output.structured.companies with a usable domain
    cost_dollars: float | None  # costDollars.total
    searches: int | None  # usage.searches
    server_ms: float | None  # completedAt minus createdAt


def parse_run(body: Mapping[str, object]) -> AgentRun:
    """Read a run object as the Agent API returns it; anything malformed reads as absent."""
    usage = body.get("usage")
    cost = body.get("costDollars")
    return AgentRun(
        id=_string(body, "id") or "",
        status=_string(body, "status") or "",
        stop_reason=_string(body, "stopReason"),
        companies=_companies(body),
        cost_dollars=_number(cost, "total") if isinstance(cost, Mapping) else None,
        searches=_count(usage, "searches") if isinstance(usage, Mapping) else None,
        server_ms=_elapsed_ms(_string(body, "createdAt"), _string(body, "completedAt")),
    )


def _companies(body: Mapping[str, object]) -> tuple[AgentCompany, ...]:
    output = body.get("output")
    structured = output.get("structured") if isinstance(output, Mapping) else None
    items = structured.get("companies") if isinstance(structured, Mapping) else None
    companies: list[AgentCompany] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, Mapping):
            continue
        domain, name = item.get("domain"), item.get("company")
        if isinstance(domain, str) and domain.strip():
            companies.append(
                AgentCompany(name.strip() if isinstance(name, str) else "", domain.strip())
            )
    return tuple(companies)


def _string(body: Mapping[str, object], key: str) -> str | None:
    value = body.get(key)
    return value if isinstance(value, str) else None


def _number(body: Mapping[str, object], key: str) -> float | None:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _count(body: Mapping[str, object], key: str) -> int | None:
    value = body.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _elapsed_ms(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    except ValueError:
        return None
    return delta.total_seconds() * 1000


@dataclass(frozen=True)
class RunResult:
    body: dict[str, object]  # the terminal run object
    client_ms: float  # wall time from the create request to the terminal poll
    polls: int


class AgentTimeout(TimeoutError):
    """The run reached no terminal status in time; it may still be running, and billing."""


def run_to_completion(
    client: httpx.Client,
    api_key: str,
    request_body: Mapping[str, object],
    *,
    poll_s: float = 3.0,
    timeout_s: float = 900.0,
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Create a run, then poll it until it completes, fails, or is cancelled."""
    started = clock()
    body = request(
        client,
        api_key,
        "POST",
        AGENT_RUNS_URL,
        request_body,
        max_attempts=max_attempts,
        sleep=sleep,
    ).body
    run_id = _string(body, "id")
    if run_id is None:
        raise ValueError("the created run has no id")
    polls = 0
    while _string(body, "status") not in TERMINAL_STATUSES:
        if clock() - started > timeout_s:
            raise AgentTimeout(
                f"run {run_id} is still {_string(body, 'status')} after {timeout_s:g} s"
            )
        sleep(poll_s)
        body = request(
            client,
            api_key,
            "GET",
            f"{AGENT_RUNS_URL}/{run_id}",
            max_attempts=max_attempts,
            sleep=sleep,
        ).body
        polls += 1
    return RunResult(body, (clock() - started) * 1000, polls)


@dataclass(frozen=True)
class CachedRun:
    result: RunResult
    from_cache: bool  # True means no run was created (and nothing was spent) this time


def cached_run(
    cache_dir: Path,
    client: httpx.Client,
    api_key: str,
    request_body: Mapping[str, object],
    *,
    poll_s: float = 3.0,
    timeout_s: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> CachedRun:
    """The stored terminal run for this exact request, else create one and store it.

    Failed and cancelled runs are stored too, so a paid failure is reported rather than retried.
    """

    def compute() -> dict[str, object]:
        result = run_to_completion(
            client,
            api_key,
            request_body,
            poll_s=poll_s,
            timeout_s=timeout_s,
            sleep=sleep,
            clock=clock,
        )
        return {"run": result.body, "client_ms": result.client_ms, "polls": result.polls}

    record, from_cache = cached_record(
        cache_dir, cache_key({"agent_request": request_body}), compute
    )
    run, client_ms, polls = record["run"], record["client_ms"], record["polls"]
    if (
        not isinstance(run, dict)
        or not isinstance(client_ms, int | float)
        or not isinstance(polls, int)
    ):
        raise ValueError("stored agent run is malformed")
    return CachedRun(RunResult(run, float(client_ms), polls), from_cache)
