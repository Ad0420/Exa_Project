"""Tests for the Exa Agent request, client, and run parsing (no network: scripted fake server)."""

import json
from pathlib import Path

import httpx
import pytest

from exa_bench.agent import (
    AGENT_RUNS_URL,
    AgentCompany,
    AgentRun,
    AgentTimeout,
    agent_request,
    cached_run,
    parse_run,
    run_to_completion,
)

API_KEY = "test-key-do-not-leak"
CREATED = {"id": "agent_run_01", "status": "queued", "createdAt": "2026-05-07T18:31:00.000Z"}
RUNNING = {**CREATED, "status": "running"}
COMPLETED: dict[str, object] = {
    "id": "agent_run_01",
    "object": "agent_run",
    "status": "completed",
    "stopReason": "schema_satisfied",
    "createdAt": "2026-05-07T18:31:00.000Z",
    "completedAt": "2026-05-07T18:31:12.500Z",
    "output": {
        "text": "Two companies.",
        "structured": {
            "companies": [
                {"company": "Acme", "domain": "acme.com"},
                {"company": " Beta ", "domain": " beta.io "},
                {"company": "No domain", "domain": ""},
                {"company": "Bad", "domain": 5},
                "junk",
            ]
        },
        "grounding": [],
    },
    "usage": {"agentComputeUnits": 1, "searches": 4},
    "costDollars": {"total": 0.045, "agentCompute": 0.025, "search": 0.02},
}


def scripted_client(replies: list[httpx.Response], seen: list[httpx.Request]) -> httpx.Client:
    pending = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return pending.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_request_adds_the_ask_and_a_bounded_schema() -> None:
    body = agent_request("fintech startups in Singapore ", effort="low", k=5)

    assert body["effort"] == "low"
    query = body["query"]
    assert isinstance(query, str)
    assert query.startswith("fintech startups in Singapore\n\nReturn up to 5 companies")
    assert "homepage domain" in query
    schema = body["outputSchema"]
    assert isinstance(schema, dict)
    companies = schema["properties"]["companies"]
    assert companies["maxItems"] == 5
    assert companies["items"]["required"] == ["company", "domain"]
    with pytest.raises(ValueError, match="effort"):
        agent_request("q", effort="auto")
    with pytest.raises(ValueError, match="k must be"):
        agent_request("q", k=11)


def test_parse_run_reads_companies_cost_usage_and_timing() -> None:
    run = parse_run(COMPLETED)

    assert run == AgentRun(
        id="agent_run_01",
        status="completed",
        stop_reason="schema_satisfied",
        companies=(AgentCompany("Acme", "acme.com"), AgentCompany("Beta", "beta.io")),
        cost_dollars=0.045,
        searches=4,
        server_ms=12500.0,
    )


def test_parse_run_treats_missing_or_malformed_fields_as_absent() -> None:
    run = parse_run({"id": "x", "status": "failed", "createdAt": "not a date", "completedAt": "z"})

    assert (run.companies, run.cost_dollars, run.searches, run.server_ms) == ((), None, None, None)
    assert run.stop_reason is None
    assert parse_run({}).status == ""
    assert parse_run({"usage": {"searches": True}, "costDollars": {"total": "1"}}).searches is None
    assert parse_run({"costDollars": {"total": "1"}}).cost_dollars is None


def test_run_to_completion_creates_then_polls_until_terminal() -> None:
    seen: list[httpx.Request] = []
    replies = [httpx.Response(200, json=body) for body in (CREATED, RUNNING, COMPLETED)]
    sleeps: list[float] = []
    ticks = iter([0.0, 1.0, 5.0, 12.5])  # start, two timeout checks, finish

    with scripted_client(replies, seen) as client:
        result = run_to_completion(
            client,
            API_KEY,
            {"query": "q"},
            poll_s=4.0,
            sleep=sleeps.append,
            clock=lambda: next(ticks),
        )

    assert [(r.method, str(r.url)) for r in seen] == [
        ("POST", AGENT_RUNS_URL),
        ("GET", f"{AGENT_RUNS_URL}/agent_run_01"),
        ("GET", f"{AGENT_RUNS_URL}/agent_run_01"),
    ]
    assert json.loads(seen[0].content) == {"query": "q"}
    assert seen[1].content == b""
    assert all(r.headers["x-api-key"] == API_KEY for r in seen)
    assert sleeps == [4.0, 4.0]
    assert (result.body, result.polls, result.client_ms) == (COMPLETED, 2, 12500.0)


def test_failed_and_cancelled_runs_are_terminal() -> None:
    failed = {**CREATED, "status": "failed", "error": {"code": "SERVER_ERROR", "message": "x"}}
    with scripted_client([httpx.Response(200, json=failed)], []) as client:
        result = run_to_completion(client, API_KEY, {"query": "q"}, sleep=lambda _: None)

    assert (result.body["status"], result.polls) == ("failed", 0)


def test_run_to_completion_gives_up_after_the_timeout() -> None:
    replies = [httpx.Response(200, json=CREATED)] + [httpx.Response(200, json=RUNNING)] * 3
    ticks = iter([0.0, 100.0, 400.0, 1000.0])

    with (
        scripted_client(replies, []) as client,
        pytest.raises(AgentTimeout, match="still running after 900"),
    ):
        run_to_completion(
            client, API_KEY, {"query": "q"}, sleep=lambda _: None, clock=lambda: next(ticks)
        )


def test_run_to_completion_requires_a_run_id() -> None:
    with (
        scripted_client([httpx.Response(200, json={"status": "queued"})], []) as client,
        pytest.raises(ValueError, match="no id"),
    ):
        run_to_completion(client, API_KEY, {"query": "q"}, sleep=lambda _: None)


def test_cached_run_stores_terminal_runs_and_never_recreates_them(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    replies = [httpx.Response(200, json=CREATED), httpx.Response(200, json=COMPLETED)]
    with scripted_client(replies, seen) as client:
        first = cached_run(tmp_path, client, API_KEY, {"query": "q"}, sleep=lambda _: None)
        second = cached_run(tmp_path, client, API_KEY, {"query": "q"}, sleep=lambda _: None)

    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.result == first.result
    assert first.result.body == COMPLETED
    assert first.result.polls == 1
    assert len(seen) == 2
    (stored,) = tmp_path.glob("*.json")
    assert API_KEY not in stored.read_text()


def test_cached_run_keys_on_the_whole_request(tmp_path: Path) -> None:
    replies = [httpx.Response(200, json=COMPLETED), httpx.Response(200, json=COMPLETED)]
    with scripted_client(replies, []) as client:
        low = cached_run(tmp_path, client, API_KEY, {"query": "q", "effort": "low"})
        medium = cached_run(tmp_path, client, API_KEY, {"query": "q", "effort": "medium"})

    assert (low.from_cache, medium.from_cache) == (False, False)
    assert len(list(tmp_path.glob("*.json"))) == 2
