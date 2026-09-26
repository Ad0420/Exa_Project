"""Tests for the OpenAI wrapper against a fake HTTP server (no network).

The openai SDK uses httpx2 (httpx 2.x), separate from the httpx our Exa client uses.
"""

import json

import httpx2
import pytest
from openai import OpenAI

from exa_bench.fact_extraction import PageFacts
from exa_bench.openai_judge import OpenAIExtractor, OpenAIJudge, OpenAIStructured, Usage


def completion(content: str | None, usage: bool = True) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.4",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}
    return body


def fake_client(reply: dict[str, object], seen: list[httpx2.Request]) -> OpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=reply)

    return OpenAI(
        api_key="test-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )


def test_judge_sends_exas_settings_and_parses_the_verdict() -> None:
    seen: list[httpx2.Request] = []
    client = fake_client(completion(json.dumps({"explanation": "matches", "score": 1.0})), seen)

    verdict = OpenAIJudge(OpenAIStructured(client, "gpt-5.4", 0.0))("SYS", "USER")

    (request,) = seen
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == "Bearer test-key"
    sent = json.loads(request.content)
    assert (sent["model"], sent["temperature"]) == ("gpt-5.4", 0.0)
    assert sent["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "RetrievalGradeResult"
    assert (verdict.explanation, verdict.score, verdict.matches) == ("matches", 1.0, True)
    assert (verdict.prompt_tokens, verdict.completion_tokens) == (1200, 30)


def test_structured_parse_returns_schema_and_handles_missing_usage() -> None:
    facts = {
        "founded_year": 2019,
        "employees": None,
        "employees_range": "51-200",
        "hq_country": "Germany",
        "funding_total_usd": 25000000,
        "latest_round_name": "Series B",
        "latest_round_date": "2024-03",
        "is_single_company_page": True,
    }
    client = fake_client(completion(json.dumps(facts), usage=False), [])

    parsed, usage = OpenAIStructured(client, "m", 0.0).parse("S", "U", PageFacts)

    assert parsed == PageFacts.model_validate(facts)
    assert usage == Usage(None, None)


def test_extractor_parses_page_facts_with_usage() -> None:
    facts = {
        "founded_year": None,
        "employees": 80,
        "employees_range": None,
        "hq_country": "Germany",
        "funding_total_usd": None,
        "latest_round_name": None,
        "latest_round_date": None,
        "is_single_company_page": True,
    }
    seen: list[httpx2.Request] = []
    client = fake_client(completion(json.dumps(facts)), seen)

    extraction = OpenAIExtractor(OpenAIStructured(client, "m", 0.0))("S", "U")

    assert extraction.facts == PageFacts.model_validate(facts)
    assert (extraction.prompt_tokens, extraction.completion_tokens) == (1200, 30)
    assert json.loads(seen[0].content)["response_format"]["json_schema"]["name"] == "PageFacts"


def test_empty_content_is_an_error() -> None:
    client = fake_client(completion(None), [])

    with pytest.raises(RuntimeError, match="no parsed output"):
        OpenAIStructured(client, "m", 0.0).parse("S", "U", PageFacts)
