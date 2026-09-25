"""Tests for the ported LLM grader: prompt fidelity, prompt building, and the verdict cache."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from exa_bench.llm_grader import (
    EXA_GRADER_SOURCE_SHA256,
    MATCH_THRESHOLD,
    MAX_TEXT_CHARS,
    RETRIEVAL_GRADING_SYSTEM,
    RETRIEVAL_GRADING_USER,
    LlmVerdict,
    build_prompt,
    cached_judgement,
)

FIXTURE = Path(__file__).parent / "fixtures" / "exa_benchmarks_retrieval_grader.py"


def exa_source_constants() -> dict[str, str]:
    """String constants assigned at module level in Exa's grader, read without importing it."""
    constants: dict[str, str] = {}
    for node in ast.parse(FIXTURE.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if isinstance(value, str):
                constants[node.targets[0].id] = value  # type: ignore[attr-defined]
    return constants


def test_vendored_grader_is_the_pinned_commit() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXA_GRADER_SOURCE_SHA256


def test_prompts_match_exas_grader_character_for_character() -> None:
    theirs = exa_source_constants()

    assert theirs["RETRIEVAL_GRADING_SYSTEM"] == RETRIEVAL_GRADING_SYSTEM
    assert theirs["RETRIEVAL_GRADING_USER"] == RETRIEVAL_GRADING_USER


def test_text_budget_and_threshold_match_exas_grader() -> None:
    source = FIXTURE.read_text()

    assert f"result.content[:{MAX_TEXT_CHARS}]" in source
    assert f"parsed.score >= {MATCH_THRESHOLD}" in source


def test_build_prompt_fills_the_template_like_exa_does() -> None:
    system, user = build_prompt(
        "fintech in Singapore", {"country": {"eq": "Singapore"}}, "https://a.test", "A", "body text"
    )

    assert system == RETRIEVAL_GRADING_SYSTEM
    assert user == (
        "Query: fintech in Singapore\nConstraints: {'country': {'eq': 'Singapore'}}\n\n"
        "Result URL: https://a.test\nTitle: A\n\nbody text"
    )


@pytest.mark.parametrize("text", [None, ""])
def test_missing_text_becomes_no_content(text: str | None) -> None:
    _, user = build_prompt("q", {}, "https://a.test", "A", text)

    assert user.endswith("\n\n(no content)")


def test_text_is_truncated_to_the_budget() -> None:
    _, user = build_prompt("q", {}, "https://a.test", "A", "x" * (MAX_TEXT_CHARS + 500))

    assert user.endswith("x" * MAX_TEXT_CHARS)
    assert len(user) < MAX_TEXT_CHARS + 200


def test_match_threshold() -> None:
    assert LlmVerdict("", 0.5, None, None).matches
    assert not LlmVerdict("", 0.49, None, None).matches


class FakeJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> LlmVerdict:
        self.calls.append((system, user))
        return LlmVerdict("looks right", 1.0, 1200, 30)


def test_judgement_is_cached_per_prompt_and_model(tmp_path: Path) -> None:
    judge = FakeJudge()

    first = cached_judgement(tmp_path, judge, "sys", "user one")
    again = cached_judgement(tmp_path, judge, "sys", "user one")
    other = cached_judgement(tmp_path, judge, "sys", "user two")
    other_model = cached_judgement(tmp_path, judge, "sys", "user one", model="other")

    assert (first.from_cache, again.from_cache, other.from_cache, other_model.from_cache) == (
        False,
        True,
        False,
        False,
    )
    assert again.verdict == first.verdict == LlmVerdict("looks right", 1.0, 1200, 30)
    assert len(judge.calls) == 3


def test_cache_file_records_model_and_verdict_and_rejects_tampering(tmp_path: Path) -> None:
    cached_judgement(tmp_path, FakeJudge(), "sys", "user")
    (cached,) = tmp_path.glob("*.json")
    record = json.loads(cached.read_text())

    assert record["model"] == "gpt-5.4"
    assert record["verdict"]["score"] == 1.0

    record["key"] = "tampered"
    cached.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        cached_judgement(tmp_path, FakeJudge(), "sys", "user")
