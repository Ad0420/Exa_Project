"""Tests for running the judges over a sample (fake judges, temp caches)."""

from pathlib import Path

from exa_bench.agreement import JudgedItem
from exa_bench.core.contents_cache import PageText
from exa_bench.crosscheck import SampleItem
from exa_bench.crosscheck_run import RunTotals, judge_sample
from exa_bench.fact_extraction import Extraction, PageFacts
from exa_bench.grader import ResultVerdict
from exa_bench.llm_grader import LlmVerdict

FACTS = PageFacts(
    founded_year=2019,
    employees=None,
    employees_range=None,
    hq_country="Germany",
    funding_total_usd=None,
    latest_round_name=None,
    latest_round_date=None,
    is_single_company_page=True,
)


class FakeExtractor:
    def __init__(self) -> None:
        self.users: list[str] = []

    def __call__(self, system: str, user: str) -> Extraction:
        self.users.append(user)
        return Extraction(FACTS, 700, 40)


class FakeJudge:
    def __init__(self) -> None:
        self.users: list[str] = []

    def __call__(self, system: str, user: str) -> LlmVerdict:
        self.users.append(user)
        return LlmVerdict("fine", 1.0, 1500, 20)


def item(url: str) -> SampleItem:
    return SampleItem(
        "q1",
        "german companies",
        {"country": {"eq": "Germany"}},
        url,
        "T",
        ResultVerdict.SATISFIES,
        (),
        {},
    )


def test_judges_only_items_with_text_and_counts_tokens(tmp_path: Path) -> None:
    sample = [item("https://a.test"), item("https://b.test"), item("https://c.test")]
    pages = {
        "https://a.test": PageText("alpha text", None),
        "https://b.test": PageText(None, {"id": "https://b.test", "status": "error"}),
    }
    extractor, judge = FakeExtractor(), FakeJudge()
    seen: list[tuple[int, int]] = []

    judged, totals = judge_sample(
        sample,
        pages,
        extractor=extractor,
        judge=judge,
        extraction_cache=tmp_path / "x",
        grader_cache=tmp_path / "g",
        model="m",
        temperature=0.0,
        progress=lambda n, total: seen.append((n, total)),
    )

    assert [j.has_text for j in judged] == [True, False, False]
    assert judged[0].facts == FACTS
    assert judged[0].grader is not None
    assert judged[0].grader.matches
    assert (judged[1].facts, judged[1].grader) == (None, None)
    assert (totals.items, totals.with_text) == (3, 1)
    assert (totals.extraction_calls, totals.grader_calls) == (1, 1)
    assert (totals.extraction_prompt_tokens, totals.grader_prompt_tokens) == (700, 1500)
    assert extractor.users == ["Page URL: https://a.test\n\nalpha text"]
    assert "Query: german companies" in judge.users[0]
    assert "Constraints: {'country': {'eq': 'Germany'}}" in judge.users[0]
    assert judge.users[0].endswith("\n\nalpha text")
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_second_run_is_served_from_the_caches(tmp_path: Path) -> None:
    sample = [item("https://a.test")]
    pages = {"https://a.test": PageText("alpha text", None)}

    def run(extractor: FakeExtractor, judge: FakeJudge) -> tuple[list[JudgedItem], RunTotals]:
        return judge_sample(
            sample,
            pages,
            extractor=extractor,
            judge=judge,
            extraction_cache=tmp_path / "x",
            grader_cache=tmp_path / "g",
            model="m",
            temperature=0.0,
        )

    run(FakeExtractor(), FakeJudge())
    extractor, judge = FakeExtractor(), FakeJudge()

    judged, totals = run(extractor, judge)

    assert (extractor.users, judge.users) == ([], [])
    assert (totals.extraction_calls, totals.grader_calls) == (0, 0)
    assert (totals.extraction_prompt_tokens, totals.grader_prompt_tokens) == (700, 1500)
    assert judged[0].facts == FACTS
