"""Run both judges over the cross-check sample, through their caches."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from exa_bench.agreement import JudgedItem
from exa_bench.contents_cache import PageText
from exa_bench.crosscheck import SampleItem
from exa_bench.fact_extraction import Extractor, build_extraction_prompt, cached_extraction
from exa_bench.llm_grader import Judge, build_prompt, cached_judgement


@dataclass
class RunTotals:
    items: int = 0
    with_text: int = 0
    extraction_calls: int = 0  # judged this run (not served from cache)
    grader_calls: int = 0
    extraction_prompt_tokens: int = 0  # over every judged item, cached or not
    extraction_completion_tokens: int = 0
    grader_prompt_tokens: int = 0
    grader_completion_tokens: int = 0


def judge_sample(
    sample: Sequence[SampleItem],
    pages: Mapping[str, PageText],
    *,
    extractor: Extractor,
    judge: Judge,
    extraction_cache: Path,
    grader_cache: Path,
    model: str,
    temperature: float,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[JudgedItem], RunTotals]:
    """Extract facts and run Exa's grader for every sampled item that has page text.

    Items without page text get neither judgement: their pages could not be read, so
    nothing about them can be checked against a page.
    """
    judged: list[JudgedItem] = []
    totals = RunTotals(items=len(sample))
    for number, item in enumerate(sample, start=1):
        page = pages.get(item.url)
        text = page.text if page is not None else None
        if not text:
            judged.append(JudgedItem(item, has_text=False, facts=None, grader=None))
        else:
            totals.with_text += 1
            system, user = build_extraction_prompt(item.url, text)
            extracted = cached_extraction(
                extraction_cache, extractor, system, user, model=model, temperature=temperature
            )
            totals.extraction_calls += not extracted.from_cache
            totals.extraction_prompt_tokens += extracted.extraction.prompt_tokens or 0
            totals.extraction_completion_tokens += extracted.extraction.completion_tokens or 0

            system, user = build_prompt(
                item.query_text, item.constraints, item.url, item.title, text
            )
            graded = cached_judgement(
                grader_cache, judge, system, user, model=model, temperature=temperature
            )
            totals.grader_calls += not graded.from_cache
            totals.grader_prompt_tokens += graded.verdict.prompt_tokens or 0
            totals.grader_completion_tokens += graded.verdict.completion_tokens or 0
            judged.append(
                JudgedItem(
                    item, has_text=True, facts=extracted.extraction.facts, grader=graded.verdict
                )
            )
        if progress is not None:
            progress(number, len(sample))
    return judged, totals
