"""Exa's own LLM retrieval grader, ported from exa-labs/benchmarks, with a disk cache.

The prompts, text budget, and match threshold are copied from
shared/shared/graders/retrieval.py at the pinned benchmark commit; a test checks them
character for character against a vendored copy. The model call itself is behind
`Judge`, so everything here runs without network.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from exa_bench.core.json_cache import cache_key, cached_record

# shared/shared/graders/retrieval.py at exa-labs/benchmarks c096f1a.
EXA_GRADER_SOURCE_SHA256 = "4c1716d6af741c699fccfe1530484f45c5017652c28e1bdb286dae6070dc8e89"
EXA_GRADER_MODEL = "gpt-5.4"  # BaseLLMGrader default at that commit
EXA_GRADER_TEMPERATURE = 0.0
MAX_TEXT_CHARS = 30000  # their grader passes result.content[:30000]
MATCH_THRESHOLD = 0.5  # their grader counts score >= 0.5 as a match

RETRIEVAL_GRADING_SYSTEM = """You are evaluating if a search result matches a company search query.
This is BINARY - score 1 if the result matches, score 0 if it doesn't.

AUTOMATIC SCORE 0 (no exceptions):
1. Job listing pages (URLs with /jobs/, /careers/) -> Score 0
2. News articles about the company (not the company's own page) -> Score 0
3. If content is empty/missing and cannot verify the company -> Score 0

For queries with constraints (industry, geography, founding year, etc.):
- Score 1 if the result is about a company that matches ALL query constraints
- For industry/geo queries: company must be in the specified industry AND location
- For founded_year queries: company must be founded in the specified year
- For employee_count queries: company has approximately the specified count (within 20% tolerance)
- For funding queries: company matches the funding stage or amount criteria

Score 0 if:
- The result doesn't match ANY of the constraints
- The result is not about a company
- Cannot verify the company matches from available content

Be strict about matching ALL constraints. Partial matches = 0.
When genuinely uncertain about a close match, lean toward score 1 if the core criteria align."""

RETRIEVAL_GRADING_USER = """Query: {query}
Constraints: {constraints}

Result URL: {url}
Title: {title}

{text}"""


@dataclass(frozen=True)
class LlmVerdict:
    explanation: str
    score: float
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def matches(self) -> bool:
        return self.score >= MATCH_THRESHOLD


class Judge(Protocol):
    """Runs one grading prompt through a model. Implemented by the OpenAI wrapper and by fakes."""

    def __call__(self, system: str, user: str) -> LlmVerdict: ...


def build_prompt(
    query: str, constraints: Mapping[str, object], url: str, title: str, text: str | None
) -> tuple[str, str]:
    """The (system, user) messages exactly as Exa's grader builds them."""
    user = RETRIEVAL_GRADING_USER.format(
        query=query,
        constraints=dict(constraints),
        url=url,
        title=title,
        text=text[:MAX_TEXT_CHARS] if text else "(no content)",
    )
    return RETRIEVAL_GRADING_SYSTEM, user


@dataclass(frozen=True)
class CachedJudgement:
    verdict: LlmVerdict
    from_cache: bool


def cached_judgement(
    cache_dir: Path,
    judge: Judge,
    system: str,
    user: str,
    *,
    model: str = EXA_GRADER_MODEL,
    temperature: float = EXA_GRADER_TEMPERATURE,
) -> CachedJudgement:
    """Return the stored verdict for this exact prompt and model, or ask the judge and store it."""
    key = cache_key({"model": model, "temperature": temperature, "system": system, "user": user})

    def compute() -> dict[str, object]:
        return {"model": model, "temperature": temperature, "verdict": asdict(judge(system, user))}

    record, from_cache = cached_record(cache_dir, key, compute)
    verdict = record["verdict"]
    assert isinstance(verdict, dict)
    return CachedJudgement(LlmVerdict(**verdict), from_cache=from_cache)
