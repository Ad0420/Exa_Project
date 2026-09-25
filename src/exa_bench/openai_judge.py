"""OpenAI-backed structured calls: the ported Exa grader and the fact extractor use it."""

from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from exa_bench.llm_grader import LlmVerdict

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None


class RetrievalGradeResult(BaseModel):
    """The output schema of Exa's RetrievalGrader (same field names and order)."""

    explanation: str
    score: float


class OpenAIStructured:
    """One chat completion parsed into a pydantic schema, at a fixed model and temperature."""

    def __init__(self, client: OpenAI, model: str, temperature: float) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def parse(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        completion = self.client.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("model returned no parsed output (refusal or empty content)")
        usage = completion.usage
        return parsed, Usage(
            usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None
        )


class OpenAIJudge:
    """Exa's grader call: `Judge` implemented with OpenAI structured output."""

    def __init__(self, structured: OpenAIStructured) -> None:
        self.structured = structured

    def __call__(self, system: str, user: str) -> LlmVerdict:
        result, usage = self.structured.parse(system, user, RetrievalGradeResult)
        return LlmVerdict(
            explanation=result.explanation,
            score=result.score,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
