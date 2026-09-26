"""From a /search response to the results a user asked for: verdicts, exclusions, dedup, policy."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from exa_filters.constraints import Verdict
from exa_filters.evaluate import Filter, evaluate_filters
from exa_filters.response import company_entity, search_results


class Status(StrEnum):
    SATISFIES = "satisfies"  # every filter passes
    VIOLATES = "violates"  # at least one filter fails
    UNEVALUABLE = "unevaluable"  # nothing fails, but a needed field is missing


class NullPolicy(StrEnum):
    STRICT = "strict"  # return only results that satisfy every filter
    LENIENT = "lenient"  # also return results with missing fields, flagged as such


@dataclass(frozen=True)
class FilteredResult:
    rank: int  # 1-based position in Exa's response
    url: str
    title: str
    entity_id: str | None
    properties: Mapping[str, object]
    verdicts: tuple[Verdict, ...]  # one per filter, in filter order

    @property
    def status(self) -> Status:
        if Verdict.FAIL in self.verdicts:
            return Status.VIOLATES
        if Verdict.UNKNOWN in self.verdicts:
            return Status.UNEVALUABLE
        return Status.SATISFIES

    @property
    def unknown_fields(self) -> int:
        return sum(v is Verdict.UNKNOWN for v in self.verdicts)


def parse_results(body: Mapping[str, object], filters: Sequence[Filter]) -> list[FilteredResult]:
    """Every result with a URL, in rank order, judged against `filters`.

    A result without a company entity is judged against empty properties, so every filter
    reads UNKNOWN for it.
    """
    parsed: list[FilteredResult] = []
    for rank, result in enumerate(search_results(body), start=1):
        if not isinstance(result, Mapping) or not isinstance(result.get("url"), str):
            continue
        entity = company_entity(result)
        properties: Mapping[str, object] = entity.properties if entity is not None else {}
        title = result.get("title")
        parsed.append(
            FilteredResult(
                rank=rank,
                url=str(result["url"]),
                title=title if isinstance(title, str) else "",
                entity_id=entity.id if entity is not None else None,
                properties=properties,
                verdicts=tuple(evaluate_filters(filters, properties)),
            )
        )
    return parsed


def canonical_host(text: str) -> str | None:
    """The host of a URL or bare domain, lower-cased and without a leading "www."."""
    candidate = text.strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = "https://" + candidate
    host = urlsplit(candidate).hostname or ""
    host = host.removeprefix("www.")
    return host or None


def exclude(
    results: Iterable[FilteredResult], exclude_entities: Iterable[str]
) -> tuple[list[FilteredResult], int]:
    """Drop results whose entity id or canonical host matches an exclusion: (kept, dropped)."""
    ids = {item for item in exclude_entities if item.strip()}
    hosts = {host for item in exclude_entities if (host := canonical_host(item)) is not None}
    kept: list[FilteredResult] = []
    dropped = 0
    for result in results:
        if result.entity_id in ids or canonical_host(result.url) in hosts:
            dropped += 1
        else:
            kept.append(result)
    return kept, dropped


def dedupe(results: Iterable[FilteredResult]) -> tuple[list[FilteredResult], int]:
    """Keep the first occurrence of each company, by entity id or, failing that, by host."""
    seen: set[str] = set()
    kept: list[FilteredResult] = []
    dropped = 0
    for result in results:
        key = result.entity_id or canonical_host(result.url) or result.url
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            kept.append(result)
    return kept, dropped


def accepted_by(policy: NullPolicy, result: FilteredResult) -> bool:
    if policy is NullPolicy.STRICT:
        return result.status is Status.SATISFIES
    return result.status is not Status.VIOLATES


@dataclass(frozen=True)
class Selection:
    results: list[FilteredResult]  # the accepted results, first k in rank order
    seen: int  # results parsed from the response
    excluded: int
    duplicates: int
    violating: int
    unevaluable: int  # rejected under the strict policy; counted, never hidden
    short_by: int  # how many fewer than k were accepted


def select(
    body: Mapping[str, object],
    filters: Sequence[Filter],
    *,
    k: int,
    null_policy: NullPolicy = NullPolicy.STRICT,
    exclude_entities: Iterable[str] = (),
) -> Selection:
    """The full pipeline for one response: parse, exclude, dedupe, apply the policy, take k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    parsed = parse_results(body, filters)
    kept, excluded = exclude(parsed, exclude_entities)
    unique, duplicates = dedupe(kept)
    accepted = [r for r in unique if accepted_by(null_policy, r)]
    violating = sum(r.status is Status.VIOLATES for r in unique)
    unevaluable = sum(r.status is Status.UNEVALUABLE for r in unique)
    return Selection(
        results=accepted[:k],
        seen=len(parsed),
        excluded=excluded,
        duplicates=duplicates,
        violating=violating,
        unevaluable=unevaluable,
        short_by=max(0, k - len(accepted)),
    )
