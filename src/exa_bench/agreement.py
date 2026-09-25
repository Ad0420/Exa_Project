"""Aggregate the cross-check: typed fields vs page facts, and our verdicts vs Exa's grader."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from exa_bench.crosscheck import SampleItem
from exa_bench.fact_extraction import PageFacts
from exa_bench.field_check import FIELDS, FieldCheck, compare_all, page_properties
from exa_bench.grader import Outcome, ResultVerdict, grade
from exa_bench.llm_grader import LlmVerdict
from exa_bench.stats import DEFAULT_RESAMPLES, Cluster, Rate, rate

# Phrases in a grader explanation that mean "the page did not let me decide" rather than
# "the page shows a mismatch". A heuristic; the count is reported alongside the raw numbers.
CANNOT_VERIFY_PHRASES = (
    "cannot verify",
    "can't verify",
    "unable to verify",
    "cannot confirm",
    "can't confirm",
    "no information",
    "not mentioned",
    "does not mention",
    "doesn't mention",
    "no mention",
    "not enough information",
    "insufficient information",
    "does not provide",
    "doesn't provide",
    "not stated",
    "not specified",
    "no content",
)


@dataclass(frozen=True)
class JudgedItem:
    item: SampleItem
    has_text: bool
    facts: PageFacts | None  # None when no page text or extraction was skipped
    grader: LlmVerdict | None  # None when Exa's grader was skipped


class Verification(StrEnum):
    CONFIRMED = "confirmed"  # the page's version of the facts fails the constraint too
    CONTRADICTED = "contradicted"  # the page's version passes it: our violation may be wrong
    UNVERIFIABLE = "unverifiable"  # the page does not state the fact


@dataclass(frozen=True)
class FieldStats:
    agrees: int
    disagrees: int
    not_stated: int
    no_typed_value: int
    accuracy: Rate  # agrees / (agrees + disagrees), clustered by query


@dataclass(frozen=True)
class OutcomeVerification:
    confirmed: int
    contradicted: int
    unverifiable: int
    confirmed_rate: Rate  # confirmed / (confirmed + contradicted), clustered by query


@dataclass(frozen=True)
class ViolatingItems:
    confirmed: int  # at least one failing constraint confirmed by the page
    contradicted: int  # every failing constraint contradicted by the page
    mixed: int  # some contradicted, the rest unverifiable
    unverifiable: int


@dataclass(frozen=True)
class SatisfyingItems:
    undisputed: int  # the page's version passes every constraint too
    disputed: int  # the page's version fails at least one
    unverifiable: int


@dataclass(frozen=True)
class UnevaluableItems:
    resolved_satisfies: int  # the page states the missing facts and they pass
    resolved_violates: int
    still_unknown: int


@dataclass(frozen=True)
class GraderStats:
    compared: int  # items we called violates or satisfies that the grader judged
    ours_satisfies_grader_matches: int
    ours_satisfies_grader_rejects: int
    ours_violates_grader_matches: int
    ours_violates_grader_rejects: int
    agreement: Rate
    kappa: float | None
    cannot_verify: int  # grader explanations that read as "could not decide"
    agreement_excluding_cannot_verify: Rate
    unevaluable_compared: int
    unevaluable_grader_matches: int


@dataclass(frozen=True)
class CrossCheck:
    items: int
    with_text: int
    with_facts: int
    single_company_pages: int
    fields: dict[str, FieldStats]
    outcome_verification: dict[str, OutcomeVerification]  # by field, plus "all"
    violating_items: ViolatingItems
    satisfying_items: SatisfyingItems
    unevaluable_items: UnevaluableItems
    grader: GraderStats
    bootstrap_seed: int


def cross_check(
    judged: Sequence[JudgedItem], *, seed: int, resamples: int = DEFAULT_RESAMPLES
) -> CrossCheck:
    return CrossCheck(
        items=len(judged),
        with_text=sum(j.has_text for j in judged),
        with_facts=sum(j.facts is not None for j in judged),
        single_company_pages=sum(bool(j.facts and j.facts.is_single_company_page) for j in judged),
        fields=_field_stats(judged, seed, resamples),
        outcome_verification=_outcome_verification(judged, seed, resamples),
        violating_items=_violating_items(judged),
        satisfying_items=_satisfying_items(judged),
        unevaluable_items=_unevaluable_items(judged),
        grader=_grader_stats(judged, seed, resamples),
        bootstrap_seed=seed,
    )


def page_outcomes(judged: JudgedItem) -> dict[tuple[str, str], Outcome]:
    """The grader's outcomes for the page's version of the facts, by (key, op)."""
    graded = grade(judged.item.constraints, page_properties(judged.facts))
    return {(o.key, o.op): o.outcome for o in graded.outcomes}


def verify_outcome(judged: JudgedItem, key: str, op: str) -> Verification:
    """How the page bears on one failing constraint of ours."""
    if judged.facts is None:
        return Verification.UNVERIFIABLE
    match page_outcomes(judged).get((key, op), Outcome.UNKNOWN):
        case Outcome.PASS:
            return Verification.CONTRADICTED
        case Outcome.FAIL:
            return Verification.CONFIRMED
        case _:
            return Verification.UNVERIFIABLE


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float | None:
    """Agreement beyond chance between two binary raters; None when undefined."""
    if len(first) != len(second) or not first:
        return None
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / len(first)
    p_first, p_second = sum(first) / len(first), sum(second) / len(second)
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def reads_as_cannot_verify(explanation: str) -> bool:
    text = explanation.casefold()
    return any(phrase in text for phrase in CANNOT_VERIFY_PHRASES)


def _field_stats(judged: Sequence[JudgedItem], seed: int, resamples: int) -> dict[str, FieldStats]:
    counts: dict[str, Counter[FieldCheck]] = {field: Counter() for field in FIELDS}
    clusters: dict[str, defaultdict[str, list[int]]] = {
        field: defaultdict(lambda: [0, 0]) for field in FIELDS
    }
    for j in judged:
        for field, check in compare_all(j.item.typed, j.facts).items():
            counts[field][check] += 1
            if check in (FieldCheck.AGREES, FieldCheck.DISAGREES):
                cluster = clusters[field][j.item.query_id]
                cluster[0] += check is FieldCheck.AGREES
                cluster[1] += 1
    return {
        field: FieldStats(
            agrees=counts[field][FieldCheck.AGREES],
            disagrees=counts[field][FieldCheck.DISAGREES],
            not_stated=counts[field][FieldCheck.NOT_STATED],
            no_typed_value=counts[field][FieldCheck.NO_TYPED_VALUE],
            accuracy=rate(_as_clusters(clusters[field]), seed=seed, resamples=resamples),
        )
        for field in FIELDS
    }


def _outcome_verification(
    judged: Sequence[JudgedItem], seed: int, resamples: int
) -> dict[str, OutcomeVerification]:
    counts: defaultdict[str, Counter[Verification]] = defaultdict(Counter)
    clusters: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for j in judged:
        for outcome in j.item.outcomes:
            if outcome.outcome is not Outcome.FAIL:
                continue
            verification = verify_outcome(j, outcome.key, outcome.op)
            for name in (outcome.key, "all"):
                counts[name][verification] += 1
                if verification is not Verification.UNVERIFIABLE:
                    cluster = clusters[name][j.item.query_id]
                    cluster[0] += verification is Verification.CONFIRMED
                    cluster[1] += 1
    return {
        name: OutcomeVerification(
            confirmed=counts[name][Verification.CONFIRMED],
            contradicted=counts[name][Verification.CONTRADICTED],
            unverifiable=counts[name][Verification.UNVERIFIABLE],
            confirmed_rate=rate(_as_clusters(clusters[name]), seed=seed, resamples=resamples),
        )
        for name in sorted(counts, key=lambda n: (n != "all", n))
    }


def _violating_items(judged: Sequence[JudgedItem]) -> ViolatingItems:
    tally: Counter[str] = Counter()
    for j in judged:
        if j.item.verdict is not ResultVerdict.VIOLATES:
            continue
        results = {
            verify_outcome(j, o.key, o.op) for o in j.item.outcomes if o.outcome is Outcome.FAIL
        }
        if Verification.CONFIRMED in results:
            tally["confirmed"] += 1
        elif results == {Verification.CONTRADICTED}:
            tally["contradicted"] += 1
        elif Verification.CONTRADICTED in results:
            tally["mixed"] += 1
        else:
            tally["unverifiable"] += 1
    return ViolatingItems(
        tally["confirmed"], tally["contradicted"], tally["mixed"], tally["unverifiable"]
    )


def _satisfying_items(judged: Sequence[JudgedItem]) -> SatisfyingItems:
    tally: Counter[str] = Counter()
    for j in judged:
        if j.item.verdict is not ResultVerdict.SATISFIES:
            continue
        outcomes = set(page_outcomes(j).values()) if j.facts is not None else set()
        if Outcome.FAIL in outcomes:
            tally["disputed"] += 1
        elif outcomes and outcomes <= {Outcome.PASS}:
            tally["undisputed"] += 1
        else:
            tally["unverifiable"] += 1
    return SatisfyingItems(tally["undisputed"], tally["disputed"], tally["unverifiable"])


def _unevaluable_items(judged: Sequence[JudgedItem]) -> UnevaluableItems:
    tally: Counter[ResultVerdict] = Counter()
    for j in judged:
        if j.item.verdict is not ResultVerdict.UNEVALUABLE:
            continue
        page_verdict = (
            grade(j.item.constraints, page_properties(j.facts)).verdict
            if j.facts is not None
            else ResultVerdict.UNEVALUABLE
        )
        tally[page_verdict] += 1
    return UnevaluableItems(
        resolved_satisfies=tally[ResultVerdict.SATISFIES],
        resolved_violates=tally[ResultVerdict.VIOLATES],
        still_unknown=tally[ResultVerdict.UNEVALUABLE] + tally[ResultVerdict.NOT_CHECKABLE],
    )


def _grader_stats(judged: Sequence[JudgedItem], seed: int, resamples: int) -> GraderStats:
    cells: Counter[tuple[ResultVerdict, bool]] = Counter()
    ours: list[bool] = []
    theirs: list[bool] = []
    clusters: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    decided: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    cannot_verify = 0
    unevaluable = [0, 0]
    for j in judged:
        if j.grader is None:
            continue
        if j.item.verdict is ResultVerdict.UNEVALUABLE:
            unevaluable[0] += 1
            unevaluable[1] += j.grader.matches
            continue
        if j.item.verdict not in (ResultVerdict.VIOLATES, ResultVerdict.SATISFIES):
            continue
        we_say_match = j.item.verdict is ResultVerdict.SATISFIES
        cells[(j.item.verdict, j.grader.matches)] += 1
        ours.append(we_say_match)
        theirs.append(j.grader.matches)
        agree = we_say_match == j.grader.matches
        clusters[j.item.query_id][0] += agree
        clusters[j.item.query_id][1] += 1
        if reads_as_cannot_verify(j.grader.explanation):
            cannot_verify += 1
        else:
            decided[j.item.query_id][0] += agree
            decided[j.item.query_id][1] += 1
    return GraderStats(
        compared=len(ours),
        ours_satisfies_grader_matches=cells[(ResultVerdict.SATISFIES, True)],
        ours_satisfies_grader_rejects=cells[(ResultVerdict.SATISFIES, False)],
        ours_violates_grader_matches=cells[(ResultVerdict.VIOLATES, True)],
        ours_violates_grader_rejects=cells[(ResultVerdict.VIOLATES, False)],
        agreement=rate(_as_clusters(clusters), seed=seed, resamples=resamples),
        kappa=cohen_kappa(ours, theirs),
        cannot_verify=cannot_verify,
        agreement_excluding_cannot_verify=rate(
            _as_clusters(decided), seed=seed, resamples=resamples
        ),
        unevaluable_compared=unevaluable[0],
        unevaluable_grader_matches=unevaluable[1],
    )


def _as_clusters(by_query: dict[str, list[int]]) -> list[Cluster]:
    return [(hits, total) for hits, total in by_query.values()]
