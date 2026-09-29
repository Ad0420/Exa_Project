"""Tests for cross-check aggregation, checked against hand-computed answers."""

import pytest

from exa_bench.constraints.grader import ConstraintOutcome, Outcome, ResultVerdict
from exa_bench.crosscheck.agreement import (
    CrossCheck,
    JudgedItem,
    Verification,
    cohen_kappa,
    cross_check,
    reads_as_cannot_verify,
    verify_outcome,
)
from exa_bench.crosscheck.fact_extraction import PageFacts
from exa_bench.crosscheck.llm_grader import LlmVerdict
from exa_bench.crosscheck.sample import SampleItem, typed_fields

RESAMPLES = 100


def facts(**overrides: object) -> PageFacts:
    base: dict[str, object] = {
        "founded_year": None,
        "employees": None,
        "employees_range": None,
        "hq_country": None,
        "funding_total_usd": None,
        "latest_round_name": None,
        "latest_round_date": None,
        "is_single_company_page": True,
    }
    return PageFacts.model_validate(base | overrides)


def item(
    query_id: str,
    constraints: dict[str, object],
    verdict: ResultVerdict,
    outcomes: list[tuple[str, str, Outcome]],
    employees: int | None = None,
    country: str | None = None,
) -> SampleItem:
    properties: dict[str, object] = {
        "workforce": {"total": employees},
        "headquarters": {"country": country},
    }
    return SampleItem(
        query_id,
        "text",
        constraints,
        f"https://{query_id}.test",
        "",
        verdict,
        tuple(ConstraintOutcome(k, op, o) for k, op, o in outcomes),
        typed_fields(properties),
    )


def grader(matches: bool, explanation: str = "decided") -> LlmVerdict:
    return LlmVerdict(explanation, 1.0 if matches else 0.0, None, None)


EMP_LTE_100: dict[str, object] = {"employees": {"lte": 100}}
DE: dict[str, object] = {"country": {"eq": "Germany"}}

# 1. We said violates (120 > 100); the page says 150: confirmed.
CONFIRMED = JudgedItem(
    item("q1", EMP_LTE_100, ResultVerdict.VIOLATES, [("employees", "lte", Outcome.FAIL)], 120),
    True,
    facts(employees=150),
    grader(False),
)
# 2. We said violates; the page says 80: contradicted (our typed value looks wrong).
CONTRADICTED = JudgedItem(
    item("q1", EMP_LTE_100, ResultVerdict.VIOLATES, [("employees", "lte", Outcome.FAIL)], 120),
    True,
    facts(employees=80),
    grader(True, "The site lists about 80 staff."),
)
# 3. We said violates; there is text but extraction yielded nothing: unverifiable.
SILENT = JudgedItem(
    item("q2", EMP_LTE_100, ResultVerdict.VIOLATES, [("employees", "lte", Outcome.FAIL)], 120),
    True,
    None,
    grader(False, "Cannot verify headcount from the page."),
)
# 4. We said satisfies (Germany); the page agrees.
UNDISPUTED = JudgedItem(
    item("q3", DE, ResultVerdict.SATISFIES, [("country", "eq", Outcome.PASS)], country="DE"),
    True,
    facts(hq_country="Germany"),
    grader(True),
)
# 5. We said satisfies; the page says France: disputed.
DISPUTED = JudgedItem(
    item("q3", DE, ResultVerdict.SATISFIES, [("country", "eq", Outcome.PASS)], country="Germany"),
    True,
    facts(hq_country="France"),
    grader(False),
)
# 6. We could not evaluate; the page resolves it to violates.
RESOLVED = JudgedItem(
    item("q4", EMP_LTE_100, ResultVerdict.UNEVALUABLE, [("employees", "lte", Outcome.UNKNOWN)]),
    True,
    facts(employees=300),
    grader(False),
)
# 7. No page text at all.
NO_TEXT = JudgedItem(
    item("q5", EMP_LTE_100, ResultVerdict.VIOLATES, [("employees", "lte", Outcome.FAIL)], 120),
    False,
    None,
    None,
)
# 8. Two failing constraints: the page contradicts one and is silent on the other: mixed.
MIXED = JudgedItem(
    item(
        "q6",
        {**EMP_LTE_100, **DE},
        ResultVerdict.VIOLATES,
        [("employees", "lte", Outcome.FAIL), ("country", "eq", Outcome.FAIL)],
        120,
        "France",
    ),
    True,
    facts(employees=80),
    grader(True),
)
JUDGED = [CONFIRMED, CONTRADICTED, SILENT, UNDISPUTED, DISPUTED, RESOLVED, NO_TEXT, MIXED]


@pytest.fixture(scope="module")
def result() -> CrossCheck:
    return cross_check(JUDGED, seed=1, resamples=RESAMPLES)


def test_verify_outcome_categories() -> None:
    assert verify_outcome(CONFIRMED, "employees", "lte") is Verification.CONFIRMED
    assert verify_outcome(CONTRADICTED, "employees", "lte") is Verification.CONTRADICTED
    assert verify_outcome(SILENT, "employees", "lte") is Verification.UNVERIFIABLE
    assert verify_outcome(NO_TEXT, "employees", "lte") is Verification.UNVERIFIABLE


def test_counts_and_page_coverage(result: CrossCheck) -> None:
    assert (result.items, result.with_text, result.with_facts) == (8, 7, 6)
    assert result.single_company_pages == 6


def test_field_stats(result: CrossCheck) -> None:
    employees = result.fields["employees"]
    # Typed 120 vs page 150 (within 20%: agrees), vs 80 twice (disagrees), two without facts,
    # and three items with no typed headcount.
    assert (employees.agrees, employees.disagrees, employees.not_stated) == (1, 2, 2)
    assert employees.no_typed_value == 3
    assert employees.accuracy.value == pytest.approx(1 / 3)
    country = result.fields["country"]
    assert (country.agrees, country.disagrees, country.not_stated) == (1, 1, 1)


def test_outcome_verification(result: CrossCheck) -> None:
    employees = result.outcome_verification["employees"]
    assert (employees.confirmed, employees.contradicted, employees.unverifiable) == (1, 2, 2)
    assert employees.confirmed_rate.value == pytest.approx(1 / 3)
    country = result.outcome_verification["country"]
    assert (country.confirmed, country.contradicted, country.unverifiable) == (0, 0, 1)
    overall = result.outcome_verification["all"]
    assert (overall.confirmed, overall.contradicted, overall.unverifiable) == (1, 2, 3)
    assert list(result.outcome_verification) == ["all", "country", "employees"]


def test_item_level_checks(result: CrossCheck) -> None:
    assert (
        result.violating_items.confirmed,
        result.violating_items.contradicted,
        result.violating_items.mixed,
        result.violating_items.unverifiable,
    ) == (1, 1, 1, 2)
    assert (
        result.satisfying_items.undisputed,
        result.satisfying_items.disputed,
        result.satisfying_items.unverifiable,
    ) == (1, 1, 0)
    assert (
        result.unevaluable_items.resolved_satisfies,
        result.unevaluable_items.resolved_violates,
        result.unevaluable_items.still_unknown,
    ) == (0, 1, 0)


def test_grader_stats(result: CrossCheck) -> None:
    g = result.grader
    # Compared: items 1-5 and 8 (item 6 is unevaluable, item 7 has no grader verdict).
    assert g.compared == 6
    assert (g.ours_satisfies_grader_matches, g.ours_satisfies_grader_rejects) == (1, 1)
    assert (g.ours_violates_grader_matches, g.ours_violates_grader_rejects) == (2, 2)
    assert g.agreement.value == pytest.approx(3 / 6)
    assert g.cannot_verify == 1
    assert g.agreement_excluding_cannot_verify.value == pytest.approx(2 / 5)
    assert (g.unevaluable_compared, g.unevaluable_grader_matches) == (1, 0)


def test_cohen_kappa_known_answers() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert cohen_kappa([True, True], [True, True]) is None  # no variation: undefined
    assert cohen_kappa([], []) is None
    assert cohen_kappa([True], [True, False]) is None


@pytest.mark.parametrize(
    "explanation",
    [
        "The page does not mention headcount.",
        "Unable to verify the funding stage.",
        "The provided content does not verify that it is based in France.",
        "The founded year cannot be confirmed from the available content.",
        "The content is too minimal to verify that the company was founded in 2016.",
        "There is no information about employee count on the page.",
        "Funding details are not stated anywhere in the text.",
        "The page lacks any evidence of a Series A round.",
    ],
)
def test_cannot_verify_wordings_are_recognised(explanation: str) -> None:
    assert reads_as_cannot_verify(explanation)


@pytest.mark.parametrize(
    "explanation",
    [
        "The company is based in France, not Germany.",
        "The page states the company has 184 employees, more than the 100 allowed.",
        "The latest round described is a Series C, so it is past Series B.",
        "Matches: the site says it was founded in 2019 and is headquartered in Berlin.",
    ],
)
def test_decided_explanations_are_not_flagged(explanation: str) -> None:
    assert not reads_as_cannot_verify(explanation)


def test_empty_input() -> None:
    result = cross_check([], seed=1, resamples=RESAMPLES)

    assert result.items == 0
    assert result.fields["employees"].accuracy.value is None
    assert result.outcome_verification == {}
    assert result.grader.kappa is None
