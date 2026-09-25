"""Tests for funding-stage name matching."""

import pytest

from exa_filters.funding_stage import normalize_stage, stage_matches


def test_normalize_lowercases_and_splits_on_hyphens() -> None:
    assert normalize_stage("Pre-Seed") == ("pre", "seed")
    assert normalize_stage("  Series B ") == ("series", "b")
    assert normalize_stage("Post_IPO-Debt") == ("post", "ipo", "debt")


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("Series b", "Series B", True),
        ("Series b", "series", True),
        ("Series b", "Series A", False),
        ("Pre seed", "Pre-Seed", True),
        ("Pre seed", "Seed", False),  # a pre-seed round is not a seed round
        ("Seed", "Seed", True),
        ("Seed", "Pre-Seed", False),
        ("Venture", "Series B", False),
        ("Post ipo debt", "IPO", True),
        ("Series b", "", False),
    ],
)
def test_stage_matches(actual: str, expected: str, matches: bool) -> None:
    assert stage_matches(actual, expected) is matches
