"""Tests for country normalization."""

import pytest

from exa_filters.country import country_key, same_country


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Germany", "DE"),
        ("DE", "DE"),
        ("de", "DE"),
        ("  germany  ", "DE"),
        ("United States", "US"),
        ("USA", "US"),
        ("United Kingdom", "GB"),
        ("UK", "GB"),
        ("South Korea", "KR"),
        ("KR", "KR"),
        ("Turkey", "TR"),
        ("Netherlands", "NL"),
        ("The Netherlands", "NL"),
        ("Taiwan", "TW"),
        ("India", "IN"),
        ("IN", "IN"),
    ],
)
def test_names_codes_and_aliases_resolve_to_alpha_2(text: str, key: str) -> None:
    assert country_key(text) == key


@pytest.mark.parametrize("text", ["Scandinavia", "Europe", "Bay Area", ""])
def test_non_countries_keep_their_cleaned_text(text: str) -> None:
    assert country_key(text) == text.casefold()


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ("Germany", "DE", True),
        ("UK", "United Kingdom", True),
        ("South Korea", "KR", True),
        ("Germany", "Austria", False),
        ("Scandinavia", "Sweden", False),
        ("Europe", "europe", True),
    ],
)
def test_same_country(first: str, second: str, same: bool) -> None:
    assert same_country(first, second) is same
