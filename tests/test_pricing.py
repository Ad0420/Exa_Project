"""Tests for Exa's list-price table."""

import pytest

from exa_filters.pricing import search_price


def test_search_price_matches_exas_published_rates() -> None:
    assert search_price(10) == pytest.approx(0.007)
    assert search_price(1) == pytest.approx(0.007)
    assert search_price(25) == pytest.approx(0.022)
    assert search_price(100) == pytest.approx(0.097)
    assert search_price(10, "deep") == pytest.approx(0.012)
    assert search_price(25, "deep") == pytest.approx(0.027)
    assert search_price(10, "deep-reasoning") == pytest.approx(0.015)
    assert search_price(10, "instant") == search_price(10, "fast") == search_price(10)


def test_search_price_rejects_unknown_types() -> None:
    with pytest.raises(ValueError, match="no list price"):
        search_price(10, "turbo")
