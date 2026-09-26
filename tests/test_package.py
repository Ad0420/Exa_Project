"""Tests for the package's public API."""

import exa_filters
from exa_filters.client import FilteredExa


def test_package_exports_the_public_api() -> None:
    for name in exa_filters.__all__:
        assert getattr(exa_filters, name) is not None
    assert exa_filters.FilteredExa is FilteredExa
