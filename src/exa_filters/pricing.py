"""Exa's list prices, for dry-run estimates and budgets. Actual charges come from each response's
`costDollars`. Source: https://exa.ai/docs/admin/pricing, checked 2026-09-27.
"""

# /search base price per request, covering up to 10 results, by search type.
SEARCH_BASE_USD = {
    "instant": 0.007,
    "fast": 0.007,
    "auto": 0.007,
    "deep-lite": 0.012,
    "deep": 0.012,
    "deep-reasoning": 0.015,
}
EXTRA_RESULT_USD = 0.001  # each /search result beyond 10, for every search type
CONTENTS_PER_PAGE_USD = 0.001  # /contents, per page and content type
AGENT_EFFORT_USD = {"minimal": 0.012, "low": 0.025, "medium": 0.10, "high": 0.50, "xhigh": 1.00}
AGENT_SEARCH_USD = 0.005  # each search an Agent run makes, billed on metered efforts


def search_price(num_results: int, search_type: str = "auto") -> float:
    """List price of one /search request returning `num_results` results."""
    if search_type not in SEARCH_BASE_USD:
        raise ValueError(f"no list price for search type {search_type!r}")
    return SEARCH_BASE_USD[search_type] + max(0, num_results - 10) * EXTRA_RESULT_USD
