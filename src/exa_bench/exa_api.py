"""Compatibility shim: the Exa HTTP client now lives in exa_filters.api."""

from exa_filters.api import (
    CONTENTS_URL,
    MAX_RETRY_DELAY_S,
    RETRYABLE_STATUS,
    SEARCH_URL,
    ApiCall,
    ExaAPIError,
    contents,
    search,
)

__all__ = [
    "CONTENTS_URL",
    "MAX_RETRY_DELAY_S",
    "RETRYABLE_STATUS",
    "SEARCH_URL",
    "ApiCall",
    "ExaAPIError",
    "contents",
    "search",
]
