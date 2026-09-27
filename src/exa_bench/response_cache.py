"""Disk cache for raw Exa /search calls, so the same request is never paid for twice."""

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from exa_bench.exa_api import ApiCall, search


@dataclass(frozen=True)
class CachedSearch:
    call: ApiCall
    from_cache: bool  # True means no request was sent (and nothing was spent) this time


def is_cached(
    cache_dir: Path,
    query: str,
    *,
    category: str | None = "company",
    num_results: int = 10,
    search_type: str = "auto",
    contents: Mapping[str, object] | None = None,
    cache_tag: str | None = None,
    include_domains: Sequence[str] | None = None,
) -> bool:
    """True if a call with exactly these parameters is already on disk."""
    params = _params(
        query, category, num_results, search_type, contents, cache_tag, include_domains
    )
    return _cache_path(cache_dir, params).exists()


def read_cached(
    cache_dir: Path,
    query: str,
    *,
    category: str | None = "company",
    num_results: int = 10,
    search_type: str = "auto",
    contents: Mapping[str, object] | None = None,
    cache_tag: str | None = None,
    include_domains: Sequence[str] | None = None,
) -> ApiCall | None:
    """The cached call for these parameters, without ever making a request; None if absent."""
    params = _params(
        query, category, num_results, search_type, contents, cache_tag, include_domains
    )
    path = _cache_path(cache_dir, params)
    return _read(path, params) if path.exists() else None


def cached_search(
    cache_dir: Path,
    client: httpx.Client,
    api_key: str,
    query: str,
    *,
    category: str | None = "company",
    num_results: int = 10,
    search_type: str = "auto",
    contents: Mapping[str, object] | None = None,
    cache_tag: str | None = None,
    include_domains: Sequence[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> CachedSearch:
    """Return the cached call for these exact parameters, or search and cache the result.

    `cache_tag` is part of the cache key but is never sent to Exa, so an otherwise identical
    request can be repeated (for example to measure run-to-run variation) and kept apart.
    """
    params = _params(
        query, category, num_results, search_type, contents, cache_tag, include_domains
    )
    path = _cache_path(cache_dir, params)
    if path.exists():
        return CachedSearch(call=_read(path, params), from_cache=True)

    call = search(
        client,
        api_key,
        query,
        category=category,
        num_results=num_results,
        search_type=search_type,
        contents=contents,
        include_domains=include_domains,
        sleep=sleep,
    )
    _write(path, params, call)
    return CachedSearch(call=call, from_cache=False)


def _params(
    query: str,
    category: str | None,
    num_results: int,
    search_type: str,
    contents: Mapping[str, object] | None,
    cache_tag: str | None = None,
    include_domains: Sequence[str] | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "query": query,
        "category": category,
        "num_results": num_results,
        "search_type": search_type,
        "contents": dict(contents) if contents is not None else None,
    }
    if cache_tag is not None:  # absent, not null, so untagged keys stay unchanged
        params["cache_tag"] = cache_tag
    if include_domains is not None:  # likewise: keys without a domain filter stay unchanged
        params["include_domains"] = list(include_domains)
    return params


def _cache_path(cache_dir: Path, params: Mapping[str, object]) -> Path:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return cache_dir / f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}.json"


def _write(path: Path, params: Mapping[str, object], call: ApiCall) -> None:
    record = {
        "params": params,
        "fetched_at": datetime.now(UTC).isoformat(),
        "call": asdict(call),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.replace(path)


def _read(path: Path, params: Mapping[str, object]) -> ApiCall:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("params") != params:
        raise ValueError(f"cache file {path.name} does not match the requested parameters")
    return ApiCall(**record["call"])
