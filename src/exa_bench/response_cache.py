"""Disk cache for raw Exa /search calls, so the same request is never paid for twice."""

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from exa_bench.exa_api import SearchCall, search


@dataclass(frozen=True)
class CachedSearch:
    call: SearchCall
    from_cache: bool  # True means no request was sent (and nothing was spent) this time


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
    sleep: Callable[[float], None] = time.sleep,
) -> CachedSearch:
    """Return the cached call for these exact parameters, or search and cache the result."""
    params: dict[str, object] = {
        "query": query,
        "category": category,
        "num_results": num_results,
        "search_type": search_type,
        "contents": dict(contents) if contents is not None else None,
    }
    path = cache_dir / f"{_params_key(params)}.json"
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
        sleep=sleep,
    )
    _write(path, params, call)
    return CachedSearch(call=call, from_cache=False)


def _params_key(params: Mapping[str, object]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write(path: Path, params: Mapping[str, object], call: SearchCall) -> None:
    record = {
        "params": params,
        "fetched_at": datetime.now(UTC).isoformat(),
        "call": asdict(call),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.replace(path)


def _read(path: Path, params: Mapping[str, object]) -> SearchCall:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("params") != params:
        raise ValueError(f"cache file {path.name} does not match the requested parameters")
    return SearchCall(**record["call"])
