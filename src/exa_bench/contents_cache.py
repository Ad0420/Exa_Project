"""Per-URL disk cache for page text from Exa /contents, so each page is paid for once."""

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from exa_bench.exa_api import ApiCall, contents

DEFAULT_BATCH_SIZE = 20


@dataclass(frozen=True)
class PageText:
    text: str | None  # None when Exa returned no text for the page
    status: Mapping[str, object] | None  # Exa's per-URL status entry, when present


@dataclass(frozen=True)
class ContentsFetch:
    pages: dict[str, PageText]  # every requested URL
    from_cache: int  # how many were served without a request
    calls: list[ApiCall]  # the batched requests made this time, for cost logging


def uncached_urls(cache_dir: Path, urls: Iterable[str]) -> list[str]:
    """The URLs (deduplicated, in order) that would have to be fetched."""
    seen: dict[str, None] = {}
    for url in urls:
        if url not in seen and not _path(cache_dir, url).exists():
            seen[url] = None
    return list(seen)


def cached_contents(
    cache_dir: Path,
    client: httpx.Client,
    api_key: str,
    urls: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep: Callable[[float], None] = time.sleep,
) -> ContentsFetch:
    """Return page text for every URL, fetching only the ones not already on disk."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    unique = list(dict.fromkeys(urls))
    missing = uncached_urls(cache_dir, unique)
    calls: list[ApiCall] = []
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        call = contents(client, api_key, batch, sleep=sleep)
        calls.append(call)
        for url, page in _pages_from(call.body, batch).items():
            _write(_path(cache_dir, url), url, page)
    pages = {url: _read(_path(cache_dir, url), url) for url in unique}
    return ContentsFetch(pages=pages, from_cache=len(unique) - len(missing), calls=calls)


def _pages_from(body: Mapping[str, object], requested: Sequence[str]) -> dict[str, PageText]:
    """Match Exa's results and statuses back to the requested URLs by id, then by url."""
    texts: dict[str, str] = {}
    for result in _as_list(body.get("results")):
        if not isinstance(result, Mapping):
            continue
        text = result.get("text")
        for key in ("id", "url"):
            identifier = result.get(key)
            if isinstance(identifier, str) and isinstance(text, str) and text:
                texts.setdefault(identifier, text)
    statuses: dict[str, Mapping[str, object]] = {}
    for status in _as_list(body.get("statuses")):
        if isinstance(status, Mapping) and isinstance(status.get("id"), str):
            statuses[str(status["id"])] = status
    return {url: PageText(texts.get(url), statuses.get(url)) for url in requested}


def _path(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"


def _write(path: Path, url: str, page: PageText) -> None:
    record = {
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "text": page.text,
        "status": dict(page.status) if page.status is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.replace(path)


def _read(path: Path, url: str) -> PageText:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("url") != url:
        raise ValueError(f"cache file {path.name} does not match {url!r}")
    return PageText(record.get("text"), record.get("status"))


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []
