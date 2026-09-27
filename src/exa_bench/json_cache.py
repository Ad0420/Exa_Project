"""A tiny on-disk cache of JSON records keyed by a content hash, shared by the LLM judges."""

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path


def cache_key(parts: Mapping[str, object]) -> str:
    """Stable hash of the inputs that determine a cached record."""
    canonical = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_cached(cache_dir: Path, key: str) -> bool:
    """True if a record for `key` is on disk, so a dry run can count what a run would compute."""
    return (cache_dir / f"{key}.json").exists()


def read_record(cache_dir: Path, key: str) -> dict[str, object] | None:
    """The stored record for `key` without computing anything; None if absent."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("key") != key:
        raise ValueError(f"cache file {path.name} does not match its key")
    return dict(stored["record"])


def cached_record(
    cache_dir: Path, key: str, compute: Callable[[], dict[str, object]]
) -> tuple[dict[str, object], bool]:
    """Return (record, from_cache): the stored record for `key`, else compute and store it."""
    stored = read_record(cache_dir, key)
    if stored is not None:
        return stored, True
    record = compute()
    path = cache_dir / f"{key}.json"
    envelope = {"key": key, "stored_at": datetime.now(UTC).isoformat(), "record": record}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(envelope), encoding="utf-8")
    partial.replace(path)
    return record, False
