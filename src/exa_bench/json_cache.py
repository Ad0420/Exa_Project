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


def cached_record(
    cache_dir: Path, key: str, compute: Callable[[], dict[str, object]]
) -> tuple[dict[str, object], bool]:
    """Return (record, from_cache): the stored record for `key`, else compute and store it."""
    path = cache_dir / f"{key}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("key") != key:
            raise ValueError(f"cache file {path.name} does not match its key")
        return dict(stored["record"]), True
    record = compute()
    envelope = {"key": key, "stored_at": datetime.now(UTC).isoformat(), "record": record}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(envelope), encoding="utf-8")
    partial.replace(path)
    return record, False
