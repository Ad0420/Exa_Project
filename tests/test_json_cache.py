"""Tests for the shared JSON record cache."""

import json
from pathlib import Path

import pytest

from exa_bench.json_cache import cache_key, cached_record


def test_key_is_stable_and_order_independent() -> None:
    assert cache_key({"a": 1, "b": "x"}) == cache_key({"b": "x", "a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})


def test_computes_once_then_reads_back(tmp_path: Path) -> None:
    calls = 0

    def compute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"answer": 42}

    first, from_cache_first = cached_record(tmp_path, "k", compute)
    second, from_cache_second = cached_record(tmp_path, "k", compute)

    assert (first, from_cache_first) == ({"answer": 42}, False)
    assert (second, from_cache_second) == ({"answer": 42}, True)
    assert calls == 1


def test_failed_compute_stores_nothing(tmp_path: Path) -> None:
    def compute() -> dict[str, object]:
        raise RuntimeError("model down")

    with pytest.raises(RuntimeError, match="model down"):
        cached_record(tmp_path, "k", compute)

    assert list(tmp_path.iterdir()) == []


def test_mismatched_file_is_rejected(tmp_path: Path) -> None:
    cached_record(tmp_path, "k", lambda: {"answer": 42})
    (cached,) = tmp_path.glob("*.json")
    envelope = json.loads(cached.read_text())
    envelope["key"] = "other"
    cached.write_text(json.dumps(envelope))

    with pytest.raises(ValueError, match="does not match"):
        cached_record(tmp_path, "k", lambda: {"answer": 42})
