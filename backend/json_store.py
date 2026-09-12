"""
json_store.py
Thread-safe, crash-safe helpers for the local JSON files used as lightweight storage.

Every append is a read-modify-write, and FastAPI serves the sync routes from a
thread pool, so concurrent submissions would otherwise drop entries. Writes also
go through a temp file plus os.replace so an interrupted write cannot truncate
an existing data file.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Dict, List


_LOCKS_GUARD = threading.Lock()
_LOCKS: Dict[str, threading.Lock] = {}


def _lock_for(path: str) -> threading.Lock:
    """Return the process-wide lock guarding one file path."""
    key = os.path.abspath(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def read_json_list(path: str) -> List[Any]:
    """Load a JSON list from disk, returning an empty list on any failure."""
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def write_json_list(path: str, entries: List[Any], *, indent: int = 2) -> None:
    """Replace the file contents atomically."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    handle, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=indent, ensure_ascii=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def append_json_list(
    path: str,
    entry: Any,
    *,
    indent: int = 2,
    max_entries: int | None = None,
) -> Any:
    """
    Append one entry to a JSON list file under a per-file lock.
    When max_entries is set, only the most recent entries are kept.
    """
    with _lock_for(path):
        entries = read_json_list(path)
        entries.append(entry)
        if max_entries is not None and len(entries) > max_entries:
            entries = entries[-max_entries:]
        write_json_list(path, entries, indent=indent)
    return entry
