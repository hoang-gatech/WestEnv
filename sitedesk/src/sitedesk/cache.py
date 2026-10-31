"""Tiny TTL cache: in-memory, optionally mirrored to disk, with in-flight de-duplication.

The upstream services here are slow and rate-limited (measured 2026-10-26: SoilGrids 3-16 s per
call, Overpass ~35 s), so a repeated or concurrent lookup of the same cell must not hit them twice.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger(__name__)

_MAX_MEMORY_ENTRIES = 2000

Entry = tuple[float, Any]  # (fetched_at as epoch seconds, value)


class Cache:
    def __init__(self, directory: Path | None) -> None:
        self._dir = directory
        self._mem: dict[str, Entry] = {}
        self._inflight: dict[str, asyncio.Task[Entry]] = {}

    async def get_or_fetch(
        self,
        namespace: str,
        key: str,
        ttl_s: float,
        fetch: Callable[[], Awaitable[Any]],
        *,
        persist: bool = True,
    ) -> Entry:
        """Return (fetched_at, value), calling `fetch` only when there is no fresh entry.

        Concurrent callers for the same key share one fetch. `value` must be JSON-serialisable
        when `persist` is true.
        """
        full = f"{namespace}:{key}"
        entry = self._read(namespace, full, ttl_s, persist)
        if entry is not None:
            return entry
        task = self._inflight.get(full)
        if task is None:
            task = asyncio.create_task(self._load(namespace, full, fetch, persist))
            self._inflight[full] = task
            task.add_done_callback(lambda t: self._finish(full, t))
        # shield: one caller being cancelled must not cancel the fetch the others are waiting on
        return await asyncio.shield(task)

    async def _load(
        self, namespace: str, full: str, fetch: Callable[[], Awaitable[Any]], persist: bool
    ) -> Entry:
        entry = (time.time(), await fetch())
        if len(self._mem) >= _MAX_MEMORY_ENTRIES:
            for stale in list(self._mem)[: _MAX_MEMORY_ENTRIES // 10]:
                del self._mem[stale]
        self._mem[full] = entry
        if persist:
            self._write(namespace, full, entry)
        return entry

    def _finish(self, full: str, task: asyncio.Task[Entry]) -> None:
        self._inflight.pop(full, None)
        if not task.cancelled():
            task.exception()  # mark as retrieved; callers that were waiting already saw it

    def _read(self, namespace: str, full: str, ttl_s: float, persist: bool) -> Entry | None:
        entry = self._mem.get(full)
        if entry is None and persist and self._dir is not None:
            entry = self._read_disk(namespace, full)
            if entry is not None:
                self._mem[full] = entry
        if entry is not None and time.time() - entry[0] < ttl_s:
            return entry
        return None

    def _path(self, namespace: str, full: str) -> Path:
        assert self._dir is not None
        return self._dir / namespace / f"{hashlib.sha1(full.encode()).hexdigest()[:24]}.json"

    def _read_disk(self, namespace: str, full: str) -> Entry | None:
        try:
            doc = json.loads(self._path(namespace, full).read_text(encoding="utf-8"))
            if doc["key"] == full:
                return float(doc["t"]), doc["v"]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # missing or corrupt: treat as a miss
        return None

    def _write(self, namespace: str, full: str, entry: Entry) -> None:
        if self._dir is None:
            return
        try:
            path = self._path(namespace, full)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps({"key": full, "t": entry[0], "v": entry[1]}), encoding="utf-8"
            )
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("cache write failed for %s: %s", full, exc)  # the cache is best-effort


default = Cache(settings.cache_dir)
