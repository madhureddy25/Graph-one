"""Shared plumbing for source collectors."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable, TypeVar

from ..config import Settings
from ..http.fetcher import Fetcher
from ..llm.orchestrator import LLMOrchestrator
from ..resolver.resolver import EntityResolver
from ..state.store import SqliteStateStore

T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger("graphone.sources")


@dataclass
class Shard:
    """Horizontal partitioning: node k of N only processes work units whose
    stable hash % N == k. Run N processes/containers with --shard k/N."""
    index: int = 0
    total: int = 1

    @classmethod
    def parse(cls, spec: str | None) -> "Shard":
        if not spec:
            return cls()
        i, n = spec.split("/")
        return cls(int(i), int(n))

    def owns(self, key: str) -> bool:
        if self.total <= 1:
            return True
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.total == self.index


@dataclass
class Context:
    settings: Settings
    fetcher: Fetcher
    store: SqliteStateStore
    llm: LLMOrchestrator
    resolver: EntityResolver
    shard: Shard = field(default_factory=Shard)
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


async def bounded_map(items: Iterable[T], fn: Callable[[T], Awaitable[R]], concurrency: int,
                      stop_when: Callable[[], bool] | None = None) -> list[R]:
    """Worker-pool map over an iterable with a bounded queue (constant memory
    regardless of input size). Exceptions are logged and yield None."""
    q: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    results: list[R] = []

    async def worker():
        while True:
            item = await q.get()
            try:
                if item is _STOP:
                    return
                if stop_when and stop_when():
                    continue
                try:
                    r = await fn(item)
                    if r is not None:
                        results.append(r)
                except Exception as exc:  # one bad item never kills the run
                    logger.debug("worker item failed: %s", exc, exc_info=False)
            finally:
                q.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    for it in items:
        if stop_when and stop_when():
            break
        await q.put(it)
    for _ in workers:
        await q.put(_STOP)
    await asyncio.gather(*workers)
    return results


_STOP = object()
