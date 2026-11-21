"""Crawl state: dedup claims, checkpoints, persisted records, LLM cache.

`claim(ns, key)` is the core primitive for "never process the same article/job
twice across distributed nodes": it is an atomic insert-if-absent.
  * SqliteStateStore  -> INSERT OR IGNORE (single node, zero infra)
  * RedisStateStore   -> SET key 1 NX EX ttl (N crawler nodes share one Redis)
Swapping backends is the env var STATE_BACKEND; no code changes.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..config import Settings

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS claims (ns TEXT, key TEXT, ts REAL, meta TEXT, PRIMARY KEY (ns, key));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS records (ns TEXT, key TEXT, ts TEXT, data TEXT, PRIMARY KEY (ns, key));
CREATE INDEX IF NOT EXISTS records_ts ON records (ns, ts);
CREATE TABLE IF NOT EXISTS mappings (raw TEXT, canonical TEXT, method TEXT, score REAL,
    record_type TEXT, source_url TEXT, PRIMARY KEY (raw, record_type));
"""


class SqliteStateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def _x(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    async def _run(self, sql: str, args: tuple = ()) -> list[tuple]:
        return await asyncio.to_thread(self._x, sql, args)

    # ---- dedup
    async def claim(self, ns: str, key: str, meta: dict | None = None, ttl: float | None = None) -> bool:
        def op() -> bool:
            with self._lock:
                if ttl is not None:
                    self._db.execute("DELETE FROM claims WHERE ns=? AND key=? AND ts<?", (ns, key, time.time() - ttl))
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO claims VALUES (?,?,?,?)", (ns, key, time.time(), json.dumps(meta or {}))
                )
                return cur.rowcount == 1
        return await asyncio.to_thread(op)

    async def is_claimed(self, ns: str, key: str) -> bool:
        return bool(await self._run("SELECT 1 FROM claims WHERE ns=? AND key=?", (ns, key)))

    async def release(self, ns: str, key: str) -> None:
        await self._run("DELETE FROM claims WHERE ns=? AND key=?", (ns, key))

    # ---- kv (checkpoints, last-run times, LLM cache)
    async def kv_get(self, key: str) -> Any:
        rows = await self._run("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0][0]) if rows else None

    async def kv_set(self, key: str, value: Any) -> None:
        await self._run("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (key, json.dumps(value, default=str), time.time()))

    # ---- records
    async def put_record(self, ns: str, key: str, ts: str | None, data: dict) -> None:
        await self._run("INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                        (ns, key, ts or "", json.dumps(data, ensure_ascii=False, default=str)))

    async def get_records(self, ns: str, since_iso: str | None = None) -> list[dict]:
        if since_iso:
            rows = await self._run("SELECT data FROM records WHERE ns=? AND ts>=? ORDER BY ts DESC", (ns, since_iso))
        else:
            rows = await self._run("SELECT data FROM records WHERE ns=? ORDER BY rowid", (ns,))
        return [json.loads(r[0]) for r in rows]

    async def get_keyed_records(self, ns: str) -> list[tuple[str, str, dict]]:
        rows = await self._run("SELECT key, ts, data FROM records WHERE ns=? ORDER BY rowid", (ns,))
        return [(k, ts, json.loads(d)) for k, ts, d in rows]

    async def has_record(self, ns: str, key: str) -> bool:
        return bool(await self._run("SELECT 1 FROM records WHERE ns=? AND key=?", (ns, key)))

    # ---- entity mapping log
    async def add_mappings(self, rows: list[dict]) -> None:
        def op():
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO mappings VALUES (:raw,:canonical,:method,:score,:recordType,:sourceUrl)", rows
                )
        if rows:
            await asyncio.to_thread(op)

    async def get_mappings(self) -> list[dict]:
        rows = await self._run("SELECT raw, canonical, method, score, record_type, source_url FROM mappings "
                               "ORDER BY canonical, raw")
        return [dict(zip(("raw", "canonical", "method", "score", "recordType", "sourceUrl"), r)) for r in rows]

    async def close(self) -> None:
        self._db.close()


class RedisStateStore(SqliteStateStore):
    """Claims + kv live in Redis (shared by all crawler nodes); bulk records stay
    in the node-local sink (in production: Postgres, see architecture doc)."""

    def __init__(self, path: Path, redis_url: str) -> None:
        super().__init__(path)
        import redis.asyncio as redis  # optional dependency

        self._r = redis.from_url(redis_url, decode_responses=True)

    async def claim(self, ns: str, key: str, meta: dict | None = None, ttl: float | None = None) -> bool:
        ok = await self._r.set(f"claim:{ns}:{key}", json.dumps(meta or {}), nx=True,
                               ex=int(ttl) if ttl else None)
        return bool(ok)

    async def is_claimed(self, ns: str, key: str) -> bool:
        return bool(await self._r.exists(f"claim:{ns}:{key}"))

    async def release(self, ns: str, key: str) -> None:
        await self._r.delete(f"claim:{ns}:{key}")

    async def kv_get(self, key: str) -> Any:
        v = await self._r.get(f"kv:{key}")
        return json.loads(v) if v else None

    async def kv_set(self, key: str, value: Any) -> None:
        await self._r.set(f"kv:{key}", json.dumps(value, default=str))

    async def close(self) -> None:
        await self._r.aclose()
        await super().close()


def make_store(settings: Settings) -> SqliteStateStore:
    if settings.state_backend == "redis":
        return RedisStateStore(settings.state_path, settings.redis_url or "redis://localhost:6379/0")
    return SqliteStateStore(settings.state_path)
