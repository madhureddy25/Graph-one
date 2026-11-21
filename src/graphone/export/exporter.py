"""Exports: JSONL (full nested records), CSV per tab, one XLSX workbook with the
6 required tabs, and optional direct upload to Google Sheets."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..dates import parse_datetime
from ..logging_setup import log
from ..models import flatten

logger = logging.getLogger("graphone.export")

TABS = ["Startups", "Products", "Research Papers", "Jobs", "News", "Entity Mapping Log"]
FILES = {"Startups": "startups", "Products": "products", "Research Papers": "papers", "Jobs": "jobs", "News": "news",
         "Entity Mapping Log": "entity_mapping"}
CELL_LIMIT = 49000  # Google Sheets hard limit is 50,000 chars per cell


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_table(rows: list[dict]) -> tuple[list[str], list[list]]:
    flat = [flatten(r) for r in rows]
    cols: list[str] = []
    for f in flat:
        for k in f:
            if k not in cols:
                cols.append(k)
    data = []
    for f in flat:
        row = []
        for c in cols:
            v = f.get(c)
            if isinstance(v, str) and len(v) > CELL_LIMIT:
                v = v[:CELL_LIMIT] + " …[truncated for Sheets; full text in news.jsonl]"
            row.append("" if v is None else v)
        data.append(row)
    return cols, data


async def gather_tables(store, out_dir: Path, freshness_hours: int) -> dict[str, list[dict]]:
    """Phase I tabs come from the JSONL snapshots, Phase II tabs from the state DB
    filtered to the freshness window *at export time*."""
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=freshness_hours)).isoformat().replace("+00:00", "Z")

    def fresh(rows, field):
        keep = []
        for r in rows:
            # inferred-fresh rows have no publication date: age them by when we first collected them
            dt = parse_datetime(r["content"].get(field)) or parse_datetime(r.get("collectedAt"))
            if dt is not None and now - dt <= timedelta(hours=freshness_hours):
                keep.append(r)
        return keep

    jobs = fresh(await store.get_records("jobs"), "date")
    news = fresh(await store.get_records("news"), "published_date")
    jobs.sort(key=lambda r: r["content"].get("date") or "", reverse=True)
    news.sort(key=lambda r: r["content"].get("published_date") or "", reverse=True)
    log(logger, logging.INFO, "export window", since=since, jobs=len(jobs), news=len(news))
    return {
        "Startups": read_jsonl(out_dir / "startups.jsonl"),
        "Products": read_jsonl(out_dir / "products.jsonl"),
        "Research Papers": read_jsonl(out_dir / "papers.jsonl"),
        "Jobs": jobs,
        "News": news,
        "Entity Mapping Log": await store.get_mappings(),
    }


def write_files(tables: dict[str, list[dict]], out_dir: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for tab in TABS:
        rows = tables.get(tab, [])
        cols, data = to_table(rows)
        name = FILES[tab]
        if tab in ("Jobs", "News", "Entity Mapping Log"):
            write_jsonl(out_dir / f"{name}.jsonl", rows)
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(data)
        ws = wb.create_sheet(tab)
        ws.append(cols)
        for r in data:
            ws.append([v if not isinstance(v, str) else v[:32000] for v in r])  # Excel cell limit
        ws.freeze_panes = "A2"
    path = out_dir / "graphone_output.xlsx"
    wb.save(path)
    log(logger, logging.INFO, "files written", xlsx=str(path), **{FILES[t]: len(tables.get(t, [])) for t in TABS})
    return path


def upload_sheets(tables: dict[str, list[dict]], sa_file: str, sheet_id: str | None) -> str:
    import gspread

    gc = gspread.service_account(filename=sa_file)
    sh = gc.open_by_key(sheet_id) if sheet_id else gc.create("GraphOne Pipeline Output")
    existing = {ws.title: ws for ws in sh.worksheets()}
    for tab in TABS:
        cols, data = to_table(tables.get(tab, []))
        ws = existing.get(tab) or sh.add_worksheet(title=tab, rows=max(2, len(data) + 1), cols=max(1, len(cols)))
        ws.clear()
        ws.resize(rows=max(2, len(data) + 1), cols=max(1, len(cols)))
        values = [cols] + data
        for i in range(0, len(values), 2000):  # chunked to stay under request size limits
            ws.update(values=values[i:i + 2000], range_name=f"A{i + 1}", value_input_option="RAW")
        ws.freeze(rows=1)
    for title, ws in existing.items():
        if title not in TABS and len(sh.worksheets()) > 1:
            sh.del_worksheet(ws)
    sh.share(None, perm_type="anyone", role="reader")
    log(logger, logging.INFO, "google sheet updated", url=sh.url)
    return sh.url
