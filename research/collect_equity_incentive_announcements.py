"""Collect CNInfo A-share equity-incentive draft announcement metadata."""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import math
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import polars as pl

API_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CATEGORY = "category_gqjl_szsh"
SEARCH_KEY = "激励计划 草案"
PAGE_SIZE = 30
MAX_PAGES = 500
MIN_INTERVAL_SECONDS = 0.15

FetchJSON = Callable[[dict[str, str]], dict[str, Any]]
_HTML_TAG = re.compile(r"<[^>]+>")


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


class CNInfoClient:
    def __init__(
        self, *, timeout: float = 30.0, min_interval: float = MIN_INTERVAL_SECONDS
    ) -> None:
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request = 0.0

    def fetch(self, payload: dict[str, str]) -> dict[str, Any]:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(
            API_URL,
            data=urllib.parse.urlencode(payload).encode("utf-8"),
            headers={
                "User-Agent": "Mozilla/5.0 research-metadata-client",
                "Referer": "https://www.cninfo.com.cn/",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        self._last_request = time.monotonic()
        if not isinstance(result, dict):
            raise ValueError("CNInfo announcement response is not an object")
        return result


def _payload(year: int, page: int) -> dict[str, str]:
    return {
        "pageNum": str(page),
        "pageSize": str(PAGE_SIZE),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": SEARCH_KEY,
        "secid": "",
        "category": CATEGORY,
        "trade": "",
        "seDate": f"{year}-01-01~{year}-12-31",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "false",
    }


def _page_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    rows = payload.get("announcements") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("CNInfo announcement list is malformed")
    if "totalAnnouncement" not in payload:
        raise ValueError("CNInfo response has no announcement total")
    total = int(payload.get("totalAnnouncement") or 0)
    pages = int(payload.get("totalpages") or math.ceil(total / PAGE_SIZE))
    return [dict(row) for row in rows], total, pages


def fetch_year(fetch_json: FetchJSON, year: int) -> list[dict[str, Any]]:
    first, total, reported_pages = _page_rows(fetch_json(_payload(year, 1)))
    expected_pages = math.ceil(total / PAGE_SIZE) if total else 0
    accepted_reported_pages = {expected_pages, max(expected_pages - 1, 0)}
    if reported_pages not in accepted_reported_pages or expected_pages > MAX_PAGES:
        raise ValueError("CNInfo equity-incentive page count is invalid")
    pages = expected_pages
    rows = list(first)
    for page in range(2, pages + 1):
        current, current_total, current_reported_pages = _page_rows(
            fetch_json(_payload(year, page))
        )
        if (
            current_total != total
            or current_reported_pages not in accepted_reported_pages
        ):
            raise ValueError("CNInfo pagination changed during collection")
        rows.extend(current)
    if len(rows) != total:
        raise ValueError(
            f"CNInfo pagination incomplete: expected {total}, got {len(rows)}"
        )
    return rows


def _symbol(value: Any) -> str | None:
    code = str(value or "").strip()
    if len(code) != 6 or not code.isdigit():
        return None
    if code.startswith(("4", "8", "92")):
        return f"{code}.BJ"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return None


def _title(value: Any) -> str:
    return html.unescape(_HTML_TAG.sub("", str(value or ""))).strip()


def _announcement_date(value: Any) -> str | None:
    try:
        timestamp_ms = int(value)
    except (TypeError, ValueError):
        return None
    return (
        datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .date()
        .isoformat()
    )


def normalize(rows: list[dict[str, Any]], year: int) -> pl.DataFrame:
    normalized = []
    for row in rows:
        symbol = _symbol(row.get("secCode"))
        ann_date = _announcement_date(row.get("announcementTime"))
        if symbol is None or ann_date is None:
            continue
        normalized.append(
            {
                "announcement_id": str(row.get("announcementId") or "").strip(),
                "ann_date": ann_date,
                "symbol": symbol,
                "company_name": _title(row.get("secName")),
                "title": _title(row.get("announcementTitle")),
                "org_id": str(row.get("orgId") or "").strip(),
                "adjunct_url": str(row.get("adjunctUrl") or "").strip(),
                "column_id": str(row.get("columnId") or "").strip(),
                "announcement_type": str(row.get("announcementType") or "").strip(),
            }
        )
    if not normalized:
        return pl.DataFrame()
    return (
        pl.DataFrame(normalized, infer_schema_length=None)
        .with_columns(pl.col("ann_date").str.to_date("%Y-%m-%d", strict=False))
        .filter(pl.col("ann_date").dt.year() == year)
        .drop_nulls(["announcement_id", "ann_date", "symbol", "title"])
        .filter((pl.col("announcement_id") != "") & (pl.col("title") != ""))
        .unique(subset=["announcement_id", "symbol"], keep="last")
        .sort(["ann_date", "symbol", "announcement_id"])
    )


def collect_year(fetch_json: FetchJSON, root: Path, year: int) -> dict[str, Any]:
    rows = fetch_year(fetch_json, year)
    frame = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(
            f"CNInfo equity-incentive search returned no A-share rows for {year}"
        )
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "path": str(path),
        "raw_rows": len(rows),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "first_ann_date": frame.get_column("ann_date").min(),
        "last_ann_date": frame.get_column("ann_date").max(),
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    client = CNInfoClient()
    root = data_dir / "event_data" / "equity_incentive"
    results = [
        collect_year(client.fetch, root, year)
        for year in range(start_year, end_year + 1)
    ]
    payload = {
        "dataset": "cninfo_equity_incentive_announcements",
        "source": API_URL,
        "category": CATEGORY,
        "search_key": SEARCH_KEY,
        "outcome_fields_persisted": False,
        "start_year": start_year,
        "end_year": end_year,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
