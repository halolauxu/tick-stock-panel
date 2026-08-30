"""Collect Eastmoney A-share restructuring announcement metadata by year."""

from __future__ import annotations

import argparse
import calendar
import contextlib
import json
import math
import os
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable

import polars as pl

API_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
PAGE_SIZE = 100
MAX_PAGES = 500
MIN_INTERVAL_SECONDS = 0.08

FetchJSON = Callable[[dict[str, str]], dict[str, Any]]


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


class EastmoneyClient:
    def __init__(
        self, *, timeout: float = 30.0, min_interval: float = MIN_INTERVAL_SECONDS
    ) -> None:
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request = 0.0

    def fetch(self, params: dict[str, str]) -> dict[str, Any]:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(
            API_URL + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": "Mozilla/5.0 research-metadata-client"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        self._last_request = time.monotonic()
        if not isinstance(payload, dict):
            raise ValueError("Eastmoney announcement response is not an object")
        return payload


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _params(year: int, month: int, page: int) -> dict[str, str]:
    start, end = _month_bounds(year, month)
    return {
        "sr": "-1",
        "page_size": str(PAGE_SIZE),
        "page_index": str(page),
        "ann_type": "A",
        "client_source": "web",
        "f_node": "6",
        "s_node": "0",
        "begin_time": start.isoformat(),
        "end_time": end.isoformat(),
    }


def _page_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if int(payload.get("success") or 0) != 1:
        raise ValueError("Eastmoney announcement request was not successful")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Eastmoney announcement response has no data object")
    rows = data.get("list") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Eastmoney announcement list is malformed")
    total_hits = int(data.get("total_hits") or 0)
    return [dict(row) for row in rows], total_hits


def fetch_month(fetch_json: FetchJSON, year: int, month: int) -> list[dict[str, Any]]:
    first, total_hits = _page_rows(fetch_json(_params(year, month, 1)))
    pages = math.ceil(total_hits / PAGE_SIZE)
    if pages > MAX_PAGES:
        raise ValueError(f"Eastmoney announcement month exceeds {MAX_PAGES} pages")
    rows = list(first)
    for page in range(2, pages + 1):
        current, current_total = _page_rows(fetch_json(_params(year, month, page)))
        if current_total != total_hits:
            raise ValueError(
                "Eastmoney announcement pagination total changed during collection"
            )
        rows.extend(current)
    if len(rows) != total_hits:
        raise ValueError(
            f"Eastmoney announcement pagination incomplete: expected {total_hits}, got {len(rows)}"
        )
    return rows


def _symbol(stock_code: str) -> str | None:
    code = stock_code.strip()
    if len(code) != 6 or not code.isdigit():
        return None
    if code.startswith(("4", "8", "92")):
        return f"{code}.BJ"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return None


def normalize(rows: list[dict[str, Any]], year: int) -> pl.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        columns = row.get("columns") or []
        column_names = sorted(
            {
                str(item.get("column_name") or "").strip()
                for item in columns
                if isinstance(item, dict) and item.get("column_name")
            }
        )
        column_codes = sorted(
            {
                str(item.get("column_code") or "").strip()
                for item in columns
                if isinstance(item, dict) and item.get("column_code")
            }
        )
        for code in row.get("codes") or []:
            if not isinstance(code, dict) or not str(
                code.get("ann_type") or ""
            ).startswith("A"):
                continue
            symbol = _symbol(str(code.get("stock_code") or ""))
            if symbol is None:
                continue
            normalized.append(
                {
                    "art_code": str(row.get("art_code") or "").strip(),
                    "ann_date": str(row.get("notice_date") or "")[:10],
                    "symbol": symbol,
                    "company_name": str(code.get("short_name") or "").strip(),
                    "title": str(row.get("title") or "").strip(),
                    "column_name": "|".join(column_names),
                    "column_code": "|".join(column_codes),
                }
            )
    if not normalized:
        return pl.DataFrame()
    return (
        pl.DataFrame(normalized, infer_schema_length=None)
        .with_columns(pl.col("ann_date").str.to_date("%Y-%m-%d", strict=False))
        .filter(pl.col("ann_date").dt.year() == year)
        .drop_nulls(["art_code", "ann_date", "symbol", "title"])
        .filter((pl.col("art_code") != "") & (pl.col("title") != ""))
        .unique(subset=["art_code", "symbol", "column_code"], keep="last")
        .sort(["ann_date", "symbol", "art_code"])
    )


def collect_year(fetch_json: FetchJSON, root: Path, year: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, int]] = []
    for month in range(1, 13):
        current = fetch_month(fetch_json, year, month)
        rows.extend(current)
        monthly_rows.append({"month": month, "raw_rows": len(current)})
    frame = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(
            f"Eastmoney restructuring announcements returned no A-share rows for {year}"
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
        "monthly_rows": monthly_rows,
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    client = EastmoneyClient()
    root = data_dir / "event_data" / "restructuring_announcements"
    results = [
        collect_year(client.fetch, root, year)
        for year in range(start_year, end_year + 1)
    ]
    payload = {
        "dataset": "eastmoney_restructuring_announcements",
        "source": API_URL,
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
