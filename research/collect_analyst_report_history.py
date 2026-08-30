"""Collect one year of point-in-time analyst-report metadata."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

API_URL = "https://reportapi.eastmoney.com/report/list"
PAGE_SIZE = 100
MAX_PAGES = 1_000
MIN_INTERVAL_SECONDS = 0.25


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


class EastmoneyReportClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        min_interval: float = MIN_INTERVAL_SECONDS,
        max_attempts: int = 6,
    ) -> None:
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self._last_request = 0.0

    def fetch(self, params: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            API_URL + "?" + urllib.parse.urlencode(params),
            headers={
                "User-Agent": "Mozilla/5.0 research-metadata-client",
                "Referer": "https://data.eastmoney.com/report/",
            },
        )
        for attempt in range(self.max_attempts):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                self._last_request = time.monotonic()
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    return payload
                raise ValueError("Eastmoney analyst-report response is malformed")
            except (
                ConnectionError,
                TimeoutError,
                http.client.IncompleteRead,
                json.JSONDecodeError,
                urllib.error.URLError,
            ):
                self._last_request = time.monotonic()
                if attempt + 1 >= self.max_attempts:
                    raise
                time.sleep(min(16.0, 2**attempt))
        raise RuntimeError("unreachable analyst-report retry state")


def _params(year: int, page: int) -> dict[str, str]:
    value = str(page)
    return {
        "industryCode": "*",
        "pageSize": str(PAGE_SIZE),
        "industry": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": f"{year}-01-01",
        "endTime": f"{year}-12-31",
        "pageNo": value,
        "qType": "0",
        "orgCode": "",
        "code": "*",
        "rcode": "",
        "p": value,
        "pageNum": value,
        "pageNumber": value,
    }


def _page_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    rows = payload.get("data") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Eastmoney analyst-report rows are malformed")
    return [dict(row) for row in rows], int(payload.get("hits") or 0), int(
        payload.get("TotalPage") or 0
    )


def fetch_year(fetch_json, year: int) -> list[dict[str, Any]]:
    first, count, pages = _page_rows(fetch_json(_params(year, 1)))
    expected_pages = math.ceil(count / PAGE_SIZE) if count else 0
    if pages != expected_pages or not 0 < pages <= MAX_PAGES:
        raise ValueError("Eastmoney analyst-report page count is inconsistent")
    rows = list(first)
    for page in range(2, pages + 1):
        current, current_count, current_pages = _page_rows(
            fetch_json(_params(year, page))
        )
        if current_count != count or current_pages != pages:
            raise ValueError("analyst-report pagination changed during collection")
        rows.extend(current)
    if len(rows) != count:
        raise ValueError(
            f"analyst-report pagination incomplete: expected {count}, got {len(rows)}"
        )
    return rows


def _symbol(code_value: Any, market_value: Any) -> str | None:
    code = str(code_value or "").strip()
    market = str(market_value or "").strip().upper()
    if len(code) != 6 or not code.isdigit():
        return None
    if "SHANGHAI" in market or code.startswith(("6", "9")):
        return f"{code}.SH"
    if "SHENZHEN" in market or code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return None


def normalize(rows: list[dict[str, Any]], year: int) -> pl.DataFrame:
    normalized = []
    for row in rows:
        report_id = str(row.get("infoCode") or "").strip()
        symbol = _symbol(row.get("stockCode"), row.get("market"))
        if not report_id or symbol is None:
            continue
        authors = row.get("author")
        normalized.append(
            {
                "report_id": report_id,
                "publish_date": str(row.get("publishDate") or "")[:10],
                "symbol": symbol,
                "stock_name": str(row.get("stockName") or "").strip(),
                "org_code": str(row.get("orgCode") or "").strip(),
                "org_name": str(row.get("orgSName") or row.get("orgName") or "").strip(),
                "researcher": str(row.get("researcher") or "").strip(),
                "authors": json.dumps(authors, ensure_ascii=False)
                if isinstance(authors, list)
                else str(authors or ""),
                "current_rating": str(row.get("emRatingName") or "").strip(),
                "last_rating": str(row.get("lastEmRatingName") or "").strip(),
                "target_price_high": row.get("indvAimPriceT"),
                "target_price_low": row.get("indvAimPriceL"),
                "report_type": row.get("reportType"),
                "is_new_coverage": str(row.get("indvIsNew") or "").strip(),
                "title": str(row.get("title") or "").strip(),
            }
        )
    if not normalized:
        return pl.DataFrame()
    return (
        pl.DataFrame(normalized, infer_schema_length=None)
        .with_columns(
            pl.col("publish_date").str.to_date("%Y-%m-%d", strict=False),
            pl.col("target_price_high").cast(pl.Float64, strict=False),
            pl.col("target_price_low").cast(pl.Float64, strict=False),
            pl.col("report_type").cast(pl.Int64, strict=False),
        )
        .filter(pl.col("publish_date").dt.year() == year)
        .sort(["publish_date", "symbol", "org_code", "report_id"])
        .unique(subset=["report_id"], keep="first", maintain_order=True)
    )


def collect_year(fetch_json, root: Path, year: int) -> dict[str, Any]:
    rows = fetch_year(fetch_json, year)
    frame = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(f"analyst-report source returned no usable rows for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    target_covered = frame.filter(
        (pl.col("target_price_high").fill_null(0) > 0)
        | (pl.col("target_price_low").fill_null(0) > 0)
    )
    return {
        "year": year,
        "path": str(path),
        "raw_rows": len(rows),
        "reports": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "brokers": frame.filter(pl.col("org_code") != "")
        .get_column("org_code")
        .n_unique(),
        "target_price_reports": target_covered.height,
        "first_publish_date": frame.get_column("publish_date").min(),
        "last_publish_date": frame.get_column("publish_date").max(),
    }


def run(data_dir: Path, year: int) -> dict[str, Any]:
    if year < 2017 or year > 2026:
        raise ValueError("analyst-report collection supports 2017-2026")
    client = EastmoneyReportClient()
    result = collect_year(
        client.fetch,
        data_dir / "event_data" / "analyst_reports",
        year,
    )
    payload = {
        "dataset": "eastmoney_analyst_report_history",
        "outcome_fields_persisted": False,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.year)


if __name__ == "__main__":
    main()
