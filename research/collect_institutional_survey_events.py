"""Collect one month of point-in-time institutional-survey attention events."""

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
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPT_ORG_SURVEY"
# This wide report returns Eastmoney code 9701 consistently at 100+ rows/page.
# Fifty rows is the largest page size verified from both the workstation and the
# production host; keep it fixed so a collection cannot silently degrade into
# repeated "server busy" responses.
PAGE_SIZE = 50
MAX_PAGES = 2_000
MIN_INTERVAL_SECONDS = 1.0


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


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


class EastmoneySurveyClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        min_interval: float = MIN_INTERVAL_SECONDS,
        max_attempts: int = 8,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self._last_request = 0.0

    def fetch(self, params: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            API_URL + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": "Mozilla/5.0 research-metadata-client"},
        )
        for attempt in range(self.max_attempts):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                self._last_request = time.monotonic()
                if isinstance(payload, dict) and payload.get("success") is True:
                    return payload
                message = str(payload.get("message") if isinstance(payload, dict) else "")
                if "繁忙" not in message:
                    raise ValueError("Eastmoney survey request was not successful")
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
            if attempt + 1 >= self.max_attempts:
                page = params.get("pageNumber", "?")
                raise ValueError(f"Eastmoney survey page {page} remained busy")
            time.sleep(min(30.0, self.retry_base_seconds * (2**attempt)))
        raise RuntimeError("unreachable survey retry state")


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _params(year: int, month: int, page: int) -> dict[str, str]:
    start, end = _month_bounds(year, month)
    return {
        "reportName": REPORT_NAME,
        "columns": (
            "SECUCODE,NOTICE_DATE,RECEIVE_START_DATE,RECEIVE_END_DATE,"
            "RECEIVE_OBJECT_TYPE,RECEIVE_OBJECT,OBJECT_CODE,SUM,ORG_TYPE,"
            "URL"
        ),
        "filter": (
            f"(NOTICE_DATE>='{start.isoformat()}')"
            f"(NOTICE_DATE<='{end.isoformat()}')"
        ),
        "pageNumber": str(page),
        "pageSize": str(PAGE_SIZE),
        "sortColumns": "NOTICE_DATE",
        "sortTypes": "1",
        "source": "WEB",
        "client": "WEB",
    }


def _page_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Eastmoney survey response has no result")
    rows = result.get("data") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Eastmoney survey rows are malformed")
    return [dict(row) for row in rows], int(result.get("count") or 0), int(
        result.get("pages") or 0
    )


def _page_cache_path(cache_dir: Path, page: int) -> Path:
    return cache_dir / f"page={page:04d}.json"


def _write_page_cache(
    cache_dir: Path,
    *,
    year: int,
    month: int,
    page: int,
    count: int,
    pages: int,
    rows: list[dict[str, Any]],
) -> None:
    _atomic_write_json(
        {
            "year": year,
            "month": month,
            "page": page,
            "count": count,
            "pages": pages,
            "rows": rows,
        },
        _page_cache_path(cache_dir, page),
    )


def _read_page_cache(
    cache_dir: Path,
    *,
    year: int,
    month: int,
    page: int,
    count: int,
    pages: int,
) -> list[dict[str, Any]] | None:
    path = _page_cache_path(cache_dir, page)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or any(
        payload.get(key) != value
        for key, value in {
            "year": year,
            "month": month,
            "page": page,
            "count": count,
            "pages": pages,
        }.items()
    ):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None
    expected_rows = PAGE_SIZE if page < pages else count - PAGE_SIZE * (pages - 1)
    return [dict(row) for row in rows] if len(rows) == expected_rows else None


def fetch_month(
    fetch_json,
    year: int,
    month: int,
    *,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    first, count, pages = _page_rows(fetch_json(_params(year, month, 1)))
    expected_pages = math.ceil(count / PAGE_SIZE) if count else 0
    if pages != expected_pages or pages > MAX_PAGES:
        raise ValueError("Eastmoney survey page count is inconsistent")
    if cache_dir is not None:
        _write_page_cache(
            cache_dir,
            year=year,
            month=month,
            page=1,
            count=count,
            pages=pages,
            rows=first,
        )
    rows = list(first)
    for page in range(2, pages + 1):
        current = (
            _read_page_cache(
                cache_dir,
                year=year,
                month=month,
                page=page,
                count=count,
                pages=pages,
            )
            if cache_dir is not None
            else None
        )
        if current is None:
            current, current_count, current_pages = _page_rows(
                fetch_json(_params(year, month, page))
            )
            if current_count != count or current_pages != pages:
                raise ValueError("Eastmoney survey pagination changed during collection")
            if cache_dir is not None:
                _write_page_cache(
                    cache_dir,
                    year=year,
                    month=month,
                    page=page,
                    count=count,
                    pages=pages,
                    rows=current,
                )
        rows.extend(current)
    if len(rows) != count:
        raise ValueError(f"survey pagination incomplete: expected {count}, got {len(rows)}")
    return rows


def _symbol(value: Any) -> str | None:
    symbol = str(value or "").strip().upper()
    if len(symbol) == 9 and symbol[6] == "." and symbol[:6].isdigit():
        return symbol if symbol.endswith((".SH", ".SZ", ".BJ")) else None
    return None


def normalize(rows: list[dict[str, Any]], year: int, month: int) -> pl.DataFrame:
    normalized = []
    for row in rows:
        symbol = _symbol(row.get("SECUCODE"))
        if symbol is None or str(row.get("RECEIVE_OBJECT_TYPE") or "") != "001":
            continue
        object_code = str(row.get("OBJECT_CODE") or "").strip()
        object_name = str(row.get("RECEIVE_OBJECT") or "").strip()
        object_key = object_code or object_name
        if not object_key:
            continue
        normalized.append(
            {
                "symbol": symbol,
                "notice_date": str(row.get("NOTICE_DATE") or "")[:10],
                "receive_start_date": str(row.get("RECEIVE_START_DATE") or "")[:10],
                "receive_end_date": str(row.get("RECEIVE_END_DATE") or "")[:10],
                "object_key": object_key,
                "org_type": str(row.get("ORG_TYPE") or "").strip(),
                "provider_sum": row.get("SUM"),
                "source_url": str(row.get("URL") or "").strip(),
            }
        )
    if not normalized:
        return pl.DataFrame()
    frame = pl.DataFrame(normalized, infer_schema_length=None).with_columns(
        pl.col("notice_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("receive_start_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("receive_end_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("provider_sum").cast(pl.Int64, strict=False),
    )
    frame = frame.filter(
        (pl.col("notice_date").dt.year() == year)
        & (pl.col("notice_date").dt.month() == month)
        & (
            pl.col("receive_end_date").is_null()
            | (pl.col("receive_end_date") <= pl.col("notice_date"))
        )
    )
    if frame.is_empty():
        return pl.DataFrame()
    return (
        frame.group_by("symbol", "notice_date")
        .agg(
            pl.col("object_key").n_unique().alias("institution_count"),
            pl.struct("receive_start_date", "receive_end_date")
            .n_unique()
            .alias("survey_session_count"),
            pl.col("provider_sum").max().alias("provider_sum_max"),
            pl.col("org_type").drop_nulls().unique().sort().str.join("|").alias("org_types"),
            pl.col("source_url").filter(pl.col("source_url") != "").first().alias("source_url"),
            pl.len().alias("institution_detail_rows"),
        )
        .with_columns(
            (
                pl.lit("survey-")
                + pl.col("symbol")
                + pl.lit("-")
                + pl.col("notice_date").dt.strftime("%Y%m%d")
            ).alias("event_id")
        )
        .select(
            "event_id",
            "notice_date",
            "symbol",
            "institution_count",
            "survey_session_count",
            "institution_detail_rows",
            "provider_sum_max",
            "org_types",
            "source_url",
        )
        .sort("notice_date", "symbol")
    )


def collect_month(fetch_json, root: Path, year: int, month: int) -> dict[str, Any]:
    cache_dir = root / "_page_cache" / f"year={year}" / f"month={month:02d}"
    rows = fetch_month(fetch_json, year, month, cache_dir=cache_dir)
    frame = normalize(rows, year, month)
    if frame.is_empty():
        raise ValueError(f"institutional survey returned no events for {year}-{month:02d}")
    path = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "month": month,
        "path": str(path),
        "page_cache": str(cache_dir),
        "raw_rows": len(rows),
        "events": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "first_notice_date": frame.get_column("notice_date").min(),
        "last_notice_date": frame.get_column("notice_date").max(),
        "max_institution_count": frame.get_column("institution_count").max(),
    }


def validate_period(year: int, month: int) -> None:
    if year < 2013 or year > 2026 or month not in range(1, 13):
        raise ValueError("collection must be one valid 2013-2026 month")


def run(data_dir: Path, year: int, month: int) -> dict[str, Any]:
    validate_period(year, month)
    client = EastmoneySurveyClient()
    result = collect_month(
        client.fetch,
        data_dir / "event_data" / "institutional_survey",
        year,
        month,
    )
    payload = {
        "dataset": "eastmoney_institutional_survey_attention",
        "outcome_fields_persisted": False,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.year, args.month)


if __name__ == "__main__":
    main()
