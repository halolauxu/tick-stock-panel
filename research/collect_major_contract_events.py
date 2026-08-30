"""Collect structured Eastmoney A-share major-contract events by year."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import polars as pl

API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPTA_WEB_ZDHT_LIST"
PUBLIC_TOKEN = "894050c76af8597a853f5b408b759f5d"
PAGE_SIZE = 500
MAX_PAGES = 100
MIN_INTERVAL_SECONDS = 0.12

FetchJSON = Callable[[dict[str, str]], dict[str, Any]]

_RATIO_PATTERNS = (
    re.compile(r"占[^%]{0,80}(?:营业收入|营收)[^%]{0,30}?([0-9]+(?:\.[0-9]+)?)%"),
    re.compile(r"(?:营业收入|营收)[^%]{0,50}?([0-9]+(?:\.[0-9]+)?)%"),
)


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


class EastmoneyContractClient:
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
            raise ValueError("Eastmoney major-contract response is not an object")
        return payload


def _params(year: int, page: int) -> dict[str, str]:
    return {
        "sortColumns": "DIM_RDATE",
        "sortTypes": "1",
        "pageSize": str(PAGE_SIZE),
        "pageNumber": str(page),
        "columns": "ALL",
        "token": PUBLIC_TOKEN,
        "reportName": REPORT_NAME,
        "filter": (f"(DIM_RDATE>='{year}-01-01')(DIM_RDATE<='{year}-12-31')"),
    }


def _page_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    if payload.get("success") is not True:
        raise ValueError("Eastmoney major-contract request was not successful")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Eastmoney major-contract response has no result object")
    rows = result.get("data") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Eastmoney major-contract data is malformed")
    return (
        [dict(row) for row in rows],
        int(result.get("count") or 0),
        int(result.get("pages") or 0),
    )


def fetch_year(fetch_json: FetchJSON, year: int) -> list[dict[str, Any]]:
    first, total_count, total_pages = _page_rows(fetch_json(_params(year, 1)))
    if total_pages > MAX_PAGES:
        raise ValueError(f"major-contract year exceeds {MAX_PAGES} pages")
    expected_pages = math.ceil(total_count / PAGE_SIZE) if total_count else 0
    if total_pages != expected_pages:
        raise ValueError("major-contract page count is inconsistent")
    rows = list(first)
    for page in range(2, total_pages + 1):
        current, current_count, current_pages = _page_rows(
            fetch_json(_params(year, page))
        )
        if current_count != total_count or current_pages != total_pages:
            raise ValueError("major-contract pagination changed during collection")
        rows.extend(current)
    if len(rows) != total_count:
        raise ValueError(
            f"major-contract pagination incomplete: expected {total_count}, got {len(rows)}"
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


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_id(row: dict[str, Any]) -> str:
    identity = "\x1f".join(
        str(row.get(field) or "").strip()
        for field in (
            "DIM_SCODE",
            "DIM_RDATE",
            "SECURITYCODE",
            "CONTRACTNAME",
            "CONTRACTTYPE",
            "COUNTERPARTY",
            "SIGNDATE",
            "AMOUNTS",
        )
    )
    return "mc-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def parse_revenue_ratio_pct(row: dict[str, Any]) -> tuple[float | None, str | None]:
    explicit = _finite_float(row.get("ZSNDYYSRBL"))
    if explicit is not None and 0 < explicit <= 10_000:
        return explicit, "provider_previous_revenue_ratio"
    amount = _finite_float(row.get("AMOUNTS"))
    previous_revenue = _finite_float(row.get("SNDYYSR"))
    if amount is not None and previous_revenue is not None and previous_revenue > 0:
        ratio = amount / previous_revenue * 100.0
        if 0 < ratio <= 10_000:
            return ratio, "amount_div_previous_revenue"
    text = str(row.get("SIGNEFFECT") or "").replace("％", "%")
    for pattern in _RATIO_PATTERNS:
        match = pattern.search(text)
        if match:
            ratio = float(match.group(1))
            if 0 < ratio <= 10_000:
                return ratio, "announcement_effect_text"
    return None, None


def normalize(rows: list[dict[str, Any]], year: int) -> pl.DataFrame:
    normalized = []
    for row in rows:
        symbol = _symbol(row.get("SECURITYCODE"))
        ratio, ratio_source = parse_revenue_ratio_pct(row)
        if symbol is None:
            continue
        normalized.append(
            {
                "event_id": _event_id(row),
                "source_security_id": str(row.get("DIM_SCODE") or "").strip(),
                "ann_date": str(row.get("DIM_RDATE") or "")[:10],
                "symbol": symbol,
                "company_name": str(row.get("SECURITYSHORTNAME") or "").strip(),
                "contract_name": str(row.get("CONTRACTNAME") or "").strip(),
                "contract_type_code": str(row.get("CONTRACTTYPE") or "").strip(),
                "contract_type_name": str(row.get("CONTRACTTYPENAME") or "").strip(),
                "signatory": str(row.get("SIGNATORY") or "").strip(),
                "signatory_relation": str(row.get("SIGNATORYRELNAME") or "").strip(),
                "counterparty": str(row.get("COUNTERPARTY") or "").strip(),
                "counterparty_relation": str(
                    row.get("COUNTERPARTYRELNAME") or ""
                ).strip(),
                "sign_date": str(row.get("SIGNDATE") or "")[:10],
                "contract_amount_cny": _finite_float(row.get("AMOUNTS")),
                "previous_revenue_cny": _finite_float(row.get("SNDYYSR")),
                "revenue_ratio_pct": ratio,
                "ratio_source": ratio_source,
                "is_abolished": str(row.get("ISABOLISHED") or "").strip(),
                "contents": str(row.get("CONTENTS") or "").strip(),
                "stated_effect": str(row.get("SIGNEFFECT") or "").strip(),
            }
        )
    if not normalized:
        return pl.DataFrame()
    return (
        pl.DataFrame(normalized, infer_schema_length=None)
        .with_columns(
            pl.col("ann_date").str.to_date("%Y-%m-%d", strict=False),
            pl.col("sign_date").str.to_date("%Y-%m-%d", strict=False),
        )
        .filter(pl.col("ann_date").dt.year() == year)
        .drop_nulls(["event_id", "ann_date", "symbol"])
        .filter(pl.col("event_id") != "")
        .unique(subset=["event_id", "symbol", "ann_date"], keep="last")
        .sort(["ann_date", "symbol", "event_id"])
    )


def collect_year(fetch_json: FetchJSON, root: Path, year: int) -> dict[str, Any]:
    rows = fetch_year(fetch_json, year)
    frame = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(f"major-contract dataset returned no A-share rows for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    return {
        "year": year,
        "path": str(path),
        "raw_rows": len(rows),
        "rows": frame.height,
        "symbols": frame.get_column("symbol").n_unique(),
        "ratio_coverage": frame.get_column("revenue_ratio_pct").is_not_null().mean(),
        "first_ann_date": frame.get_column("ann_date").min(),
        "last_ann_date": frame.get_column("ann_date").max(),
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 2:
        raise ValueError("each collection run is bounded to at most three years")
    client = EastmoneyContractClient()
    root = data_dir / "event_data" / "major_contract"
    results = [
        collect_year(client.fetch, root, year)
        for year in range(start_year, end_year + 1)
    ]
    payload = {
        "dataset": "eastmoney_major_contract",
        "source": API_URL,
        "report_name": REPORT_NAME,
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
