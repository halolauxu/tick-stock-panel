"""Collect one quarter of point-in-time public-fund ownership breadth."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl

CNINFO_BASE_URL = "https://webapi.cninfo.com.cn"
HOLDINGS_PATH = "/api/sysapi/p_sysapi1112"
ASSET_ALLOCATION_PATH = "/api/sysapi/p_sysapi1114"
START_YEAR = 2017
START_QUARTER = 1
END_YEAR = 2026
END_QUARTER = 2
AES_KEY = b"1234567887654321"
EVENT_SCHEMA = {
    "period_end": pl.Date,
    "available_after": pl.Date,
    "symbol": pl.String,
    "name_at_source": pl.String,
    "fund_coverage_count": pl.UInt32,
    "market_fund_count": pl.UInt32,
    "coverage_share": pl.Float64,
    "total_shares": pl.Float64,
    "total_market_value_cny": pl.Float64,
    "average_market_value_per_fund_cny": pl.Float64,
}


def period_end(year: int, quarter: int) -> date:
    month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    if quarter not in month_day:
        raise ValueError("quarter must be 1..4")
    month, day_number = month_day[quarter]
    return date(year, month, day_number)


def available_after(year: int, quarter: int) -> date:
    # Use the conservative end of the disclosure window rather than pretending
    # every constituent fund published on the first possible day.
    if quarter == 1:
        return date(year, 4, 30)
    if quarter == 2:
        return date(year, 8, 31)
    if quarter == 3:
        return date(year, 10, 31)
    if quarter == 4:
        return date(year + 1, 4, 30)
    raise ValueError("quarter must be 1..4")


def validate_period(year: int, quarter: int) -> None:
    current = year * 10 + quarter
    lower = START_YEAR * 10 + START_QUARTER
    upper = END_YEAR * 10 + END_QUARTER
    if quarter not in range(1, 5) or not lower <= current <= upper:
        raise ValueError("fund ownership collection must be within 2017Q1..2026Q2")


def _encrypted_access_key(now_seconds: int | None = None) -> str:
    openssl = shutil.which("openssl")
    if not openssl:
        raise RuntimeError("openssl is required for the CNInfo access key")
    timestamp = str(now_seconds if now_seconds is not None else int(time.time()))
    completed = subprocess.run(
        [
            openssl,
            "enc",
            "-aes-128-cbc",
            "-K",
            AES_KEY.hex(),
            "-iv",
            AES_KEY.hex(),
            "-nosalt",
            "-a",
            "-A",
        ],
        input=timestamp,
        text=True,
        capture_output=True,
        check=True,
    )
    value = completed.stdout.strip()
    try:
        base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise RuntimeError("openssl returned an invalid CNInfo access key") from exc
    return value


class CNInfoFundClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._http = httpx.Client(base_url=CNINFO_BASE_URL, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, params: dict[str, str] | None = None) -> list[dict]:
        response = self._http.post(
            path,
            params=params or {},
            headers={
                "Accept": "*/*",
                "Accept-Enckey": _encrypted_access_key(),
                "Origin": CNINFO_BASE_URL,
                "Referer": f"{CNINFO_BASE_URL}/",
                "User-Agent": "Mozilla/5.0 point-in-time-research-client",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("resultcode") != 200:
            message = str(payload.get("resultmsg") if isinstance(payload, dict) else "")
            raise RuntimeError(f"CNInfo fund metadata request failed: {message[:300]}")
        records = payload.get("records")
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise RuntimeError("CNInfo fund metadata records are malformed")
        return [dict(row) for row in records]

    def holdings(self, current_period_end: date) -> list[dict]:
        return self._post(
            HOLDINGS_PATH, {"rdate": current_period_end.isoformat()}
        )

    def market_statistics(self) -> list[dict]:
        return self._post(ASSET_ALLOCATION_PATH)


def _symbol(value: Any) -> str | None:
    code = str(value or "").strip()
    if len(code) != 6 or not code.isdigit():
        return None
    if code.startswith(("60", "68")):
        return f"{code}.SH"
    if code.startswith(("00", "30")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return None


def _market_fund_count(rows: list[dict[str, Any]], current: date) -> int:
    matches = [
        row for row in rows if str(row.get("ENDDATE") or "")[:10] == current.isoformat()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one market fund-count row for {current}")
    value = int(float(matches[0].get("F001N") or 0))
    if value <= 0:
        raise ValueError(f"invalid market fund count for {current}")
    return value


def normalize(
    holdings: list[dict[str, Any]],
    market_statistics: list[dict[str, Any]],
    year: int,
    quarter: int,
) -> pl.DataFrame:
    current = period_end(year, quarter)
    market_count = _market_fund_count(market_statistics, current)
    normalized = []
    for row in holdings:
        symbol = _symbol(row.get("SECCODE"))
        coverage = int(float(row.get("F001N") or 0))
        total_shares = float(row.get("F002N") or 0.0)
        market_value_cny = float(row.get("F003N") or 0.0) * 10_000.0
        if (
            symbol is None
            or str(row.get("ENDDATE") or "")[:10] != current.isoformat()
            or coverage <= 0
            or coverage > market_count
            or total_shares <= 0
            or market_value_cny <= 0
        ):
            continue
        normalized.append(
            {
                "period_end": current,
                "available_after": available_after(year, quarter),
                "symbol": symbol,
                "name_at_source": str(row.get("SECNAME") or "").strip(),
                "fund_coverage_count": coverage,
                "market_fund_count": market_count,
                "coverage_share": coverage / market_count,
                "total_shares": total_shares,
                "total_market_value_cny": market_value_cny,
                "average_market_value_per_fund_cny": market_value_cny / coverage,
            }
        )
    if not normalized:
        raise ValueError(f"no valid fund ownership rows for {current}")
    return (
        pl.DataFrame(normalized, schema=EVENT_SCHEMA)
        .unique(subset=["period_end", "symbol"], keep="last")
        .sort("symbol")
    )


def _atomic_write(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def collect_quarter(
    fetch_holdings: Callable[[date], list[dict]],
    fetch_market_statistics: Callable[[], list[dict]],
    root: Path,
    year: int,
    quarter: int,
) -> dict[str, Any]:
    validate_period(year, quarter)
    holdings = fetch_holdings(period_end(year, quarter))
    market_statistics = fetch_market_statistics()
    frame = normalize(holdings, market_statistics, year, quarter)
    target = root / f"year={year}" / f"quarter={quarter}" / "part.parquet"
    _atomic_write(frame, target)
    return {
        "year": year,
        "quarter": quarter,
        "path": str(target),
        "raw_rows": len(holdings),
        "events": frame.height,
        "market_fund_count": frame.get_column("market_fund_count")[0],
        "symbols": frame.get_column("symbol").n_unique(),
        "maximum_coverage": frame.get_column("fund_coverage_count").max(),
    }


def run(data_dir: Path, year: int, quarter: int) -> dict[str, Any]:
    client = CNInfoFundClient()
    try:
        result = collect_quarter(
            client.holdings,
            client.market_statistics,
            data_dir / "event_data" / "fund_ownership_breadth",
            year,
            quarter,
        )
    finally:
        client.close()
    payload = {
        "dataset": "cninfo_public_fund_ownership_breadth",
        "outcome_fields_persisted": False,
        "exact_component_announcement_times_available": False,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--quarter", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.year, args.quarter)


if __name__ == "__main__":
    main()
