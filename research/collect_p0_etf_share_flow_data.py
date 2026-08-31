"""Collect ETF share-history metadata without opening price outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import secrets_store  # noqa: E402
from app.plugins.tushare.client import TushareClient  # noqa: E402

START = date(2013, 1, 1)
END = date(2020, 12, 31)
FIELDS = ("ts_code", "trade_date", "fd_share")
OUTCOME_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "return",
    "future_return",
    "forward_return",
    "net_return",
}
MIN_MASTER_SYMBOLS = 100
MIN_SYMBOL_COVERAGE = 0.80
MIN_YEAR_SYMBOLS = 20
MIN_YEAR_DATES = 100


def eligible_master(master: pl.DataFrame) -> pl.DataFrame:
    """Keep point-in-time stock ETFs that overlap the frozen metadata window."""
    return (
        master.filter(
            (pl.col("fund_type") == "股票型")
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
            & pl.col("list_date").is_not_null()
            & (pl.col("list_date") <= pl.lit(END))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("delist_date") >= pl.lit(START))
            )
        )
        .unique("symbol", keep="last")
        .sort("symbol")
    )


def normalize_shares(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "date": pl.Date,
                "shares_10k": pl.Float64,
            }
        )
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename(
            {
                "ts_code": "symbol",
                "trade_date": "date",
                "fd_share": "shares_10k",
            }
        )
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("shares_10k").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ)$")
            & pl.col("date").is_between(START, END, closed="both")
            & pl.col("shares_10k").is_not_null()
            & (pl.col("shares_10k") > 0)
        )
        .unique(["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )


def audit(master: pl.DataFrame, shares: pl.DataFrame) -> dict[str, Any]:
    master_symbols = set(master["symbol"].to_list())
    share_symbols = set(shares["symbol"].unique().to_list())
    coverage = len(share_symbols & master_symbols) / len(master_symbols)
    yearly = (
        shares.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("date").n_unique().alias("observation_dates"),
            pl.len().alias("rows"),
        )
        .sort("year")
    )
    expected_years = set(range(START.year, END.year + 1))
    yearly_rows = {row["year"]: row for row in yearly.to_dicts()}
    weak_years = [
        year
        for year in sorted(expected_years)
        if year not in yearly_rows
        or yearly_rows[year]["symbols"] < MIN_YEAR_SYMBOLS
        or yearly_rows[year]["observation_dates"] < MIN_YEAR_DATES
    ]
    checks = {
        "master_has_at_least_100_stock_etfs": master.height >= MIN_MASTER_SYMBOLS,
        "symbol_date_unique": shares.unique(["symbol", "date"]).height
        == shares.height,
        "shares_positive": shares.filter(pl.col("shares_10k") <= 0).height == 0,
        "outcome_fields_absent": not (OUTCOME_FIELDS & set(shares.columns)),
        "symbol_coverage_at_least_80pct": coverage >= MIN_SYMBOL_COVERAGE,
        "every_year_has_20_symbols_and_100_dates": not weak_years,
    }
    integrity = (
        "master_has_at_least_100_stock_etfs",
        "symbol_date_unique",
        "shares_positive",
        "outcome_fields_absent",
    )
    if not all(checks[name] for name in integrity):
        status = "DATA_GAP"
    elif all(checks.values()):
        status = "SAMPLE_SUFFICIENT"
    else:
        status = "SAMPLE_SPARSE"
    return {
        "status": status,
        "price_data_read": False,
        "future_returns_read": False,
        "period": {"start": START, "end": END},
        "counts": {
            "eligible_master_symbols": master.height,
            "share_rows": shares.height,
            "share_symbols": len(share_symbols),
            "symbol_coverage": coverage,
            "observation_dates": shares["date"].n_unique(),
        },
        "yearly": yearly.to_dicts(),
        "weak_years": weak_years,
        "checks": checks,
    }


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    source_master = (
        data_dir / "research" / "etf_cross_asset_v2" / "master.parquet"
    )
    master = eligible_master(pl.read_parquet(source_master))
    token = secrets_store.get_env_backed_secret(
        "tushare_api_key", "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    frames: list[pl.DataFrame] = []
    try:
        for index, symbol in enumerate(master["symbol"].to_list(), start=1):
            frame = normalize_shares(
                client.query(
                    "fund_share",
                    {
                        "ts_code": symbol,
                        "start_date": START.strftime("%Y%m%d"),
                        "end_date": END.strftime("%Y%m%d"),
                    },
                    FIELDS,
                )
            )
            if not frame.is_empty():
                frames.append(frame)
            if index == 1 or index % 25 == 0 or index == master.height:
                print(
                    f"etf_share_progress={index}/{master.height} "
                    f"covered={len(frames)}",
                    flush=True,
                )
    finally:
        client.close()
    shares = (
        pl.concat(frames, how="vertical_relaxed")
        if frames
        else normalize_shares([])
    )
    shares = (
        shares.join(
            master.select("symbol", "list_date", "delist_date"),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("date") >= pl.col("list_date"))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("date") <= pl.col("delist_date"))
            )
        )
        .drop("list_date", "delist_date")
        .unique(["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )
    root = data_dir / "research" / "etf_share_flow"
    _atomic_parquet(master, root / "master.parquet")
    _atomic_parquet(shares, root / "share_history.parquet")
    payload = {
        "schema_version": "p0-etf-share-flow-data-v1",
        "contract_frozen": "2026-08-31",
        **audit(master, shares),
        "artifacts": {
            "master": str(root / "master.parquet"),
            "share_history": str(root / "share_history.parquet"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": digest},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_etf_share_flow_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
