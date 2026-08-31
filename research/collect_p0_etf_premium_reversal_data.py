"""Collect point-in-time ETF NAV data without evaluating strategy returns."""

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
NAV_FIELDS = (
    "ts_code",
    "ann_date",
    "nav_date",
    "unit_nav",
    "accum_nav",
    "adj_nav",
)


def normalize_nav(rows: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
    schema = {
        "symbol": pl.Utf8,
        "ann_date": pl.Date,
        "nav_date": pl.Date,
        "unit_nav": pl.Float64,
        "accum_nav": pl.Float64,
        "adj_nav": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
            pl.col("ann_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("nav_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in ("unit_nav", "accum_nav", "adj_nav")
            ],
        )
        .filter(
            (pl.col("symbol") == symbol)
            & pl.col("nav_date").is_between(START, END, closed="both")
            & pl.col("ann_date").is_not_null()
            & pl.col("nav_date").is_not_null()
        )
        .select(*schema)
        .sort(["symbol", "nav_date", "ann_date"])
        .unique(["symbol", "nav_date"], keep="first", maintain_order=True)
    )


def _atomic_parquet(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


def audit(
    master: pl.DataFrame, daily: pl.DataFrame, nav: pl.DataFrame
) -> dict[str, Any]:
    priced_symbols = set(daily["symbol"].unique().to_list())
    nav_symbols = set(nav["symbol"].unique().to_list()) if nav.height else set()
    covered = priced_symbols.intersection(nav_symbols)
    exact_matches = nav.join(
        daily.select("symbol", pl.col("date").alias("nav_date")).unique(),
        on=["symbol", "nav_date"],
        how="inner",
    )
    lag = nav.with_columns(
        (pl.col("ann_date") - pl.col("nav_date")).dt.total_days().alias("lag_days")
    )
    coverage = len(covered) / len(priced_symbols) if priced_symbols else 0.0
    exact_match_rate = exact_matches.height / nav.height if nav.height else 0.0
    checks = {
        "master_keys_unique": master.unique("symbol").height == master.height,
        "nav_keys_unique": nav.unique(["symbol", "nav_date"]).height == nav.height,
        "nav_symbol_coverage_at_least_90pct": coverage >= 0.90,
        "at_least_200000_nav_rows": nav.height >= 200_000,
        "announcement_not_before_nav": lag.filter(pl.col("lag_days") < 0).is_empty(),
        "unit_nav_positive": nav.filter(
            pl.col("unit_nav").is_null() | (pl.col("unit_nav") <= 0)
        ).is_empty(),
        "exact_market_date_match_at_least_90pct": exact_match_rate >= 0.90,
    }
    lag_values = lag["lag_days"].drop_nulls()
    return {
        "status": "DATA_QUALIFIED" if all(checks.values()) else "DATA_GAP",
        "returns_evaluated": False,
        "strategy_metrics_computed": False,
        "counts": {
            "stock_etf_master": master.height,
            "priced_symbols": len(priced_symbols),
            "nav_symbols": len(nav_symbols),
            "covered_priced_symbols": len(covered),
            "nav_symbol_coverage": coverage,
            "nav_rows": nav.height,
            "exact_market_date_matches": exact_matches.height,
            "exact_market_date_match_rate": exact_match_rate,
            "announcement_lag_median_days": lag_values.median()
            if lag_values.len()
            else None,
            "announcement_lag_p95_days": (
                lag_values.quantile(0.95, interpolation="nearest")
                if lag_values.len()
                else None
            ),
        },
        "checks": checks,
    }


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    source = data_dir / "research" / "etf_cross_asset_v2"
    master = pl.read_parquet(source / "master.parquet").filter(
        pl.col("fund_type") == "股票型"
    )
    daily = pl.read_parquet(source / "daily_raw.parquet").filter(
        pl.col("symbol").is_in(master["symbol"])
    )
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    target_root = data_dir / "research" / "etf_premium_reversal"
    client = TushareClient(token)
    frames: list[pl.DataFrame] = []
    try:
        symbols = master["symbol"].to_list()
        for index, symbol in enumerate(symbols, start=1):
            target = target_root / "nav" / f"symbol={symbol}" / "part.parquet"
            if target.exists():
                frame = pl.read_parquet(target)
            else:
                frame = normalize_nav(
                    client.query(
                        "fund_nav",
                        {
                            "ts_code": symbol,
                            "start_date": START.strftime("%Y%m%d"),
                            "end_date": END.strftime("%Y%m%d"),
                        },
                        NAV_FIELDS,
                    ),
                    symbol,
                )
                _atomic_parquet(frame, target)
            frames.append(frame)
            if index == 1 or index % 25 == 0 or index == len(symbols):
                print(f"nav_progress={index}/{len(symbols)}", flush=True)
    finally:
        client.close()
    nav = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    summary = audit(master, daily, nav)
    payload = {
        "schema_version": "p0-etf-premium-reversal-data-v1",
        "contract_frozen": "2026-08-31",
        "period": {"start": START, "end": END},
        **summary,
        "artifacts": {
            "source_master": str(source / "master.parquet"),
            "source_daily": str(source / "daily_raw.parquet"),
            "nav_root": str(target_root / "nav"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "sha256": digest}, ensure_ascii=False, indent=2, default=str
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_etf_premium_reversal_data_audit.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
