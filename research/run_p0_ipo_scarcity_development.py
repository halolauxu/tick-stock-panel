"""Run the frozen development-only recent-IPO scarcity momentum study."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
MIN_LISTING_AGE_DAYS = 60
MAX_LISTING_AGE_DAYS = 365
MAX_FLOAT_SHARE_RATIO = 0.50
MIN_MOMENTUM_20D = 0.10
MAX_MOMENTUM_20D = 0.50
MIN_MEAN_AMOUNT_20D = 100_000_000.0
MAX_SIGNAL_DAY_RETURN = 0.05


def attach_ipo_point_in_time_data(
    panel: pl.DataFrame, data_dir: Path
) -> pl.DataFrame:
    research = data_dir / "research"
    universe_path = research / "historical_stock_universe_all_a.parquet"
    names_path = research / "historical_stock_names_all_a.parquet"
    if not universe_path.is_file() or not names_path.is_file():
        raise ValueError("all-A PIT security master is required")
    universe = (
        pl.read_parquet(universe_path)
        .with_columns(
            pl.col("list_date").cast(pl.Date, strict=False),
            pl.col("delist_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "list_date", "delist_date")
    )
    names = (
        pl.read_parquet(names_path)
        .with_columns(
            pl.col("start_date").cast(pl.Date, strict=False),
            pl.col("end_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "name", "start_date", "end_date")
        .sort(["symbol", "start_date"])
    )
    shares = baseline.load_share_history(data_dir)
    return (
        panel.with_columns(pl.col("date").cast(pl.Date))
        .join(universe, on="symbol", how="left")
        .filter(
            pl.col("list_date").is_not_null()
            & (pl.col("date") >= pl.col("list_date"))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("date") <= pl.col("delist_date"))
            )
        )
        .sort(["symbol", "date"])
        .join_asof(
            names,
            left_on="date",
            right_on="start_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("name").is_not_null()
            & (
                pl.col("end_date").is_null()
                | (pl.col("date") <= pl.col("end_date"))
            )
            & ~pl.col("name").str.to_uppercase().str.contains(r"(?:\*?ST|退)")
        )
        .join_asof(
            shares,
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            (pl.col("total_shares") > 0)
            & (pl.col("float_shares") > 0)
            & (pl.col("float_shares") <= pl.col("total_shares"))
        )
        .drop(
            "delist_date",
            "start_date",
            "end_date",
            "available_date",
        )
    )


def attach_scarcity_features(panel: pl.DataFrame) -> pl.DataFrame:
    return shared.attach_stock_features(panel).with_columns(
        (pl.col("date") - pl.col("list_date"))
        .dt.total_days()
        .alias("listing_age_days"),
        (pl.col("float_shares") / pl.col("total_shares")).alias(
            "float_share_ratio"
        ),
    )


def build_candidates(panel: pl.DataFrame) -> pl.DataFrame:
    weekly = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%G-%V").alias("week"),
        )
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
    )
    return (
        panel.join(weekly, left_on="date", right_on="signal_date", how="inner")
        .filter(
            pl.col("listing_age_days").is_between(
                MIN_LISTING_AGE_DAYS, MAX_LISTING_AGE_DAYS, closed="both"
            )
            & (pl.col("float_share_ratio") > 0)
            & (pl.col("float_share_ratio") <= MAX_FLOAT_SHARE_RATIO)
            & pl.col("stock_momentum_20d").is_between(
                MIN_MOMENTUM_20D, MAX_MOMENTUM_20D, closed="both"
            )
            & (pl.col("close") > pl.col("ma20"))
            & (pl.col("mean_amount_20d") >= MIN_MEAN_AMOUNT_20D)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("daily_return") <= MAX_SIGNAL_DAY_RETURN)
        )
        .sort(
            [
                "date",
                "stock_momentum_20d",
                "float_share_ratio",
                "amount",
                "symbol",
            ],
            descending=[False, True, False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= shared.TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "list_date",
            "listing_age_days",
            "float_share_ratio",
            "stock_momentum_20d",
            "daily_return",
            "mean_amount_20d",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START
    )
    if raw_source.is_empty():
        raise ValueError("no development daily data")
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = attach_ipo_point_in_time_data(raw_source, data_dir)
    panel = attach_scarcity_features(baseline.prepare_panel(pit))
    del pit
    gc.collect()
    candidates = build_candidates(panel)
    benchmark = shared.benchmark_metrics(panel)
    del panel
    gc.collect()
    result = shared.simulate(candidates, raw_source, all_dates, data_dir)
    decision = shared.evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-ipo-scarcity-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "initial_cash_cny": shared.INITIAL_CASH,
            "target_positions": shared.TARGET_POSITIONS,
            "listing_age_days": [MIN_LISTING_AGE_DAYS, MAX_LISTING_AGE_DAYS],
            "maximum_float_share_ratio": MAX_FLOAT_SHARE_RATIO,
            "stock_momentum_20d_range": [
                MIN_MOMENTUM_20D,
                MAX_MOMENTUM_20D,
            ],
            "minimum_mean_amount_20d_cny": MIN_MEAN_AMOUNT_20D,
            "maximum_signal_day_return": MAX_SIGNAL_DAY_RETURN,
            "execution": "weekly next trading day open, sells before buys",
            "benchmark": "PIT eligible all-A equal-weight daily return",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "signal_rows": candidates.height,
            "rebalance_days": candidates.get_column("entry_date").n_unique(),
            "signal_symbols": candidates.get_column("symbol").n_unique(),
        },
        "benchmark": benchmark,
        "strategy": result,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "data": payload["data"],
                "benchmark": benchmark,
                "strategy": {
                    "metrics": result["metrics"],
                    "execution": result["execution"],
                    "integrity": result["integrity"],
                    "account": result["account"],
                },
                "decision": decision,
                "output": str(output),
                "sha256": digest,
            },
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
        default=Path(
            "/app/data/research/p0_ipo_scarcity_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
