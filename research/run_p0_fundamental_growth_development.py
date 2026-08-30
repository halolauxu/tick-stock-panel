"""Run the frozen development-only fundamental growth persistence study."""
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
MAX_REPORT_AGE_DAYS = 180
MIN_REVENUE_YOY = 15.0
MAX_REVENUE_YOY = 100.0
MIN_NET_INCOME_YOY = 30.0
MAX_NET_INCOME_YOY = 200.0
MIN_ROE = 8.0
MIN_NET_MARGIN = 5.0
MIN_OPERATING_CASH_TO_REVENUE = 5.0
MAX_DEBT_RATIO = 70.0
MIN_DAILY_AMOUNT = 50_000_000.0


def load_metrics(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "financials" / "metrics" / "part.parquet"
    if not path.is_file():
        raise ValueError("financial metrics history is required")
    needed = (
        "symbol",
        "period_end",
        "announce_date",
        "roe",
        "net_margin",
        "debt_to_asset_ratio",
        "revenue_yoy",
        "net_income_yoy",
        "operating_cash_to_revenue",
    )
    frame = pl.read_parquet(path)
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f"financial metrics missing columns: {sorted(missing)}")
    return (
        frame.select(needed)
        .with_columns(
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("report_period_end"),
            pl.col("announce_date")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("report_announce_date"),
        )
        .drop("period_end", "announce_date")
        .filter(
            pl.col("report_period_end").is_not_null()
            & pl.col("report_announce_date").is_not_null()
            & (pl.col("report_announce_date") >= pl.col("report_period_end"))
        )
        .sort(["symbol", "report_announce_date", "report_period_end"])
        .unique(subset=["symbol", "report_announce_date"], keep="last")
        .sort(["symbol", "report_announce_date"])
    )


def attach_latest_metrics(
    panel: pl.DataFrame, metrics: pl.DataFrame
) -> pl.DataFrame:
    return (
        panel.sort(["symbol", "date"])
        .join_asof(
            metrics,
            left_on="date",
            right_on="report_announce_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("date") - pl.col("report_announce_date"))
            .dt.total_days()
            .alias("report_age_days")
        )
        .with_columns(
            pl.when(pl.col("date") > pl.col("report_announce_date"))
            .then(pl.col("report_age_days"))
            .otherwise(None)
            .alias("report_age_days")
        )
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
            pl.col("report_age_days").is_between(
                1, MAX_REPORT_AGE_DAYS, closed="both"
            )
            & pl.col("revenue_yoy").is_between(
                MIN_REVENUE_YOY, MAX_REVENUE_YOY, closed="both"
            )
            & pl.col("net_income_yoy").is_between(
                MIN_NET_INCOME_YOY, MAX_NET_INCOME_YOY, closed="both"
            )
            & (pl.col("roe") >= MIN_ROE)
            & (pl.col("net_margin") >= MIN_NET_MARGIN)
            & (
                pl.col("operating_cash_to_revenue")
                >= MIN_OPERATING_CASH_TO_REVENUE
            )
            & (pl.col("debt_to_asset_ratio") <= MAX_DEBT_RATIO)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("amount") >= MIN_DAILY_AMOUNT)
        )
        .sort(
            [
                "date",
                "net_income_yoy",
                "revenue_yoy",
                "roe",
                "report_announce_date",
                "symbol",
            ],
            descending=[False, True, True, True, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank")
        )
        .filter(pl.col("cap_rank") <= shared.TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "report_period_end",
            "report_announce_date",
            "report_age_days",
            "revenue_yoy",
            "net_income_yoy",
            "roe",
            "net_margin",
            "operating_cash_to_revenue",
            "debt_to_asset_ratio",
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
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = attach_latest_metrics(baseline.prepare_panel(pit), load_metrics(data_dir))
    del pit
    gc.collect()
    candidates = build_candidates(panel)
    benchmark = shared.benchmark_metrics(panel)
    del panel
    gc.collect()
    result = shared.simulate(candidates, raw_source, all_dates, data_dir)
    decision = shared.evaluate_gate(result, benchmark)
    payload = {
        "schema_version": "p0-fundamental-growth-development-v1",
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
            "maximum_report_age_days": MAX_REPORT_AGE_DAYS,
            "revenue_yoy_range_pct": [MIN_REVENUE_YOY, MAX_REVENUE_YOY],
            "net_income_yoy_range_pct": [
                MIN_NET_INCOME_YOY,
                MAX_NET_INCOME_YOY,
            ],
            "minimum_roe_pct": MIN_ROE,
            "minimum_net_margin_pct": MIN_NET_MARGIN,
            "minimum_operating_cash_to_revenue_pct": MIN_OPERATING_CASH_TO_REVENUE,
            "maximum_debt_ratio_pct": MAX_DEBT_RATIO,
            "minimum_signal_day_amount_cny": MIN_DAILY_AMOUNT,
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
            "/app/data/research/p0_fundamental_growth_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
