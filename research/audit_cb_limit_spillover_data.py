"""Audit causal data for stock-limit-up demand spillover into convertible bonds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_cb_stock_lead_lag_discovery as lead  # noqa: E402
import run_p0_microcap_baseline as stock_rules  # noqa: E402

START = date(2026, 8, 3)
END = date(2026, 8, 28)
DISCOVERY_DAYS = 10
MIN_SIGNALS_PER_HALF = 30
MIN_EVENT_DAYS_PER_HALF = 8
MIN_EXECUTION_BAR_COVERAGE = 0.90


def build_stock_limit_reference(
    stock_daily: pl.DataFrame,
    names: pl.DataFrame,
) -> pl.DataFrame:
    """Compute each stock's point-in-time limit-up price and exclude ST names."""
    point_in_time_names = (
        stock_daily.sort(["symbol", "date"])
        .join_asof(
            names.select("symbol", "name", "start_date", "end_date").sort(
                ["symbol", "start_date"]
            ),
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
    )
    limit_pct = (
        pl.when(
            pl.col("symbol").str.starts_with("30")
            | pl.col("symbol").str.starts_with("68")
        )
        .then(pl.lit(0.20))
        .otherwise(pl.lit(0.10))
    )
    return (
        point_in_time_names.sort(["symbol", "date"])
        .with_columns(
            pl.col("close").shift(1).over("symbol").alias("previous_close")
        )
        .filter(
            pl.col("date").is_between(START, END, closed="both")
            & (pl.col("previous_close") > 0)
        )
        .with_columns(
            stock_rules.polars_limit_price(
                pl.col("previous_close"), limit_pct, up=True
            ).alias("limit_up_price")
        )
        .select(
            pl.col("symbol").alias("stock_symbol"),
            "date",
            "name",
            "previous_close",
            "limit_up_price",
        )
        .sort(["date", "stock_symbol"])
    )


def build_trigger_events(
    universe: pl.DataFrame,
    reference: pl.DataFrame,
    stock_minute: pl.DataFrame,
) -> pl.DataFrame:
    """Keep the first minute-close limit lock for each eligible bond-day."""
    minutes = (
        stock_minute.with_columns(
            pl.col("datetime").dt.date().alias("date"),
            pl.col("datetime").dt.time().alias("clock"),
        )
        .filter(
            pl.col("clock").is_between(time(9, 30), time(11, 14))
            | pl.col("clock").is_between(time(13, 0), time(14, 44))
        )
        .rename({"symbol": "stock_symbol"})
    )
    return (
        universe.join(reference, on=["stock_symbol", "date"], how="inner")
        .join(minutes, on=["stock_symbol", "date"], how="inner")
        .filter(
            (pl.col("amount") > 0)
            & (pl.col("close") >= pl.col("limit_up_price") - 0.005)
        )
        .sort(["date", "symbol", "datetime"])
        .unique(subset=["date", "symbol"], keep="first", maintain_order=True)
        .select(
            "date",
            "datetime",
            "symbol",
            "stock_symbol",
            "previous_close",
            "limit_up_price",
            pl.col("clock").alias("trigger_clock"),
        )
        .sort(["datetime", "symbol"])
    )


def attach_execution_bar_audit(
    events: pl.DataFrame,
    cb_minute: pl.DataFrame,
) -> pl.DataFrame:
    """Check next-minute entry and 15-minute exit bars without calculating returns."""
    prepared = (
        lead._with_session(cb_minute)
        .sort(["date", "symbol", "session", "datetime"])
        .with_columns(
            pl.col("datetime")
            .shift(-1)
            .over(["date", "symbol", "session"])
            .alias("entry_datetime"),
            pl.col("open")
            .shift(-1)
            .over(["date", "symbol", "session"])
            .alias("entry_open"),
            pl.col("amount_cny")
            .shift(-1)
            .over(["date", "symbol", "session"])
            .alias("entry_amount_cny"),
            pl.col("datetime")
            .shift(-15)
            .over(["date", "symbol", "session"])
            .alias("exit_datetime"),
            pl.col("close")
            .shift(-15)
            .over(["date", "symbol", "session"])
            .alias("exit_close"),
            pl.col("amount_cny")
            .shift(-15)
            .over(["date", "symbol", "session"])
            .alias("exit_amount_cny"),
        )
        .select(
            "symbol",
            "datetime",
            "entry_datetime",
            "entry_open",
            "entry_amount_cny",
            "exit_datetime",
            "exit_close",
            "exit_amount_cny",
        )
    )
    return (
        events.join(prepared, on=["symbol", "datetime"], how="left")
        .with_columns(
            (
                (pl.col("entry_datetime") == pl.col("datetime") + pl.duration(minutes=1))
                & (pl.col("exit_datetime") == pl.col("datetime") + pl.duration(minutes=15))
                & (pl.col("entry_open") > 0)
                & (pl.col("exit_close") > 0)
                & (pl.col("entry_amount_cny") > 0)
                & (pl.col("exit_amount_cny") > 0)
            )
            .fill_null(False)
            .alias("execution_bars_usable")
        )
        .sort(["datetime", "symbol"])
    )


def summarize_half(frame: pl.DataFrame) -> dict[str, Any]:
    usable = frame.filter(pl.col("execution_bars_usable"))
    return {
        "signals": frame.height,
        "event_days": frame.get_column("date").n_unique(),
        "cb_symbols": frame.get_column("symbol").n_unique(),
        "usable_execution_bars": usable.height,
        "execution_bar_coverage": usable.height / frame.height if frame.height else 0.0,
    }


def evaluate_data_gate(
    events: pl.DataFrame,
    market_dates: list[date],
) -> dict[str, Any]:
    discovery_dates = market_dates[:DISCOVERY_DAYS]
    confirmation_dates = market_dates[DISCOVERY_DAYS:]
    discovery = summarize_half(events.filter(pl.col("date").is_in(discovery_dates)))
    confirmation = summarize_half(
        events.filter(pl.col("date").is_in(confirmation_dates))
    )
    checks = {
        "exactly_20_market_days": len(market_dates) == 20,
        "discovery_at_least_30_signals": discovery["signals"]
        >= MIN_SIGNALS_PER_HALF,
        "confirmation_at_least_30_signals": confirmation["signals"]
        >= MIN_SIGNALS_PER_HALF,
        "discovery_at_least_8_event_days": discovery["event_days"]
        >= MIN_EVENT_DAYS_PER_HALF,
        "confirmation_at_least_8_event_days": confirmation["event_days"]
        >= MIN_EVENT_DAYS_PER_HALF,
        "discovery_execution_bars_at_least_90pct": discovery[
            "execution_bar_coverage"
        ]
        >= MIN_EXECUTION_BAR_COVERAGE,
        "confirmation_execution_bars_at_least_90pct": confirmation[
            "execution_bar_coverage"
        ]
        >= MIN_EXECUTION_BAR_COVERAGE,
    }
    return {
        "discovery_dates": discovery_dates,
        "confirmation_dates": confirmation_dates,
        "discovery": discovery,
        "confirmation": confirmation,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _partition_paths(root: Path, dataset: str, start: date, end: date) -> list[Path]:
    return lead._partition_paths(root, dataset, start, end)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    cb_root = data_dir / "convertible_bond"
    basic = pl.read_parquet(cb_root / "basic" / "part.parquet")
    cb_daily_paths = _partition_paths(cb_root, "daily", START, END)
    cb_minute_paths = _partition_paths(cb_root, "minute", START, END)
    stock_minute_paths = _partition_paths(data_dir, "kline_minute", START, END)
    stock_daily_paths = _partition_paths(
        data_dir, "kline_daily", date(2026, 7, 1), END
    )
    if not all(
        (cb_daily_paths, cb_minute_paths, stock_minute_paths, stock_daily_paths)
    ):
        raise ValueError("CB and stock daily/minute partitions are required")
    cb_daily = pl.read_parquet(cb_daily_paths).sort(["date", "symbol"])
    universe = lead.build_causal_universe(basic, cb_daily)
    stock_symbols = universe.get_column("stock_symbol").unique().to_list()
    stock_daily = (
        pl.scan_parquet(stock_daily_paths)
        .filter(pl.col("symbol").is_in(stock_symbols))
        .collect()
    )
    names = pl.read_parquet(
        data_dir / "research" / "historical_stock_names_all_a.parquet"
    )
    reference = build_stock_limit_reference(stock_daily, names)
    stock_minute = (
        pl.scan_parquet(stock_minute_paths)
        .filter(pl.col("symbol").is_in(stock_symbols))
        .collect()
    )
    events = build_trigger_events(universe, reference, stock_minute)
    cb_minute = pl.read_parquet(cb_minute_paths)
    audited_events = attach_execution_bar_audit(events, cb_minute)
    market_dates = sorted(cb_daily.get_column("date").unique().to_list())
    gate = evaluate_data_gate(audited_events, market_dates)
    payload = {
        "schema_version": "p0-cb-limit-spillover-data-audit-v1",
        "audited_at": "2026-08-31",
        "period": {"start": START, "end": END},
        "outcomes_read": False,
        "mechanism": "stock minute close locks at limit-up; corresponding CB remains T+0 tradable",
        "data": {
            "market_days": len(market_dates),
            "eligible_cb_symbol_days": universe.height,
            "eligible_cb_symbols": universe.get_column("symbol").n_unique(),
            "mapped_stock_symbols": len(stock_symbols),
            "stock_minute_rows": stock_minute.height,
            "cb_minute_rows": cb_minute.height,
            "trigger_signals": audited_events.height,
            "trigger_days": audited_events.get_column("date").n_unique(),
        },
        "gate": gate,
        "decision": {
            "verdict": "FREEZE_DISCOVERY_CONTRACT" if gate["passed"] else "TERMINATE_BEFORE_OUTCOMES",
            "counts_toward_50pct_goal": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": sha256},
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
        default=Path("/app/data/research/p0_cb_limit_spillover_data_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
