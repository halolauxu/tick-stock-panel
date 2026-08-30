"""Run the frozen development-only first-limit-down rescue study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.run_p0_forecast_drift_development import (  # noqa: E402
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

START = date(2013, 8, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 2, 28)
HOLD_TRADING_DAYS = 1
MAX_EXIT_DELAY = 20
LOOKBACK_TRADING_DAYS = 60
SIGNAL_AMOUNT_FLOOR = 50_000_000.0
RESCUE_FROM_LIMIT_FLOOR = 0.02
MAX_CLOSE_VS_REFERENCE = 0.98
CATEGORY = "first_limit_down_rescue"
MAIN_BOARD_PATTERN = r"^(?:00\d{4}\.SZ|60\d{4}\.SH)$"


def build_rescue_events(panel: pl.DataFrame) -> pl.DataFrame:
    ordered = panel.sort(["symbol", "date"])
    touched = (
        pl.col("limit_down_price").is_not_null()
        & (pl.col("raw_low") <= pl.col("limit_down_price") + 0.005)
    )
    return (
        ordered.with_columns(touched.alias("touched_limit_down"))
        .with_columns(
            pl.col("touched_limit_down")
            .cast(pl.Int8)
            .shift(1)
            .rolling_max(
                window_size=LOOKBACK_TRADING_DAYS,
                min_samples=1,
            )
            .over("symbol")
            .fill_null(0)
            .alias("prior_60d_limit_down_touch")
        )
        .filter(
            pl.col("date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
            & ~pl.col("excluded_name").fill_null(True)
            & pl.col("touched_limit_down")
            & (pl.col("prior_60d_limit_down_touch") == 0)
            & (pl.col("amount") >= SIGNAL_AMOUNT_FLOOR)
            & (
                pl.col("raw_close")
                >= pl.col("limit_down_price")
                * (1.0 + RESCUE_FROM_LIMIT_FLOOR)
            )
            & (
                pl.col("raw_close")
                <= (pl.col("limit_down_price") / 0.90)
                * MAX_CLOSE_VS_REFERENCE
            )
        )
        .select(
            "symbol",
            pl.col("date").alias("ann_date"),
            pl.lit(CATEGORY).alias("category"),
            "raw_low",
            "raw_close",
            "limit_down_price",
            pl.col("amount").alias("signal_amount"),
        )
        .sort(["ann_date", "symbol"])
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    events = build_rescue_events(panel)
    trades = build_trades(
        events,
        panel,
        holding_trading_days=HOLD_TRADING_DAYS,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    summary = summarize_category(
        trades,
        CATEGORY,
        positive_categories=(CATEGORY,),
        min_tradable_events=500,
        min_announcement_days=200,
    )
    payload = {
        "schema_version": "p0-limit-down-rescue-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "main_board_only": True,
            "first_touch_lookback_trading_days": LOOKBACK_TRADING_DAYS,
            "minimum_close_above_limit_down": RESCUE_FROM_LIMIT_FLOOR,
            "maximum_close_vs_reference": MAX_CLOSE_VS_REFERENCE,
            "signal_amount_floor_cny": SIGNAL_AMOUNT_FLOOR,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "position_notional_cny": 20_000,
            "daily_participation_rate": 0.01,
            "benchmark": "same-entry-date eligible A-share one-day median return",
        },
        "data": {
            "rescue_events": events.height,
            "event_symbols": events.get_column("symbol").n_unique()
            if events.height
            else 0,
            "event_days": events.get_column("ann_date").n_unique()
            if events.height
            else 0,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "result": summary,
        "decision": {
            "passed": summary["promotion_passed"],
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_account_contract_before_independent_validation"
                if summary["promotion_passed"]
                else "terminate_limit_down_rescue_and_move_to_next_mechanism"
            ),
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
        default=Path(
            "/app/data/research/p0_limit_down_rescue_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
