"""Run the frozen development-only Dragon-Tiger retail-seat consensus study."""
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
from research.run_p0_top_inst_flow_development import (  # noqa: E402
    deduplicate_seat_amounts,
    load_top_inst,
)

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 2, 28)
HOLD_TRADING_DAYS = 1
MAX_EXIT_DELAY = 20
COOLDOWN_DAYS = 20
MIN_POSITIVE_SEATS = 8
MIN_NET_BUY_AMOUNT_SHARE = 0.05
MIN_POSITIVE_BUY_RATE_SUM = 10.0
CATEGORY = "retail_seat_consensus"


def aggregate_retail_consensus(
    details: pl.DataFrame, panel: pl.DataFrame
) -> pl.DataFrame:
    unique = deduplicate_seat_amounts(details)
    special = pl.col("seat_name").str.contains(
        r"(?:机构专用|沪股通专用|深股通专用)"
    )
    per_seat = (
        unique.filter(~special)
        .group_by("trade_date", "symbol", "seat_name")
        .agg(
            pl.col("buy").max().alias("buy"),
            pl.col("sell").max().alias("sell"),
            pl.col("net_buy").max().alias("net_buy"),
            pl.col("buy_rate").max().alias("buy_rate"),
        )
    )
    events = per_seat.group_by("trade_date", "symbol").agg(
        (pl.col("net_buy") > 0).sum().alias("positive_seats"),
        pl.col("net_buy").sum().alias("retail_net_buy"),
        pl.when(pl.col("net_buy") > 0)
        .then(pl.col("buy_rate"))
        .otherwise(0.0)
        .sum()
        .alias("positive_buy_rate_sum"),
    )
    signal = panel.select(
        "symbol",
        pl.col("date").alias("trade_date"),
        pl.col("amount").alias("signal_amount"),
        pl.col("raw_close").alias("signal_raw_close"),
        pl.col("limit_up_price").alias("signal_limit_up"),
        pl.col("excluded_name").alias("signal_excluded_name"),
    )
    candidates = (
        events.join(signal, on=["trade_date", "symbol"], how="inner")
        .filter(
            (pl.col("positive_seats") >= MIN_POSITIVE_SEATS)
            & (pl.col("retail_net_buy") > 0)
            & (
                pl.col("retail_net_buy")
                >= pl.col("signal_amount") * MIN_NET_BUY_AMOUNT_SHARE
            )
            & (pl.col("positive_buy_rate_sum") >= MIN_POSITIVE_BUY_RATE_SUM)
            & ~pl.col("signal_excluded_name").fill_null(True)
            & (pl.col("signal_amount") >= 20_000_000.0)
            & (
                pl.col("signal_raw_close")
                < pl.col("signal_limit_up") - 0.005
            ).fill_null(False)
        )
        .rename({"trade_date": "ann_date"})
        .with_columns(pl.lit(CATEGORY).alias("category"))
        .sort(["symbol", "ann_date"])
    )
    last_kept: dict[str, date] = {}
    keep = []
    for row in candidates.iter_rows(named=True):
        symbol = row["symbol"]
        event_date = row["ann_date"]
        previous = last_kept.get(symbol)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[symbol] = event_date
    return candidates.filter(
        pl.Series("_keep", keep, dtype=pl.Boolean)
    ).sort(["ann_date", "symbol"])


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    details = load_top_inst(data_dir)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    events = aggregate_retail_consensus(details, panel)
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
        "schema_version": "p0-retail-seat-consensus-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "minimum_positive_retail_seats": MIN_POSITIVE_SEATS,
            "minimum_retail_net_buy_share_of_daily_amount": (
                MIN_NET_BUY_AMOUNT_SHARE
            ),
            "minimum_positive_buy_rate_sum_pct": MIN_POSITIVE_BUY_RATE_SUM,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "position_notional_cny": 20_000,
            "daily_participation_rate": 0.01,
            "benchmark": "same-entry-date eligible A-share one-day median return",
        },
        "data": {
            "raw_seat_rows": details.height,
            "consensus_events": events.height,
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
                else "terminate_retail_seat_consensus_and_move_to_next_mechanism"
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
            "/app/data/research/p0_retail_seat_consensus_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
