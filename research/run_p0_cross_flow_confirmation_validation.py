"""Run the frozen 2021-2023 cross-source capital-flow confirmation test."""

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
    DAILY_PARTICIPATION,
    POSITION_NOTIONAL,
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_large_order_flow_development import (  # noqa: E402
    FLOW_RATIO,
    MAX_ABS_EVENT_RETURN,
    MAX_EXIT_DELAY,
    MIN_ABS_FLOW,
    MIN_DAILY_AMOUNT,
    build_event_day_panel,
)
from research.run_p0_margin_acceleration_development import (  # noqa: E402
    MIN_BALANCE_GROWTH,
    MIN_BUY_INTENSITY,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

VALIDATION_START = date(2021, 1, 1)
VALIDATION_END = date(2023, 12, 31)
PANEL_START = date(2020, 10, 1)
PANEL_END = date(2024, 3, 31)
HOLD_TRADING_DAYS = 5
COOLDOWN_DAYS = 20
CATEGORY = "cross_flow_confirmation"


def _load_year_partitions(
    data_dir: Path,
    dataset: str,
    start_year: int,
    end_year: int,
) -> pl.DataFrame:
    paths: list[Path] = []
    for path in (data_dir / "event_data" / dataset).glob("year=*/part.parquet"):
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError:
            continue
        if start_year <= year <= end_year:
            paths.append(path)
    expected = end_year - start_year + 1
    if len(paths) != expected:
        found = sorted(path.parent.name for path in paths)
        raise ValueError(
            f"{dataset} requires {start_year}-{end_year} yearly partitions; found {found}"
        )
    return pl.read_parquet(sorted(paths), hive_partitioning=False)


def prepare_margin(data_dir: Path) -> pl.DataFrame:
    frame = _load_year_partitions(data_dir, "margin_detail", 2020, 2023).filter(
        pl.col("trade_date").is_between(date(2020, 10, 1), VALIDATION_END, closed="both")
    )
    calendar = (
        frame.select("trade_date").unique().sort("trade_date").with_row_index("margin_trade_index")
    )
    return (
        frame.join(calendar, on="trade_date", how="left")
        .sort(["symbol", "trade_date"])
        .with_columns(
            pl.col("rzye").shift(1).over("symbol").alias("previous_rzye"),
            pl.col("margin_trade_index")
            .shift(1)
            .over("symbol")
            .alias("previous_margin_trade_index"),
        )
        .with_columns(
            (pl.col("rzye") / pl.col("previous_rzye") - 1.0).alias("margin_balance_change"),
            (pl.col("margin_trade_index") == pl.col("previous_margin_trade_index") + 1).alias(
                "margin_dates_adjacent"
            ),
        )
        .filter(pl.col("trade_date").is_between(VALIDATION_START, VALIDATION_END, closed="both"))
        .sort(["trade_date", "symbol"])
    )


def prepare_moneyflow(data_dir: Path) -> pl.DataFrame:
    return (
        _load_year_partitions(data_dir, "moneyflow", 2021, 2023)
        .filter(pl.col("trade_date").is_between(VALIDATION_START, VALIDATION_END, closed="both"))
        .sort(["trade_date", "symbol"])
    )


def build_cross_flow_events(
    margin: pl.DataFrame,
    moneyflow: pl.DataFrame,
    event_day_panel: pl.DataFrame,
) -> tuple[pl.DataFrame, int, int]:
    day = event_day_panel.filter(
        pl.col("trade_date").is_between(VALIDATION_START, VALIDATION_END, closed="both")
    )
    margin_candidates = (
        margin.join(day, on=["symbol", "trade_date"], how="left")
        .with_columns(
            (pl.col("rzmre") / pl.col("event_daily_amount")).alias("margin_buy_intensity")
        )
        .filter(
            (pl.col("previous_rzye") > 0)
            & pl.col("margin_dates_adjacent").fill_null(False)
            & (pl.col("event_daily_amount") >= MIN_DAILY_AMOUNT)
            & (pl.col("margin_balance_change") >= MIN_BALANCE_GROWTH)
            & (pl.col("rzmre") >= MIN_ABS_FLOW)
            & (pl.col("margin_buy_intensity") >= MIN_BUY_INTENSITY)
            & pl.col("event_return").is_between(-MAX_ABS_EVENT_RETURN, 0.0, closed="both")
        )
        .select(
            "symbol",
            "trade_date",
            "margin_balance_change",
            "rzmre",
            "margin_buy_intensity",
            "event_daily_amount",
            "event_return",
        )
    )
    large_net = (
        pl.col("buy_lg_cny").fill_null(0)
        + pl.col("buy_elg_cny").fill_null(0)
        - pl.col("sell_lg_cny").fill_null(0)
        - pl.col("sell_elg_cny").fill_null(0)
    )
    flow_candidates = (
        moneyflow.join(day, on=["symbol", "trade_date"], how="left")
        .with_columns(large_net.alias("large_net_flow_cny"))
        .with_columns(
            (pl.col("large_net_flow_cny") / pl.col("event_daily_amount")).alias(
                "large_net_flow_ratio"
            )
        )
        .filter(
            (pl.col("event_daily_amount") >= MIN_DAILY_AMOUNT)
            & (pl.col("large_net_flow_cny") >= MIN_ABS_FLOW)
            & (pl.col("large_net_flow_ratio") >= FLOW_RATIO)
            & pl.col("event_return").is_between(-MAX_ABS_EVENT_RETURN, 0.0, closed="both")
        )
        .select(
            "symbol",
            "trade_date",
            "large_net_flow_cny",
            "large_net_flow_ratio",
        )
    )
    work = (
        margin_candidates.join(flow_candidates, on=["symbol", "trade_date"], how="inner")
        .rename({"trade_date": "ann_date"})
        .with_columns(pl.lit(CATEGORY).alias("category"))
        .sort(["symbol", "ann_date"])
    )
    last_kept: dict[str, date] = {}
    keep: list[bool] = []
    for row in work.iter_rows(named=True):
        symbol = row["symbol"]
        event_date = row["ann_date"]
        previous = last_kept.get(symbol)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[symbol] = event_date
    events = work.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(["ann_date", "symbol"])
    return events, margin_candidates.height, flow_candidates.height


def validation_passed(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["tradable_events"] >= 300
        and metrics["announcement_days"] >= 150
        and metrics["tradable_rate"] >= 0.90
        and metrics["benchmark_coverage"] >= 0.99
        and metrics["entry_capacity_feasible_rate"] >= 0.95
        and metrics["unresolved_exits"] == 0
        and (metrics["mean_net_return"] or -math.inf) >= 0.01
        and (metrics["mean_excess_return"] or -math.inf) >= 0.0075
        and (metrics["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and metrics["positive_excess_years"] >= 2
        and (metrics["max_year_positive_excess_share"] or math.inf) <= 0.60
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_margin = prepare_margin(data_dir)
    raw_flow = prepare_moneyflow(data_dir)
    raw_panel = load_panel(data_dir, PANEL_START, PANEL_END)
    events, margin_candidates, flow_candidates = build_cross_flow_events(
        raw_margin, raw_flow, build_event_day_panel(raw_panel)
    )
    panel = prepare_panel(raw_panel)
    trades = build_trades(
        events,
        panel,
        holding_trading_days=HOLD_TRADING_DAYS,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    metrics = summarize_category(
        trades,
        CATEGORY,
        positive_categories=(CATEGORY,),
        min_tradable_events=300,
        min_announcement_days=150,
    )
    metrics["promotion_passed"] = validation_passed(metrics)
    passed = metrics["promotion_passed"]
    payload = {
        "schema_version": "p0-cross-flow-confirmation-validation-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": VALIDATION_START,
            "end": VALIDATION_END,
            "joint_development_returns_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "max_exit_delay": MAX_EXIT_DELAY,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "minimum_margin_balance_growth": MIN_BALANCE_GROWTH,
            "minimum_margin_buy_intensity": MIN_BUY_INTENSITY,
            "minimum_large_flow_ratio": FLOW_RATIO,
            "minimum_daily_amount_cny": MIN_DAILY_AMOUNT,
            "minimum_financing_buy_cny": MIN_ABS_FLOW,
            "minimum_large_net_flow_cny": MIN_ABS_FLOW,
            "maximum_absolute_event_return": MAX_ABS_EVENT_RETURN,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "benchmark": "same-entry-date eligible A-share 5-day median return",
        },
        "data": {
            "raw_margin_rows": raw_margin.height,
            "raw_moneyflow_rows": raw_flow.height,
            "margin_candidates": margin_candidates,
            "flow_candidates": flow_candidates,
            "joint_events_after_cooldown": events.height,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
            "benchmark_entry_dates": benchmark.height,
        },
        "metrics": metrics,
        "decision": {
            "validation_passed": passed,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_stress_and_account_simulation_before_reading_2024_plus"
                if passed
                else "terminate_cross_flow_confirmation_mechanism"
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
        default=Path("/app/data/research/p0_cross_flow_confirmation_validation.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
