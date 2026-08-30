"""Run the frozen pre-close first-near-limit overnight discovery study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, time
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.run_p0_forecast_drift_development import (  # noqa: E402
    COMMISSION_PCT,
    DAILY_PARTICIPATION,
    SLIPPAGE_PCT,
    STAMP_TAX_CURRENT,
    load_panel,
    prepare_panel,
)

CONTEXT_START = date(2025, 8, 1)
START = date(2025, 8, 27)
END = date(2026, 8, 28)
MIN_PREVIOUS_AMOUNT = 100_000_000.0
MIN_LATE_AMOUNT = 20_000_000.0
MIN_LATE_RETURN = 0.02
NEAR_LIMIT_LOW = 0.98
NEAR_LIMIT_HIGH = 0.995
POSITIONS = 10
CAPITAL_LEVELS = (200_000, 300_000, 500_000, 1_000_000)
MAIN_BOARD_PATTERN = r"^(?:00\d{4}\.SZ|60\d{4}\.SH)$"


def _partition_paths(root: Path, start: date, end: date) -> list[Path]:
    paths = []
    for path in root.glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= day <= end:
            paths.append(path)
    return sorted(paths)


def load_minute_features(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    paths = _partition_paths(data_dir / "kline_minute", START, END)
    if not paths:
        raise ValueError("minute partitions are required")
    scan = pl.scan_parquet(paths, hive_partitioning=False)
    session_to_signal = (
        scan.filter(
            (pl.col("datetime").dt.time() >= time(9, 30))
            & (pl.col("datetime").dt.time() <= time(14, 45))
        )
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .group_by("symbol", "date")
        .agg(pl.col("high").max().alias("pre_signal_high"))
        .collect(engine="streaming")
    )
    late = (
        scan.filter(
            (pl.col("datetime").dt.hour() == 14)
            & pl.col("datetime").dt.minute().is_between(31, 45)
        )
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .sort(["symbol", "date", "datetime"])
        .group_by("symbol", "date", maintain_order=True)
        .agg(
            pl.col("open").first().alias("late_open"),
            pl.col("close").last().alias("signal_close"),
            pl.col("amount").sum().alias("late_amount"),
            pl.len().alias("late_bars"),
        )
        .with_columns(
            (pl.col("signal_close") / pl.col("late_open") - 1.0).alias(
                "late_return"
            )
        )
        .collect(engine="streaming")
        .join(session_to_signal, on=["symbol", "date"], how="inner")
    )
    entry = (
        scan.filter(
            (pl.col("datetime").dt.hour() == 14)
            & (pl.col("datetime").dt.minute() == 46)
        )
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .select(
            "symbol",
            "date",
            pl.col("open").alias("entry_open"),
            pl.col("amount").alias("entry_amount"),
        )
        .unique(["symbol", "date"], keep="last")
        .collect(engine="streaming")
    )
    exit_frame = (
        scan.filter(
            (pl.col("datetime").dt.hour() == 9)
            & (pl.col("datetime").dt.minute() == 31)
        )
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .select(
            "symbol",
            "date",
            pl.col("open").alias("exit_open"),
            pl.col("amount").alias("exit_amount"),
        )
        .unique(["symbol", "date"], keep="last")
        .collect(engine="streaming")
    )
    return late, entry, exit_frame, {
        "partitions": len(paths),
        "first_date": START,
        "last_date": END,
        "late_symbol_days": late.height,
        "entry_symbol_days": entry.height,
        "exit_symbol_days": exit_frame.height,
    }


def build_daily_context(panel: pl.DataFrame) -> pl.DataFrame:
    prepared = prepare_panel(panel)
    factors = (
        panel.sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("raw_close")).alias("adjustment_factor"),
            pl.col("amount").shift(1).over("symbol").alias("previous_amount"),
            pl.col("raw_close")
            .shift(1)
            .over("symbol")
            .alias("previous_raw_close"),
        )
        .select(
            "symbol",
            "date",
            "adjustment_factor",
            "previous_amount",
            "previous_raw_close",
        )
    )
    return prepared.join(factors, on=["symbol", "date"], how="left")


def build_observations(
    late: pl.DataFrame,
    entry: pl.DataFrame,
    exit_frame: pl.DataFrame,
    context: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    context_columns = context.select(
        "symbol",
        "date",
        "trade_index",
        "excluded_name",
        "limit_up_price",
        "limit_down_price",
        "adjustment_factor",
        "previous_amount",
        "previous_raw_close",
    )
    base = (
        late.join(entry, on=["symbol", "date"], how="inner")
        .join(context_columns, on=["symbol", "date"], how="inner")
        .filter(
            (pl.col("late_bars") == 15)
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
            & ~pl.col("excluded_name").fill_null(True)
            & pl.col("previous_raw_close").is_between(3.0, 100.0, closed="both")
            & (pl.col("previous_amount") >= MIN_PREVIOUS_AMOUNT)
            & (pl.col("entry_open") > 0)
        )
    )
    exit_lookup = (
        exit_frame.join(context_columns, on=["symbol", "date"], how="inner")
        .select(
            "symbol",
            pl.col("trade_index").alias("exit_trade_index"),
            pl.col("date").alias("exit_date"),
            "exit_open",
            "exit_amount",
            pl.col("excluded_name").alias("exit_excluded_name"),
            pl.col("limit_down_price").alias("exit_limit_down_price"),
            pl.col("adjustment_factor").alias("exit_adjustment_factor"),
        )
    )
    executed = (
        base.with_columns((pl.col("trade_index") + 1).alias("exit_trade_index"))
        .join(exit_lookup, on=["symbol", "exit_trade_index"], how="left")
        .with_columns(
            (
                (pl.col("entry_open") < pl.col("limit_up_price") - 0.005)
                & (pl.col("entry_amount") > 0)
                & pl.col("exit_open").is_not_null()
                & (pl.col("exit_open") > pl.col("exit_limit_down_price") + 0.005)
                & (pl.col("exit_amount") > 0)
                & ~pl.col("exit_excluded_name").fill_null(True)
                & (pl.col("adjustment_factor") == pl.col("exit_adjustment_factor"))
            ).alias("tradable")
        )
        .with_columns(
            pl.min_horizontal("entry_amount", "exit_amount")
            .mul(DAILY_PARTICIPATION)
            .alias("capacity_cny"),
            (
                (
                    pl.col("exit_open")
                    * (1.0 - COMMISSION_PCT - SLIPPAGE_PCT - STAMP_TAX_CURRENT)
                )
                / (pl.col("entry_open") * (1.0 + COMMISSION_PCT + SLIPPAGE_PCT))
                - 1.0
            ).alias("_net_return"),
        )
        .with_columns(
            pl.when(pl.col("tradable"))
            .then(pl.col("_net_return"))
            .otherwise(None)
            .alias("net_return")
        )
        .drop("_net_return")
    )
    benchmark = (
        executed.filter(pl.col("tradable"))
        .group_by("date")
        .agg(pl.col("net_return").median().alias("market_median_return"))
    )
    signals = executed.filter(
        (pl.col("late_return") >= MIN_LATE_RETURN)
        & (pl.col("late_amount") >= MIN_LATE_AMOUNT)
        & (
            pl.col("signal_close").is_between(
                pl.col("limit_up_price") * NEAR_LIMIT_LOW,
                pl.col("limit_up_price") * NEAR_LIMIT_HIGH,
                closed="both",
            )
        )
        & (pl.col("pre_signal_high") < pl.col("limit_up_price") - 0.005)
    )
    return (
        signals.join(benchmark, on="date", how="left").with_columns(
            (pl.col("net_return") - pl.col("market_median_return")).alias(
                "excess_return"
            )
        ),
        base,
    )


def _cluster_t(frame: pl.DataFrame, column: str) -> float | None:
    daily = frame.group_by("date").agg(pl.col(column).mean().alias("value"))
    if daily.height < 2:
        return None
    mean = daily.get_column("value").mean()
    std = daily.get_column("value").std(ddof=1)
    return mean / (std / math.sqrt(daily.height)) if std and mean is not None else None


def summarize(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"events": 0, "days": 0}
    tradable = frame.filter(pl.col("tradable"))
    monthly = (
        tradable.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.col("excess_return").sum().alias("positive_contribution"))
        .sort("month")
    )
    positive = monthly.filter(pl.col("positive_contribution") > 0).get_column(
        "positive_contribution"
    ).to_list()
    capacities = {
        str(capital): (
            tradable.filter(pl.col("capacity_cny") >= capital / POSITIONS).height
            / tradable.height
            if tradable.height
            else 0.0
        )
        for capital in CAPITAL_LEVELS
    }
    return {
        "events": frame.height,
        "tradable_events": tradable.height,
        "days": tradable.get_column("date").n_unique() if tradable.height else 0,
        "tradable_rate": tradable.height / frame.height,
        "mean_net_return": tradable.get_column("net_return").mean()
        if tradable.height
        else None,
        "mean_excess_return": tradable.get_column("excess_return").mean()
        if tradable.height
        else None,
        "median_excess_return": tradable.get_column("excess_return").median()
        if tradable.height
        else None,
        "daily_excess_t": _cluster_t(tradable, "excess_return"),
        "positive_excess_months": len(positive),
        "max_month_positive_share": max(positive) / sum(positive)
        if positive
        else None,
        "capacity_feasible_rate": capacities,
        "monthly": monthly.to_dicts(),
    }


def evaluate(observations: pl.DataFrame, market_dates: list[date]) -> dict[str, Any]:
    midpoint = len(market_dates) // 2
    discovery_dates = market_dates[:midpoint]
    confirmation_dates = market_dates[midpoint:]
    discovery = summarize(
        observations.filter(pl.col("date").is_in(discovery_dates))
    )
    confirmation = summarize(
        observations.filter(pl.col("date").is_in(confirmation_dates))
    )
    passed = all(
        (
            (discovery.get("mean_net_return") or -math.inf) >= 0.005,
            (confirmation.get("mean_net_return") or -math.inf) >= 0.005,
            (discovery.get("mean_excess_return") or -math.inf) >= 0.003,
            (confirmation.get("mean_excess_return") or -math.inf) >= 0.003,
            confirmation.get("days", 0) >= 80,
            confirmation.get("events", 0) >= 500,
            (confirmation.get("daily_excess_t") or -math.inf) >= 2.0,
            confirmation.get("positive_excess_months", 0) >= 4,
            (confirmation.get("max_month_positive_share") or math.inf) <= 0.50,
            confirmation.get("capacity_feasible_rate", {}).get("200000", 0)
            >= 0.90,
            confirmation.get("tradable_rate", 0) >= 0.90,
        )
    )
    return {
        "discovery_dates": discovery_dates,
        "confirmation_dates": confirmation_dates,
        "discovery": discovery,
        "confirmation": confirmation,
        "promotion_passed": passed,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    context = build_daily_context(load_panel(data_dir, CONTEXT_START, END))
    late, entry, exit_frame, minute_meta = load_minute_features(data_dir)
    observations, eligible = build_observations(late, entry, exit_frame, context)
    market_dates = sorted(
        context.filter(pl.col("date").is_between(START, END, closed="both"))
        .get_column("date")
        .unique()
        .to_list()
    )
    evaluation = evaluate(observations, market_dates)
    payload = {
        "schema_version": "p0-preclose-near-limit-discovery-v1",
        "contract_frozen": "2026-08-30",
        "period": {"start": START, "end": END},
        "assumptions": {
            "signal_window": "14:31-14:45",
            "entry": "14:46 minute open",
            "exit": "next trading day 09:31 minute open",
            "minimum_late_return": MIN_LATE_RETURN,
            "minimum_late_amount_cny": MIN_LATE_AMOUNT,
            "near_limit_range": [NEAR_LIMIT_LOW, NEAR_LIMIT_HIGH],
            "positions": POSITIONS,
            "capital_levels_cny": CAPITAL_LEVELS,
            "minute_participation": DAILY_PARTICIPATION,
        },
        "data": {
            **minute_meta,
            "daily_context_rows": context.height,
            "eligible_symbol_days": eligible.height,
            "signal_rows": observations.height,
        },
        "evaluation": evaluation,
        "decision": {
            "passed": evaluation["promotion_passed"],
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_forward_account"
                if evaluation["promotion_passed"]
                else "terminate_preclose_near_limit_mechanism"
            ),
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
        default=Path(
            "/app/data/research/p0_preclose_near_limit_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
