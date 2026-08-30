"""Run the frozen convertible-bond versus underlying-stock lead-lag diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable
from datetime import date, datetime, time
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

COST_RATE = 0.0014
PARTICIPATION_RATE = 0.01
POSITIONS = 10
CAPITAL_TIERS = (200_000, 300_000, 500_000, 1_000_000)
STOCK_MOVE_FLOOR = 0.012
CB_MOVE_CEILING = 0.0025
LAG_FLOOR = 0.006
SIGNAL_TIMES = (
    time(9, 45),
    time(10, 0),
    time(10, 15),
    time(10, 30),
    time(10, 45),
    time(11, 0),
    time(13, 15),
    time(13, 30),
    time(13, 45),
    time(14, 0),
    time(14, 15),
    time(14, 30),
)


def _partition_paths(root: Path, dataset: str, start: date, end: date) -> list[Path]:
    paths = []
    for path in (root / dataset).glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= value <= end:
            paths.append(path)
    return sorted(paths)


def _with_session(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("datetime").dt.date().alias("date"),
            pl.col("datetime").dt.time().alias("clock"),
        )
        .with_columns(
            pl.when(pl.col("clock").is_between(time(9, 30), time(11, 30)))
            .then(pl.lit("AM"))
            .when(pl.col("clock").is_between(time(13, 0), time(15, 0)))
            .then(pl.lit("PM"))
            .otherwise(None)
            .alias("session")
        )
        .filter(pl.col("session").is_not_null())
    )


def build_causal_universe(basic: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Build today's eligible mapped CBs using only the previous trading day."""
    dates = sorted(daily.get_column("date").unique().to_list())
    previous = {current: prior for prior, current in pairwise(dates)}
    current = (
        daily.select("symbol", "date")
        .unique()
        .with_columns(
            pl.col("date")
            .replace_strict(previous, default=None, return_dtype=pl.Date)
            .alias("previous_date")
        )
        .filter(pl.col("previous_date").is_not_null())
    )
    prior = daily.select(
        "symbol",
        pl.col("date").alias("previous_date"),
        pl.col("close").alias("previous_close"),
        pl.col("amount_cny").alias("previous_amount_cny"),
        pl.col("cb_over_rate").alias("previous_cb_over_rate"),
    )
    return (
        current.join(prior, on=["symbol", "previous_date"], how="inner")
        .join(
            basic.select(
                "symbol", "stock_symbol", "cb_type", "list_date"
            ),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("cb_type") == "CB")
            & pl.col("stock_symbol").is_not_null()
            & (pl.col("list_date") <= pl.col("date") - pl.duration(days=30))
            & pl.col("previous_close").is_between(95.0, 140.0, closed="both")
            & (pl.col("previous_amount_cny") >= 100_000_000.0)
            & pl.col("previous_cb_over_rate").is_between(
                0.0, 50.0, closed="both"
            )
        )
        .select(
            "symbol",
            "stock_symbol",
            "date",
            "previous_date",
            "previous_close",
            "previous_amount_cny",
            "previous_cb_over_rate",
        )
        .sort(["date", "symbol"])
    )


def prepare_cb_minutes(minute: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    frame = (
        _with_session(minute)
        .join(universe, on=["symbol", "date"], how="inner")
        .sort(["date", "symbol", "session", "datetime"])
    )
    groups = ["date", "symbol", "session"]
    return frame.with_columns(
        (pl.col("close") / pl.col("close").shift(5).over(groups) - 1.0).alias(
            "cb_past_5m"
        ),
        pl.col("open").shift(-1).over(groups).alias("entry_open"),
        pl.col("amount_cny").shift(-1).over(groups).alias("entry_amount_cny"),
        pl.col("close").shift(-15).over(groups).alias("exit_close"),
        pl.col("amount_cny").shift(-15).over(groups).alias("exit_amount_cny"),
    )


def prepare_stock_minutes(minute: pl.DataFrame) -> pl.DataFrame:
    frame = _with_session(minute).sort(
        ["date", "symbol", "session", "datetime"]
    )
    groups = ["date", "symbol", "session"]
    return frame.with_columns(
        (pl.col("close") / pl.col("close").shift(5).over(groups) - 1.0).alias(
            "stock_past_5m"
        )
    ).select(
        pl.col("symbol").alias("stock_symbol"),
        "datetime",
        "stock_past_5m",
    )


def build_observations(
    cb_minutes: pl.DataFrame, stock_minutes: pl.DataFrame
) -> pl.DataFrame:
    executable = cb_minutes.filter(
        pl.col("clock").is_in(SIGNAL_TIMES)
        & pl.col("cb_past_5m").is_not_null()
        & pl.col("entry_open").is_not_null()
        & pl.col("exit_close").is_not_null()
        & (pl.col("entry_open") > 0)
        & (pl.col("exit_close") > 0)
        & (pl.col("entry_amount_cny") > 0)
        & (pl.col("exit_amount_cny") > 0)
    ).with_columns(
        (pl.col("exit_close") / pl.col("entry_open") - 1.0).alias(
            "gross_return"
        ),
        (
            pl.min_horizontal("entry_amount_cny", "exit_amount_cny")
            * PARTICIPATION_RATE
        ).alias("capacity_cny"),
    )
    benchmark = executable.group_by("datetime").agg(
        pl.col("gross_return").median().alias("benchmark_gross_return")
    )
    candidates = (
        executable.join(
            stock_minutes,
            on=["stock_symbol", "datetime"],
            how="inner",
        )
        .filter(pl.col("stock_past_5m").is_not_null())
        .with_columns(
            (
                pl.col("stock_past_5m")
                / (1.0 + pl.col("previous_cb_over_rate") / 100.0)
            ).alias("expected_pass_through")
        )
        .with_columns(
            (pl.col("expected_pass_through") - pl.col("cb_past_5m")).alias(
                "lag"
            )
        )
        .filter(
            (pl.col("stock_past_5m") >= STOCK_MOVE_FLOOR)
            & (pl.col("cb_past_5m") <= CB_MOVE_CEILING)
            & (pl.col("lag") >= LAG_FLOOR)
        )
        .sort(["date", "symbol", "datetime"])
        .unique(subset=["date", "symbol"], keep="first", maintain_order=True)
        .sort(
            ["datetime", "lag", "symbol"],
            descending=[False, True, False],
        )
        .with_columns(
            pl.col("lag")
            .rank(method="ordinal", descending=True)
            .over("datetime")
            .alias("slot")
        )
        .filter(pl.col("slot") <= POSITIONS)
        .join(benchmark, on="datetime", how="inner")
        .with_columns(
            (pl.col("gross_return") - COST_RATE).alias("net_return"),
            (
                pl.col("gross_return") - pl.col("benchmark_gross_return")
            ).alias("excess_return"),
        )
    )
    return candidates.select(
        "date",
        "datetime",
        "symbol",
        "stock_symbol",
        "stock_past_5m",
        "cb_past_5m",
        "expected_pass_through",
        "lag",
        "gross_return",
        "net_return",
        "benchmark_gross_return",
        "excess_return",
        "capacity_cny",
        "slot",
    ).sort(["datetime", "slot", "symbol"])


def _safe_mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _t_stat(values: list[float]) -> float | None:
    mean = _safe_mean(values)
    if mean is None or len(values) < 2:
        return None
    std = statistics.stdev(values)
    if std == 0:
        return math.inf if mean > 0 else None
    return mean / (std / math.sqrt(len(values)))


def summarize(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "events": 0,
            "event_days": 0,
            "capacity_feasible_rate": {str(v): 0.0 for v in CAPITAL_TIERS},
        }
    daily = frame.group_by("date").agg(
        pl.col("net_return").mean(),
        pl.col("excess_return").mean(),
    ).sort("date")
    capacities = {}
    for capital in CAPITAL_TIERS:
        required = capital / POSITIONS
        capacities[str(capital)] = frame.filter(
            pl.col("capacity_cny") >= required
        ).height / frame.height
    return {
        "events": frame.height,
        "event_days": daily.height,
        "mean_gross_return": frame.get_column("gross_return").mean(),
        "mean_net_return": frame.get_column("net_return").mean(),
        "median_net_return": frame.get_column("net_return").median(),
        "event_win_rate_net": frame.filter(pl.col("net_return") > 0).height
        / frame.height,
        "mean_benchmark_gross_return": frame.get_column(
            "benchmark_gross_return"
        ).mean(),
        "mean_excess_return": frame.get_column("excess_return").mean(),
        "positive_day_rate": daily.filter(pl.col("net_return") > 0).height
        / daily.height,
        "daily_excess_t_stat": _t_stat(
            daily.get_column("excess_return").to_list()
        ),
        "capacity_feasible_rate": capacities,
    }


def _passes(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "at_least_8_event_days": summary.get("event_days", 0) >= 8,
        "at_least_100_events": summary.get("events", 0) >= 100,
        "mean_net_at_least_20bps": summary.get("mean_net_return", -math.inf)
        >= 0.002,
        "mean_excess_at_least_15bps": summary.get(
            "mean_excess_return", -math.inf
        )
        >= 0.0015,
        "positive_day_rate_above_60pct": summary.get("positive_day_rate", 0)
        > 0.6,
        "daily_excess_t_at_least_1_5": (
            summary.get("daily_excess_t_stat") or -math.inf
        )
        >= 1.5,
        "capacity_200k_at_least_90pct": summary.get(
            "capacity_feasible_rate", {}
        ).get("200000", 0)
        >= 0.9,
    }
    return all(checks.values()), checks


def evaluate(
    observations: pl.DataFrame, market_dates: list[date]
) -> dict[str, Any]:
    midpoint = len(market_dates) // 2
    discovery_dates = market_dates[:midpoint]
    confirmation_dates = market_dates[midpoint:]
    discovery = summarize(
        observations.filter(pl.col("date").is_in(discovery_dates))
    )
    confirmation = summarize(
        observations.filter(pl.col("date").is_in(confirmation_dates))
    )
    discovery_passed, discovery_checks = _passes(discovery)
    confirmation_passed, confirmation_checks = _passes(confirmation)
    return {
        "discovery_dates": discovery_dates,
        "confirmation_dates": confirmation_dates,
        "discovery": discovery,
        "confirmation": confirmation,
        "checks": {
            "discovery": discovery_checks,
            "confirmation": confirmation_checks,
        },
        "promotion_passed": discovery_passed and confirmation_passed,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, start: date, end: date, output: Path) -> dict[str, Any]:
    cb_root = data_dir / "convertible_bond"
    basic = pl.read_parquet(cb_root / "basic" / "part.parquet")
    daily_paths = _partition_paths(cb_root, "daily", start, end)
    cb_minute_paths = _partition_paths(cb_root, "minute", start, end)
    stock_minute_paths = _partition_paths(data_dir, "kline_minute", start, end)
    if not daily_paths or not cb_minute_paths or not stock_minute_paths:
        raise ValueError("CB daily, CB minute, and stock minute partitions are required")
    daily = pl.read_parquet(daily_paths).sort(["date", "symbol"])
    universe = build_causal_universe(basic, daily)
    cb_minute = pl.read_parquet(cb_minute_paths).sort(["datetime", "symbol"])
    stock_symbols = universe.get_column("stock_symbol").unique().to_list()
    stock_minute = (
        pl.scan_parquet(stock_minute_paths)
        .filter(pl.col("symbol").is_in(stock_symbols))
        .collect()
        .sort(["datetime", "symbol"])
    )
    prepared_cb = prepare_cb_minutes(cb_minute, universe)
    prepared_stock = prepare_stock_minutes(stock_minute)
    observations = build_observations(prepared_cb, prepared_stock)
    market_dates = sorted(cb_minute.get_column("datetime").dt.date().unique())
    evaluation = evaluate(observations, market_dates)
    payload = {
        "schema_version": "p0-cb-stock-lead-lag-discovery-v1",
        "contract_frozen": "2026-08-30",
        "start": start,
        "end": end,
        "assumptions": {
            "stock_move_floor": STOCK_MOVE_FLOOR,
            "cb_move_ceiling": CB_MOVE_CEILING,
            "lag_floor": LAG_FLOOR,
            "round_trip_cost_bps": COST_RATE * 10_000,
            "participation_rate": PARTICIPATION_RATE,
            "positions": POSITIONS,
            "capital_tiers_cny": CAPITAL_TIERS,
            "prior_day_amount_floor_cny": 100_000_000,
            "prior_day_price_range": [95.0, 140.0],
            "prior_day_conversion_premium_range_pct": [0.0, 50.0],
            "minimum_listing_calendar_days": 30,
        },
        "data": {
            "market_dates": len(market_dates),
            "daily_rows": daily.height,
            "cb_minute_rows": cb_minute.height,
            "stock_minute_rows_for_mapped_underlyings": stock_minute.height,
            "universe_symbol_days": universe.height,
            "universe_cb_symbols": universe.get_column("symbol").n_unique(),
            "mapped_stock_symbols": universe.get_column(
                "stock_symbol"
            ).n_unique(),
            "prepared_cb_minute_rows": prepared_cb.height,
            "prepared_stock_minute_rows": prepared_stock.height,
            "signal_rows": observations.height,
        },
        "evaluation": evaluation,
        "decision": {
            "passed": evaluation["promotion_passed"],
            "counts_toward_50pct_goal": False,
            "next_step": (
                "collect_one_year_and_freeze_account_strategy"
                if evaluation["promotion_passed"]
                else "terminate_lead_lag_hypothesis_and_move_to_next_mechanism"
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
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.start, args.end, args.output)


if __name__ == "__main__":
    main()
