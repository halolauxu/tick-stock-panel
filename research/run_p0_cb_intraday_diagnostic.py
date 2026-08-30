"""Run a frozen one-month convertible-bond intraday structure diagnostic."""
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
POSITIONS = 3
CAPITAL_TIERS = (200_000, 300_000, 500_000, 1_000_000)
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


def build_causal_universe(basic: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Build each day's universe exclusively from the previous trading day."""
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
            basic.select("symbol", "cb_type", "list_date"),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("cb_type") == "CB")
            & (pl.col("list_date") <= pl.col("date") - pl.duration(days=30))
            & pl.col("previous_close").is_between(90.0, 150.0, closed="both")
            & (pl.col("previous_amount_cny") >= 30_000_000.0)
            & pl.col("previous_cb_over_rate").is_between(
                -10.0, 80.0, closed="both"
            )
        )
        .select(
            "symbol",
            "date",
            "previous_date",
            "previous_close",
            "previous_amount_cny",
            "previous_cb_over_rate",
        )
        .sort(["date", "symbol"])
    )


def prepare_minutes(minute: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """Restrict minute bars to ordinary-CB sessions and compute causal legs."""
    frame = (
        minute.with_columns(
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
        .join(universe, on=["symbol", "date"], how="inner")
        .sort(["date", "symbol", "session", "datetime"])
    )
    groups = ["date", "symbol", "session"]
    return frame.with_columns(
        (pl.col("close") / pl.col("close").shift(5).over(groups) - 1.0).alias(
            "past_5m"
        ),
        (pl.col("close") / pl.col("close").shift(15).over(groups) - 1.0).alias(
            "past_15m"
        ),
        pl.col("open").shift(-1).over(groups).alias("entry_open"),
        pl.col("amount_cny").shift(-1).over(groups).alias("entry_amount_cny"),
        pl.col("close").shift(-15).over(groups).alias("exit_close"),
        pl.col("amount_cny").shift(-15).over(groups).alias("exit_amount_cny"),
    )


def _rank_arms(frame: pl.DataFrame, signal_column: str) -> pl.DataFrame:
    ranked = frame.with_columns(
        pl.col(signal_column).rank("average").over("datetime").alias("rank"),
        pl.len().over("datetime").alias("universe_size"),
    ).with_columns((pl.col("rank") / pl.col("universe_size")).alias("percentile"))
    return pl.concat(
        [
            ranked.filter(pl.col("percentile") <= 0.2).with_columns(
                pl.lit("bottom20").alias("arm")
            ),
            ranked.filter(pl.col("percentile") >= 0.8).with_columns(
                pl.lit("top20").alias("arm")
            ),
        ]
    )


def build_observations(minutes: pl.DataFrame) -> pl.DataFrame:
    tradable = minutes.filter(
        pl.col("entry_open").is_not_null()
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
    intraday = []
    for lookback in (5, 15):
        signal_column = f"past_{lookback}m"
        ranked = _rank_arms(
            tradable.filter(
                pl.col("clock").is_in(SIGNAL_TIMES)
                & pl.col(signal_column).is_not_null()
            ),
            signal_column,
        ).with_columns(pl.lit(f"past_{lookback}m").alias("diagnostic"))
        intraday.append(ranked)

    opening = _rank_arms(
        tradable.filter(pl.col("clock") == time(9, 30)).with_columns(
            (pl.col("open") / pl.col("previous_close") - 1.0).alias("open_gap")
        ),
        "open_gap",
    ).with_columns(pl.lit("open_gap").alias("diagnostic"))
    return (
        pl.concat([*intraday, opening], how="diagonal_relaxed")
        .select(
            "diagnostic",
            "arm",
            "date",
            "datetime",
            "symbol",
            "gross_return",
            "capacity_cny",
            "universe_size",
        )
        .sort(["diagnostic", "arm", "datetime", "symbol"])
    )


def _safe_mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def summarize(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"events": 0, "dates": 0}
    net = frame.with_columns((pl.col("gross_return") - COST_RATE).alias("net_return"))
    daily = net.group_by("date").agg(pl.col("net_return").mean()).sort("date")
    daily_values = daily.get_column("net_return").to_list()
    daily_mean = _safe_mean(daily_values)
    daily_std = statistics.stdev(daily_values) if len(daily_values) >= 2 else None
    if daily_mean is not None and daily_std not in (None, 0.0):
        daily_t = daily_mean / (daily_std / math.sqrt(len(daily_values)))
    elif daily_mean is not None and daily_mean > 0 and daily_std == 0.0:
        daily_t = math.inf
    else:
        daily_t = None
    capacities = {}
    for capital in CAPITAL_TIERS:
        required = capital / POSITIONS
        capacities[str(capital)] = net.filter(
            pl.col("capacity_cny") >= required
        ).height / net.height
    return {
        "events": net.height,
        "dates": daily.height,
        "mean_gross_return": net.get_column("gross_return").mean(),
        "median_gross_return": net.get_column("gross_return").median(),
        "mean_net_return": net.get_column("net_return").mean(),
        "median_net_return": net.get_column("net_return").median(),
        "event_win_rate_net": net.filter(pl.col("net_return") > 0).height
        / net.height,
        "positive_day_rate": daily.filter(pl.col("net_return") > 0).height
        / daily.height,
        "daily_mean_net_return": daily_mean,
        "daily_t_stat": daily_t,
        "capacity_feasible_rate": capacities,
    }


def evaluate(observations: pl.DataFrame) -> dict[str, Any]:
    dates = sorted(observations.get_column("date").unique().to_list())
    midpoint = len(dates) // 2
    discovery_dates = dates[:midpoint]
    confirmation_dates = dates[midpoint:]
    results = []
    for diagnostic in ("past_5m", "past_15m", "open_gap"):
        for arm in ("bottom20", "top20"):
            subset = observations.filter(
                (pl.col("diagnostic") == diagnostic) & (pl.col("arm") == arm)
            )
            discovery = summarize(subset.filter(pl.col("date").is_in(discovery_dates)))
            confirmation = summarize(
                subset.filter(pl.col("date").is_in(confirmation_dates))
            )
            passed = (
                discovery.get("mean_net_return", -math.inf) > 0
                and confirmation.get("mean_net_return", -math.inf) > 0
                and confirmation.get("dates", 0) >= 8
                and confirmation.get("events", 0) >= 100
                and confirmation.get("positive_day_rate", 0) > 0.5
                and (confirmation.get("daily_t_stat") or -math.inf) >= 1.0
                and confirmation.get("capacity_feasible_rate", {}).get(
                    "200000", 0
                )
                >= 0.9
            )
            results.append(
                {
                    "diagnostic": diagnostic,
                    "arm": arm,
                    "discovery": discovery,
                    "confirmation": confirmation,
                    "promotion_passed": passed,
                }
            )
    return {
        "discovery_dates": discovery_dates,
        "confirmation_dates": confirmation_dates,
        "arms": results,
        "promoted_arms": [
            f"{result['diagnostic']}:{result['arm']}"
            for result in results
            if result["promotion_passed"]
        ],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, start: date, end: date, output: Path) -> dict[str, Any]:
    root = data_dir / "convertible_bond"
    basic = pl.read_parquet(root / "basic" / "part.parquet")
    daily_paths = _partition_paths(root, "daily", start, end)
    minute_paths = _partition_paths(root, "minute", start, end)
    if not daily_paths or not minute_paths:
        raise ValueError("daily and minute convertible-bond partitions are required")
    daily = pl.read_parquet(daily_paths).sort(["date", "symbol"])
    minute = pl.read_parquet(minute_paths).sort(["datetime", "symbol"])
    universe = build_causal_universe(basic, daily)
    prepared = prepare_minutes(minute, universe)
    observations = build_observations(prepared)
    evaluation = evaluate(observations)
    payload = {
        "schema_version": "p0-cb-intraday-diagnostic-v1",
        "contract_frozen": "2026-08-30",
        "start": start,
        "end": end,
        "assumptions": {
            "round_trip_cost_bps": COST_RATE * 10_000,
            "participation_rate": PARTICIPATION_RATE,
            "positions": POSITIONS,
            "capital_tiers_cny": CAPITAL_TIERS,
            "ordinary_cb_only": True,
            "prior_day_amount_floor_cny": 30_000_000,
            "prior_day_price_range": [90.0, 150.0],
            "prior_day_conversion_premium_range_pct": [-10.0, 80.0],
            "minimum_listing_calendar_days": 30,
        },
        "data": {
            "daily_rows": daily.height,
            "minute_rows": minute.height,
            "universe_symbol_days": universe.height,
            "universe_symbols": universe.get_column("symbol").n_unique(),
            "prepared_minute_rows": prepared.height,
            "observation_rows": observations.height,
        },
        "evaluation": evaluation,
        "decision": {
            "passed": bool(evaluation["promoted_arms"]),
            "promoted_arms": evaluation["promoted_arms"],
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_account_strategy_and_collect_one_year"
                if evaluation["promoted_arms"]
                else "terminate_intraday_batch_and_move_to_next_mechanism"
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
