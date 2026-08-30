"""Run the frozen discovery-only opening-auction imbalance study."""
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
    COMMISSION_PCT,
    DAILY_PARTICIPATION,
    POSITION_NOTIONAL,
    SLIPPAGE_PCT,
    historical_stamp_tax,
    load_panel,
    prepare_panel,
)

CONTEXT_START = date(2025, 7, 1)
DISCOVERY_START = date(2025, 8, 27)
DISCOVERY_END = date(2026, 2, 13)
EXECUTION_END = date(2026, 3, 31)
MAX_EXIT_DELAY = 20
COOLDOWN_DAYS = 20
MIN_AUCTION_AMOUNT = 3_000_000.0
MIN_ACTIVE_SHARE = 0.01
MAX_CONTROL_SHARE = 0.005
MIN_ABS_GAP = 0.01
MAX_ABS_GAP = 0.04

CANDIDATES = ("demand_continuation", "supply_absorption")
CONTROLS = ("positive_gap_low_activity_control", "negative_gap_low_activity_control")
CATEGORIES = (*CANDIDATES, *CONTROLS)


def _partition_paths(root: Path, start: date, end: date) -> list[Path]:
    paths = []
    for path in root.glob("date=*/part.parquet"):
        try:
            value = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= value <= end:
            paths.append(path)
    return sorted(paths)


def load_auction(data_dir: Path) -> pl.DataFrame:
    paths = _partition_paths(
        data_dir / "tushare_supplemental" / "auction",
        DISCOVERY_START,
        DISCOVERY_END,
    )
    if not paths:
        raise ValueError("opening-auction partitions are required")
    return (
        pl.read_parquet(paths, hive_partitioning=False)
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .filter(
            (pl.col("session") == "open")
            & pl.col("date").is_between(
                DISCOVERY_START, DISCOVERY_END, closed="both"
            )
        )
        .unique(subset=["symbol", "date", "session"], keep="last")
        .sort(["date", "symbol"])
    )


def load_0931(data_dir: Path) -> pl.DataFrame:
    paths = _partition_paths(
        data_dir / "kline_minute", DISCOVERY_START, EXECUTION_END
    )
    if not paths:
        raise ValueError("minute partitions are required")
    return (
        pl.scan_parquet(paths, hive_partitioning=False)
        .filter(
            (pl.col("datetime").dt.hour() == 9)
            & (pl.col("datetime").dt.minute() == 31)
        )
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .select("symbol", "date", "open", "high", "low", "close", "volume", "amount")
        .unique(subset=["symbol", "date"], keep="last")
        .collect(engine="streaming")
        .sort(["date", "symbol"])
    )


def build_daily_context(panel: pl.DataFrame) -> pl.DataFrame:
    prepared = prepare_panel(panel)
    factors = (
        panel.sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("raw_close")).alias("factor"),
            pl.col("close").shift(1).over("symbol").alias("previous_close"),
            pl.col("raw_close").shift(1).over("symbol").alias("previous_raw_close"),
            (pl.col("close").shift(1).over("symbol") / pl.col("raw_close").shift(1).over("symbol")).alias("previous_factor"),
            pl.col("amount").shift(1).over("symbol").alias("previous_amount"),
        )
        .with_columns(
            pl.when((pl.col("factor") - pl.col("previous_factor")).abs() > 1e-6)
            .then(pl.col("previous_close"))
            .otherwise(pl.col("previous_raw_close"))
            .alias("reference_close")
        )
        .select("symbol", "date", "reference_close", "previous_amount")
    )
    return prepared.join(factors, on=["symbol", "date"], how="left")


def categorize_events(
    auction: pl.DataFrame, daily_context: pl.DataFrame
) -> pl.DataFrame:
    work = (
        auction.join(daily_context, on=["symbol", "date"], how="left")
        .with_columns(
            (pl.col("close") / pl.col("reference_close") - 1.0).alias("gap_return"),
            (pl.col("close") / pl.col("open") - 1.0).alias("auction_internal_return"),
            (pl.col("amount") / pl.col("previous_amount")).alias("auction_amount_share"),
        )
        .with_columns(
            (
                pl.col("reference_close").is_between(3.0, 300.0, closed="both")
                & (pl.col("previous_amount") >= 20_000_000.0)
                & ~pl.col("excluded_name").fill_null(True)
            ).alias("universe_eligible")
        )
        .with_columns(
            pl.when(
                pl.col("gap_return").is_between(
                    MIN_ABS_GAP, MAX_ABS_GAP, closed="both"
                )
                & (pl.col("auction_internal_return") >= 0)
                & (pl.col("amount") >= MIN_AUCTION_AMOUNT)
                & (pl.col("auction_amount_share") >= MIN_ACTIVE_SHARE)
            )
            .then(pl.lit("demand_continuation"))
            .when(
                pl.col("gap_return").is_between(
                    -MAX_ABS_GAP, -MIN_ABS_GAP, closed="both"
                )
                & (pl.col("auction_internal_return") >= 0)
                & (pl.col("amount") >= MIN_AUCTION_AMOUNT)
                & (pl.col("auction_amount_share") >= MIN_ACTIVE_SHARE)
            )
            .then(pl.lit("supply_absorption"))
            .when(
                pl.col("gap_return").is_between(
                    MIN_ABS_GAP, MAX_ABS_GAP, closed="both"
                )
                & (pl.col("auction_amount_share") < MAX_CONTROL_SHARE)
            )
            .then(pl.lit("positive_gap_low_activity_control"))
            .when(
                pl.col("gap_return").is_between(
                    -MAX_ABS_GAP, -MIN_ABS_GAP, closed="both"
                )
                & (pl.col("auction_amount_share") < MAX_CONTROL_SHARE)
            )
            .then(pl.lit("negative_gap_low_activity_control"))
            .otherwise(None)
            .alias("category")
        )
        .filter(pl.col("category").is_not_null())
        .rename({"date": "ann_date"})
        .sort(["symbol", "ann_date"])
    )
    last_kept: dict[str, date] = {}
    keep = []
    for row in work.iter_rows(named=True):
        symbol = row["symbol"]
        event_date = row["ann_date"]
        previous = last_kept.get(symbol)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[symbol] = event_date
    return work.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol"]
    )


def _execution_lookup(execution: pl.DataFrame, index_name: str, prefix: str) -> pl.DataFrame:
    return execution.select(
        "symbol",
        pl.col("trade_index").alias(index_name),
        *[
            pl.col(column).alias(f"{prefix}_{column}")
            for column in (
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "limit_up_price",
                "limit_down_price",
                "excluded_name",
            )
        ],
    )


def build_trades(
    events: pl.DataFrame,
    minute_0931: pl.DataFrame,
    daily_context: pl.DataFrame,
) -> pl.DataFrame:
    execution = minute_0931.join(
        daily_context.select(
            "symbol",
            "date",
            "trade_index",
            "limit_up_price",
            "limit_down_price",
            "excluded_name",
        ),
        on=["symbol", "date"],
        how="left",
    )
    work = events.join(
        _execution_lookup(execution, "trade_index", "entry"),
        on=["symbol", "trade_index"],
        how="left",
    ).with_columns((pl.col("trade_index") + 1).alias("planned_exit_index"))
    for delay in range(MAX_EXIT_DELAY + 1):
        index_name = f"exit_index_{delay}"
        work = work.with_columns(
            (pl.col("planned_exit_index") + delay).alias(index_name)
        ).join(
            _execution_lookup(execution, index_name, f"exit_{delay}"),
            on=["symbol", index_name],
            how="left",
        )

    entry_sealed = (
        (pl.col("entry_low") >= pl.col("entry_limit_up_price") - 0.005)
        & (pl.col("entry_high") >= pl.col("entry_limit_up_price") - 0.005)
    ).fill_null(True)
    entry_valid = (
        pl.col("universe_eligible")
        & pl.col("entry_date").is_not_null()
        & ~pl.col("entry_excluded_name").fill_null(True)
        & (pl.col("entry_volume").fill_null(0) > 0)
        & (pl.col("entry_open").fill_null(0) > 0)
        & ~entry_sealed
        & (
            pl.col("entry_amount").fill_null(0) * DAILY_PARTICIPATION
            >= POSITION_NOTIONAL
        )
    )
    sellable = []
    for delay in range(MAX_EXIT_DELAY + 1):
        prefix = f"exit_{delay}"
        sealed_down = (
            (pl.col(f"{prefix}_open") <= pl.col(f"{prefix}_limit_down_price") + 0.005)
            & (pl.col(f"{prefix}_high") <= pl.col(f"{prefix}_limit_down_price") + 0.005)
        ).fill_null(True)
        sellable.append(
            (pl.col(f"{prefix}_volume").fill_null(0) > 0)
            & (pl.col(f"{prefix}_open").fill_null(0) > 0)
            & ~sealed_down
            & (
                pl.col(f"{prefix}_amount").fill_null(0) * DAILY_PARTICIPATION
                >= POSITION_NOTIONAL
            )
        )
    selected_delay = pl.coalesce(
        [
            pl.when(condition).then(pl.lit(delay))
            for delay, condition in enumerate(sellable)
        ]
    )
    work = work.with_columns(
        entry_valid.alias("entry_valid"), selected_delay.alias("exit_delay")
    )
    exit_date = pl.coalesce(
        [
            pl.when(pl.col("exit_delay") == delay).then(pl.col(f"exit_{delay}_date"))
            for delay in range(MAX_EXIT_DELAY + 1)
        ]
    )
    exit_open = pl.coalesce(
        [
            pl.when(pl.col("exit_delay") == delay).then(pl.col(f"exit_{delay}_open"))
            for delay in range(MAX_EXIT_DELAY + 1)
        ]
    )
    return (
        work.with_columns(
            (pl.col("entry_valid") & pl.col("exit_delay").is_not_null()).alias("tradable"),
            exit_date.alias("actual_exit_date"),
            exit_open.alias("actual_exit_open"),
        )
        .with_columns(
            (
                (
                    pl.col("actual_exit_open")
                    * (
                        1.0
                        - COMMISSION_PCT
                        - SLIPPAGE_PCT
                        - historical_stamp_tax(pl.col("actual_exit_date"))
                    )
                )
                / (pl.col("entry_open") * (1.0 + COMMISSION_PCT + SLIPPAGE_PCT))
                - 1.0
            ).alias("_net_return")
        )
        .with_columns(
            pl.when(pl.col("tradable"))
            .then(pl.col("_net_return"))
            .otherwise(None)
            .alias("net_return")
        )
        .drop("_net_return")
    )


def build_market_benchmark(
    minute_0931: pl.DataFrame, daily_context: pl.DataFrame
) -> pl.DataFrame:
    execution = minute_0931.join(
        daily_context.select(
            "symbol",
            "date",
            "trade_index",
            "excluded_name",
            "reference_close",
            "previous_amount",
        ),
        on=["symbol", "date"],
        how="left",
    )
    exits = execution.select(
        "symbol",
        (pl.col("trade_index") - 1).alias("trade_index"),
        pl.col("open").alias("benchmark_exit_open"),
    )
    return (
        execution.join(exits, on=["symbol", "trade_index"], how="inner")
        .filter(
            ~pl.col("excluded_name").fill_null(True)
            & pl.col("reference_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("previous_amount").fill_null(0) >= 20_000_000.0)
            & (pl.col("open").fill_null(0) > 0)
            & (pl.col("benchmark_exit_open").fill_null(0) > 0)
            & (pl.col("amount").fill_null(0) * DAILY_PARTICIPATION >= POSITION_NOTIONAL)
        )
        .with_columns(
            (pl.col("benchmark_exit_open") / pl.col("open") - 1.0).alias("market_return")
        )
        .group_by("trade_index")
        .agg(
            pl.col("market_return").median().alias("market_median_return"),
            pl.len().alias("market_symbols"),
        )
    )


def _cluster_t(frame: pl.DataFrame, column: str) -> float | None:
    daily = frame.drop_nulls(column).group_by("ann_date").agg(
        pl.col(column).mean().alias("return")
    )
    if daily.height <= 1:
        return None
    mean = daily.get_column("return").mean()
    std = daily.get_column("return").std(ddof=1)
    if mean is None or std in (None, 0.0):
        return None
    return mean / (std / math.sqrt(daily.height))


def summarize(trades: pl.DataFrame, category: str) -> dict[str, Any]:
    scoped = trades.filter(pl.col("category") == category)
    eligible = scoped.filter(pl.col("universe_eligible"))
    tradable = scoped.filter(pl.col("tradable"))
    benchmarked = tradable.filter(pl.col("excess_return").is_not_null())
    capacity_base = eligible.filter(
        pl.col("entry_date").is_not_null()
        & (pl.col("entry_volume").fill_null(0) > 0)
        & (pl.col("entry_open").fill_null(0) > 0)
    )
    capacity_feasible = capacity_base.filter(
        pl.col("entry_amount").fill_null(0) * DAILY_PARTICIPATION
        >= POSITION_NOTIONAL
    )
    monthly = (
        benchmarked.with_columns(pl.col("ann_date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(
            pl.col("net_return").mean().alias("mean_net_return"),
            pl.col("excess_return").mean().alias("mean_excess_return"),
            pl.col("excess_return").sum().alias("sum_excess_return"),
        )
        .sort("month")
    )
    positive_months = monthly.filter(pl.col("mean_excess_return") > 0).height
    positive_sums = [
        float(value)
        for value in monthly.get_column("sum_excess_return").to_list()
        if value is not None and value > 0
    ]
    maximum_month_share = (
        max(positive_sums) / sum(positive_sums) if positive_sums else None
    )
    result = {
        "events": scoped.height,
        "universe_eligible_events": eligible.height,
        "tradable_events": tradable.height,
        "benchmarked_events": benchmarked.height,
        "event_days": benchmarked.get_column("ann_date").n_unique() if benchmarked.height else 0,
        "tradable_rate": tradable.height / eligible.height if eligible.height else 0.0,
        "benchmark_coverage": benchmarked.height / tradable.height if tradable.height else 0.0,
        "entry_capacity_feasible_rate": capacity_feasible.height / capacity_base.height if capacity_base.height else 0.0,
        "unresolved_exits": eligible.filter(
            pl.col("entry_valid") & pl.col("exit_delay").is_null()
        ).height,
        "mean_net_return": benchmarked.get_column("net_return").mean() if benchmarked.height else None,
        "mean_excess_return": benchmarked.get_column("excess_return").mean() if benchmarked.height else None,
        "excess_daily_cluster_t": _cluster_t(benchmarked, "excess_return"),
        "positive_excess_months": positive_months,
        "max_month_positive_excess_share": maximum_month_share,
        "monthly": monthly.to_dicts(),
    }
    result["base_promotion_passed"] = bool(
        category in CANDIDATES
        and result["tradable_events"] >= 500
        and result["event_days"] >= 80
        and result["tradable_rate"] >= 0.90
        and result["benchmark_coverage"] >= 0.99
        and result["entry_capacity_feasible_rate"] >= 0.95
        and result["unresolved_exits"] == 0
        and (result["mean_net_return"] or -math.inf) >= 0.003
        and (result["mean_excess_return"] or -math.inf) >= 0.002
        and (result["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and result["positive_excess_months"] >= 4
        and (result["max_month_positive_excess_share"] or math.inf) <= 0.50
    )
    result["promotion_passed"] = result["base_promotion_passed"]
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    panel = load_panel(data_dir, CONTEXT_START, EXECUTION_END)
    daily_context = build_daily_context(panel)
    auction = load_auction(data_dir)
    minute_0931 = load_0931(data_dir)
    events = categorize_events(auction, daily_context)
    trades = build_trades(events, minute_0931, daily_context)
    benchmark = build_market_benchmark(minute_0931, daily_context)
    trades = trades.join(benchmark, on="trade_index", how="left").with_columns(
        pl.when(pl.col("tradable") & pl.col("market_median_return").is_not_null())
        .then(pl.col("net_return") - pl.col("market_median_return"))
        .otherwise(None)
        .alias("excess_return")
    )
    summaries = {category: summarize(trades, category) for category in CATEGORIES}
    comparisons = {
        "demand_continuation": "positive_gap_low_activity_control",
        "supply_absorption": "negative_gap_low_activity_control",
    }
    for candidate, control in comparisons.items():
        candidate_mean = summaries[candidate]["mean_excess_return"]
        control_mean = summaries[control]["mean_excess_return"]
        direction_specific = (
            candidate_mean is not None
            and control_mean is not None
            and candidate_mean > control_mean
        )
        summaries[candidate]["direction_specific_vs_control"] = direction_specific
        summaries[candidate]["promotion_passed"] = bool(
            summaries[candidate]["base_promotion_passed"] and direction_specific
        )
    promoted = [
        candidate for candidate in CANDIDATES if summaries[candidate]["promotion_passed"]
    ]
    selected = (
        max(promoted, key=lambda name: summaries[name]["excess_daily_cluster_t"])
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-opening-auction-imbalance-discovery-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "discovery_start": DISCOVERY_START,
            "discovery_end": DISCOVERY_END,
            "confirmation_read": False,
        },
        "assumptions": {
            "entry_time": "09:31 open after 09:30 auction record is complete",
            "planned_exit": "next trading day 09:31 open",
            "max_exit_delay": MAX_EXIT_DELAY,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "minimum_auction_amount_cny": MIN_AUCTION_AMOUNT,
            "minimum_active_share": MIN_ACTIVE_SHARE,
            "maximum_control_share": MAX_CONTROL_SHARE,
            "gap_range": [MIN_ABS_GAP, MAX_ABS_GAP],
            "position_notional_cny": POSITION_NOTIONAL,
            "minute_participation_rate": DAILY_PARTICIPATION,
        },
        "data": {
            "auction_rows": auction.height,
            "minute_0931_rows": minute_0931.height,
            "categorized_events": events.height,
            "panel_rows": daily_context.height,
            "market_entry_days": benchmark.height,
        },
        "categories": summaries,
        "decision": {
            "promoted_candidates": promoted,
            "selected_candidate": selected,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_selected_candidate_before_confirmation"
                if selected
                else "terminate_opening_auction_imbalance_mechanism"
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
            "/app/data/research/p0_opening_auction_imbalance_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
