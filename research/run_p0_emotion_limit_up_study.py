"""P0-A causal event study for the A-share limit-up emotion hypothesis."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.price_limits import (  # noqa: E402
    polars_is_risk_warning_name,
    polars_limit_price,
    polars_price_limit_pct,
)

START = date(2013, 8, 29)
DEVELOPMENT_END = date(2020, 12, 31)
VALIDATION_END = date(2023, 12, 31)
STAMP_TAX_CUT = date(2023, 8, 28)

COMMISSION_PCT = 0.0002
STAMP_TAX_OLD = 0.001
STAMP_TAX_CURRENT = 0.0005
SLIPPAGE_PCT = 5.0 / 10_000.0
POSITION_NOTIONAL = 100_000.0
DAILY_PARTICIPATION = 0.01
OPENING_PARTICIPATION = 0.05
MIN_LISTING_DAYS = 90
HOLD_DAYS = (1, 2, 3)
MAX_EXIT_DELAY = 5

MAIN_BOARD_PATTERN = r"^(?:00[0-3]\d{3}\.SZ|60[0135]\d{3}\.SH)$"

STATE_QUANTILES = {
    "breadth_q20": ("limit_up_rate", 0.20),
    "breadth_q30": ("limit_up_rate", 0.30),
    "breadth_q60": ("limit_up_rate", 0.60),
    "breadth_q90": ("limit_up_rate", 0.90),
    "ge2_q90": ("ge2_rate", 0.90),
    "promo_q30": ("promotion_rate", 0.30),
    "promo_q40": ("promotion_rate", 0.40),
    "promo_q50": ("promotion_rate", 0.50),
    "promo_q60": ("promotion_rate", 0.60),
    "winner_q30": ("winner_return", 0.30),
    "winner_q40": ("winner_return", 0.40),
    "winner_q50": ("winner_return", 0.50),
    "winner_q60": ("winner_return", 0.60),
    "broken_q70": ("broken_rate", 0.70),
    "up_q30": ("up_ratio", 0.30),
    "up_q40": ("up_ratio", 0.40),
    "up_q50": ("up_ratio", 0.50),
    "up_q60": ("up_ratio", 0.60),
    "up_q70": ("up_ratio", 0.70),
}

MAX_YEAR_POSITIVE_SHARE = 0.60
MIN_PROFITABLE_YEARS = 2
MIN_INDUSTRY_COVERAGE = 0.70
MAX_INDUSTRY_POSITIVE_SHARE = 0.35


def period_expr(value: pl.Expr) -> pl.Expr:
    return (
        pl.when(value <= pl.lit(DEVELOPMENT_END))
        .then(pl.lit("development"))
        .when(value <= pl.lit(VALIDATION_END))
        .then(pl.lit("validation"))
        .otherwise(pl.lit("known_stress"))
    )


def historical_stamp_tax_expr(value: pl.Expr) -> pl.Expr:
    return (
        pl.when(value < pl.lit(STAMP_TAX_CUT))
        .then(pl.lit(STAMP_TAX_OLD))
        .otherwise(pl.lit(STAMP_TAX_CURRENT))
    )


def load_daily_panel(
    data_dir: Path,
    end: date | None = None,
    *,
    start: date = START,
) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        return pl.DataFrame()
    query = (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "raw_close",
            "raw_high",
            "raw_low",
        )
        .filter(
            (pl.col("date") >= pl.lit(start))
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        )
    )
    if end is not None:
        query = query.filter(pl.col("date") <= pl.lit(end))
    return query.collect(engine="streaming")


def attach_point_in_time_universe(panel: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    root = data_dir / "research"
    universe_path = root / "historical_stock_universe.parquet"
    names_path = root / "historical_stock_names.parquet"
    if not universe_path.is_file() or not names_path.is_file():
        raise ValueError("P0-A requires point-in-time universe and historical names")
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
            & (
                (pl.col("date") - pl.col("list_date")).dt.total_days()
                >= MIN_LISTING_DAYS
            )
        )
        .drop("start_date", "end_date", "delist_date")
    )


def prepare_market_panel(panel: pl.DataFrame) -> pl.DataFrame:
    if panel.is_empty():
        return panel
    dates = panel.select("date").unique().sort("date").with_row_index("_global_index")
    work = panel.join(dates, on="date", how="left").sort(["symbol", "date"])
    is_st = polars_is_risk_warning_name(pl.col("name"))
    is_exit_name = pl.col("name").fill_null("").str.contains("退", literal=True)
    work = work.with_columns(
        is_st.alias("_is_st"),
        (is_st | is_exit_name).alias("_is_excluded"),
        (pl.col("close") / pl.col("raw_close")).alias("_adj_factor"),
        pl.col("_global_index").shift(1).over("symbol").alias("_prev_global_index"),
        pl.col("close").shift(1).over("symbol").alias("_prev_close"),
        pl.col("raw_close").shift(1).over("symbol").alias("_prev_raw_close_base"),
        (pl.col("close").shift(1).over("symbol") / pl.col("raw_close").shift(1).over("symbol"))
        .alias("_prev_adj_factor"),
    ).with_columns(
        (pl.col("_global_index") == pl.col("_prev_global_index") + 1).alias("_adjacent"),
        (pl.col("open") / pl.col("_adj_factor")).alias("raw_open"),
    )
    adj_changed = (pl.col("_adj_factor") - pl.col("_prev_adj_factor")).abs() > 1e-6
    work = work.with_columns(
        pl.when(pl.col("_adjacent"))
        .then(
            pl.when(adj_changed)
            .then(pl.col("_prev_close"))
            .otherwise(pl.col("_prev_raw_close_base"))
        )
        .otherwise(None)
        .alias("_reference_close"),
        polars_price_limit_pct(pl.col("symbol"), pl.col("date"), pl.col("_is_st"))
        .alias("_limit_pct"),
    ).with_columns(
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=True)
        .alias("limit_up_price"),
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=False)
        .alias("limit_down_price"),
    )
    valid_reference = pl.col("_reference_close").is_not_null() & (
        pl.col("_reference_close") > 0
    )
    work = work.with_columns(
        (
            valid_reference
            & (pl.col("raw_close") >= pl.col("limit_up_price") - 0.005)
        ).alias("is_limit_up"),
        (
            valid_reference
            & (pl.col("raw_close") <= pl.col("limit_down_price") + 0.005)
        ).alias("is_limit_down"),
        (
            valid_reference
            & (pl.col("raw_high") >= pl.col("limit_up_price") - 0.005)
        ).alias("hit_limit_up"),
    ).with_columns(
        (pl.col("hit_limit_up") & ~pl.col("is_limit_up")).alias("is_broken_board"),
        (
            pl.col("is_limit_up")
            & (pl.col("raw_open") >= pl.col("limit_up_price") - 0.005)
        ).alias("open_limit_up"),
        (
            pl.col("is_limit_down")
            & (pl.col("raw_open") <= pl.col("limit_down_price") + 0.005)
        ).alias("open_limit_down"),
    )
    work = work.with_columns(
        pl.col("is_limit_up").shift(1).over("symbol").fill_null(False).alias("_prev_limit_up"),
        pl.when(pl.col("_adjacent"))
        .then(pl.col("close") / pl.col("_prev_close") - 1.0)
        .otherwise(None)
        .alias("daily_return"),
    ).with_columns(
        (
            pl.col("_adjacent")
            & pl.col("_prev_limit_up")
        ).alias("_promotion_pool"),
        (
            pl.col("_adjacent")
            & pl.col("_prev_limit_up")
            & pl.col("is_limit_up")
        ).alias("_promotion_ok"),
    )
    reset = (~pl.col("is_limit_up") | ~pl.col("_adjacent")).cast(pl.UInt32)
    work = work.with_columns(reset.cum_sum().over("symbol").alias("_limit_group"))
    work = work.with_columns(
        pl.when(pl.col("is_limit_up"))
        .then(pl.col("is_limit_up").cast(pl.UInt32).cum_sum().over("symbol", "_limit_group"))
        .otherwise(0)
        .cast(pl.UInt32)
        .alias("consecutive_limit_ups")
    )
    return work.select(
        "symbol",
        "date",
        "name",
        "_global_index",
        "_is_excluded",
        "open",
        "close",
        "raw_close",
        "raw_open",
        "volume",
        "amount",
        "limit_up_price",
        "limit_down_price",
        "is_limit_up",
        "is_limit_down",
        "is_broken_board",
        "open_limit_up",
        "open_limit_down",
        "consecutive_limit_ups",
        "_promotion_pool",
        "_promotion_ok",
        "daily_return",
    )


def build_daily_state_inputs(panel: pl.DataFrame) -> pl.DataFrame:
    eligible = panel.filter(~pl.col("_is_excluded"))
    daily = (
        eligible.group_by("date")
        .agg(
            pl.len().alias("universe_count"),
            pl.col("is_limit_up").sum().alias("limit_up_count"),
            (pl.col("consecutive_limit_ups") == 1).sum().alias("first_board_count"),
            (pl.col("consecutive_limit_ups") >= 2).sum().alias("ge2_count"),
            pl.col("consecutive_limit_ups").max().alias("max_board"),
            pl.col("is_broken_board").sum().alias("broken_count"),
            pl.col("_promotion_pool").sum().alias("promotion_pool"),
            pl.col("_promotion_ok").sum().alias("promotion_ok"),
            pl.col("daily_return").filter(pl.col("_promotion_pool")).mean().alias("winner_return"),
            (pl.col("daily_return") > 0).mean().alias("up_ratio"),
        )
        .sort("date")
        .with_columns(
            (pl.col("limit_up_count") / pl.col("universe_count")).alias(
                "limit_up_rate"
            ),
            (pl.col("ge2_count") / pl.col("universe_count")).alias("ge2_rate"),
            pl.when(pl.col("promotion_pool") > 0)
            .then(pl.col("promotion_ok") / pl.col("promotion_pool"))
            .otherwise(0.0)
            .alias("promotion_rate"),
            pl.when(pl.col("limit_up_count") + pl.col("broken_count") > 0)
            .then(pl.col("limit_up_count") / (pl.col("limit_up_count") + pl.col("broken_count")))
            .otherwise(0.0)
            .alias("seal_rate"),
            pl.when(pl.col("limit_up_count") + pl.col("broken_count") > 0)
            .then(pl.col("broken_count") / (pl.col("limit_up_count") + pl.col("broken_count")))
            .otherwise(0.0)
            .alias("broken_rate"),
        )
        .with_columns(
            (
                pl.col("limit_up_rate")
                - pl.col("limit_up_rate")
                .shift(1)
                .rolling_mean(window_size=5, min_samples=3)
            ).alias("breadth_delta_5d"),
            pl.col("winner_return").fill_null(0.0),
            period_expr(pl.col("date")).alias("period"),
        )
    )
    return daily


def fit_state_thresholds(daily: pl.DataFrame) -> dict[str, float]:
    development = daily.filter(pl.col("date") <= pl.lit(DEVELOPMENT_END))
    if development.is_empty():
        raise ValueError("development period is empty")
    result: dict[str, float] = {}
    for key, (column, quantile) in STATE_QUANTILES.items():
        value = development.get_column(column).drop_nulls().quantile(quantile, "linear")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"cannot fit state threshold: {key}")
        result[key] = float(value)
    return result


def classify_states(daily: pl.DataFrame, thresholds: dict[str, float]) -> pl.DataFrame:
    c = pl.col
    climax = (
        (c("limit_up_rate") >= thresholds["breadth_q90"])
        | (c("ge2_rate") >= thresholds["ge2_q90"])
    ) & (c("up_ratio") >= thresholds["up_q70"])
    divergence = (c("limit_up_rate") >= thresholds["breadth_q60"]) & (
        (c("broken_rate") >= thresholds["broken_q70"])
        | (c("winner_return") <= thresholds["winner_q30"])
    )
    repair = (
        (c("limit_up_rate").shift(1) <= thresholds["breadth_q30"])
        & (c("breadth_delta_5d") > 0)
        & (c("promotion_rate") >= thresholds["promo_q50"])
        & (c("winner_return") >= thresholds["winner_q50"])
        & (c("up_ratio") >= thresholds["up_q50"])
    )
    ferment = (
        (c("limit_up_rate") >= thresholds["breadth_q60"])
        & (c("promotion_rate") >= thresholds["promo_q60"])
        & (c("winner_return") >= thresholds["winner_q60"])
        & (c("up_ratio") >= thresholds["up_q60"])
        & (c("breadth_delta_5d") > 0)
    )
    ice = (
        (c("limit_up_rate") <= thresholds["breadth_q20"])
        & (c("promotion_rate") <= thresholds["promo_q30"])
        & (c("winner_return") <= thresholds["winner_q30"])
        & (c("up_ratio") <= thresholds["up_q30"])
    )
    recede = (
        (c("breadth_delta_5d") < 0)
        & (c("promotion_rate") <= thresholds["promo_q40"])
        & (c("winner_return") <= thresholds["winner_q40"])
        & (c("up_ratio") <= thresholds["up_q40"])
    )
    return daily.with_columns(
        pl.when(climax).then(pl.lit("climax"))
        .when(divergence).then(pl.lit("divergence"))
        .when(repair).then(pl.lit("repair"))
        .when(ferment).then(pl.lit("ferment"))
        .when(ice).then(pl.lit("ice"))
        .when(recede).then(pl.lit("recede"))
        .otherwise(pl.lit("neutral"))
        .alias("state")
    )


def build_events(panel: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    event_type = (
        pl.when(pl.col("is_broken_board"))
        .then(pl.lit("broken_board"))
        .when(pl.col("consecutive_limit_ups") == 1)
        .then(pl.lit("first_board"))
        .when(pl.col("consecutive_limit_ups") == 2)
        .then(pl.lit("second_board"))
        .when(pl.col("consecutive_limit_ups") >= 3)
        .then(pl.lit("high_board"))
        .otherwise(None)
    )
    return (
        panel.filter(~pl.col("_is_excluded"))
        .with_columns(event_type.alias("event_type"))
        .drop_nulls("event_type")
        .select(
            "symbol",
            "date",
            "name",
            "_global_index",
            "close",
            pl.col("raw_close").alias("signal_raw_close"),
            pl.col("amount").alias("signal_day_amount"),
            "event_type",
        )
        .join(
            daily.select(
                "date",
                "period",
                "state",
                "limit_up_count",
                "promotion_rate",
                "seal_rate",
                "broken_rate",
                "winner_return",
                "breadth_delta_5d",
            ),
            on="date",
            how="left",
        )
        .rename({"date": "signal_date"})
    )


def attach_event_outcomes(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    position_notional: float = POSITION_NOTIONAL,
    daily_participation: float = DAILY_PARTICIPATION,
) -> pl.DataFrame:
    if events.is_empty():
        return events
    max_offset = max(HOLD_DAYS) + MAX_EXIT_DELAY
    future_base = panel.select(
        "symbol",
        pl.col("_global_index").alias("_target_index"),
        "date",
        "open",
        "raw_close",
        "raw_open",
        "volume",
        "amount",
        "limit_up_price",
        "limit_down_price",
    )
    work = events.with_row_index("_event_id")
    requests = (
        work.select("_event_id", "symbol", "_global_index")
        .join(pl.DataFrame({"_offset": range(1, max_offset + 1)}), how="cross")
        .with_columns(
            (pl.col("_global_index") + pl.col("_offset")).alias("_target_index")
        )
    )
    future = requests.join(
        future_base,
        on=["symbol", "_target_index"],
        how="left",
    )
    entry = future.filter(pl.col("_offset") == 1).select(
        "_event_id",
        pl.col("date").alias("entry_date"),
        pl.col("open").alias("entry_price"),
        pl.col("raw_close").alias("entry_raw_close"),
        pl.col("raw_open").alias("entry_raw_price"),
        pl.col("volume").alias("entry_volume"),
        pl.col("amount").alias("entry_day_amount"),
        pl.col("limit_up_price").alias("entry_limit_up_price"),
    )
    work = work.join(entry, on="_event_id", how="left")
    entry_present = (
        pl.col("entry_price").is_not_null()
        & (pl.col("entry_price") > 0)
        & (pl.col("entry_volume").fill_null(0) > 0)
    )
    entry_open_limit = (
        pl.col("entry_raw_price") >= pl.col("entry_limit_up_price") - 0.005
    )
    signal_capacity = (
        pl.col("signal_day_amount").fill_null(0) * daily_participation
        >= position_notional
    )
    entry_capacity = (
        pl.col("entry_day_amount").fill_null(0) * daily_participation
        >= position_notional
    )
    entry_valid = (
        entry_present
        & ~entry_open_limit.fill_null(True)
        & signal_capacity
        & entry_capacity
    )
    work = work.with_columns(
        entry_valid.alias("entry_valid"),
        pl.when(~entry_present)
        .then(pl.lit("missing_or_suspended"))
        .when(entry_open_limit.fill_null(True))
        .then(pl.lit("open_limit_up"))
        .when(~signal_capacity)
        .then(pl.lit("insufficient_signal_day_capacity"))
        .when(~entry_capacity)
        .then(pl.lit("insufficient_entry_day_capacity"))
        .otherwise(pl.lit("ok"))
        .alias("entry_status"),
        (pl.col("entry_price") / pl.col("close") - 1.0).alias("entry_gap"),
    )
    for hold in HOLD_DAYS:
        exit_candidates = (
            future.filter(
                pl.col("_offset").is_between(
                    hold + 1,
                    hold + MAX_EXIT_DELAY,
                )
            )
            .filter(
                pl.col("open").is_not_null()
                & (pl.col("open") > 0)
                & (pl.col("volume").fill_null(0) > 0)
                & (
                    pl.col("raw_open")
                    > pl.col("limit_down_price") + 0.005
                ).fill_null(False)
            )
            .sort(["_event_id", "_offset"])
            .group_by("_event_id", maintain_order=True)
            .agg(
                pl.col("open").first().alias(f"exit_price_h{hold}"),
                pl.col("raw_open").first().alias(f"exit_raw_price_h{hold}"),
                pl.col("date").first().alias(f"exit_date_h{hold}"),
                pl.col("_offset").first().cast(pl.Int32).alias(f"exit_offset_h{hold}"),
            )
        )
        work = work.join(exit_candidates, on="_event_id", how="left")
        work = work.with_columns(
            pl.col(f"exit_price_h{hold}").is_not_null().alias(f"exit_valid_h{hold}"),
            (
                pl.col("entry_valid") & pl.col(f"exit_price_h{hold}").is_not_null()
            ).alias(f"tradable_h{hold}"),
            (pl.col(f"exit_price_h{hold}") / pl.col("entry_price") - 1.0)
            .alias(f"gross_return_h{hold}"),
            (
                (
                    pl.col(f"exit_price_h{hold}")
                    * (
                        1.0
                        - COMMISSION_PCT
                        - SLIPPAGE_PCT
                        - historical_stamp_tax_expr(pl.col(f"exit_date_h{hold}"))
                    )
                )
                / (pl.col("entry_price") * (1.0 + COMMISSION_PCT + SLIPPAGE_PCT))
                - 1.0
            ).alias(f"_net_return_h{hold}"),
        ).with_columns(
            pl.when(pl.col(f"tradable_h{hold}"))
            .then(pl.col(f"_net_return_h{hold}"))
            .otherwise(None)
            .alias(f"net_return_h{hold}")
        ).drop(f"_net_return_h{hold}")
    return work.drop("_event_id", "_global_index")


def outcomes_long(outcomes: pl.DataFrame) -> pl.DataFrame:
    rows = []
    base = [
        "symbol",
        "name",
        "signal_date",
        "entry_date",
        "period",
        "state",
        "event_type",
        "industry",
        "entry_status",
        "entry_gap",
    ]
    for hold in HOLD_DAYS:
        rows.append(
            outcomes.select(
                *base,
                pl.lit(hold).alias("hold_days"),
                pl.col(f"tradable_h{hold}").alias("tradable"),
                pl.col(f"gross_return_h{hold}").alias("gross_return"),
                pl.col(f"net_return_h{hold}").alias("net_return"),
                pl.col(f"exit_offset_h{hold}").alias("exit_offset"),
                pl.col(f"exit_date_h{hold}").alias("exit_date"),
            )
        )
    return pl.concat(rows, how="vertical")


def summarize_outcomes(long: pl.DataFrame) -> pl.DataFrame:
    keys = ["period", "state", "event_type", "hold_days"]
    event_summary = long.group_by(keys).agg(
        pl.len().alias("event_count"),
        pl.col("signal_date").n_unique().alias("signal_day_count"),
        pl.col("tradable").sum().alias("tradable_count"),
        pl.col("net_return").mean().alias("event_mean_net"),
        pl.col("net_return").median().alias("event_median_net"),
        (pl.col("net_return") > 0).mean().alias("event_win_rate"),
        pl.col("net_return").quantile(0.05, "linear").alias("event_p05"),
        pl.col("entry_gap").median().alias("median_entry_gap"),
        (pl.col("entry_status") == "open_limit_up").mean().alias("open_limit_block_rate"),
        pl.col("entry_status").is_in(
            [
                "insufficient_signal_day_capacity",
                "insufficient_entry_day_capacity",
            ]
        ).mean()
        .alias("capacity_block_rate"),
    ).with_columns(
        (pl.col("tradable_count") / pl.col("event_count")).alias("tradable_rate")
    )
    daily = (
        long.filter(pl.col("tradable"))
        .group_by(*keys, "signal_date")
        .agg(pl.col("net_return").mean().alias("daily_cluster_return"))
    )
    clustered = daily.group_by(keys).agg(
        pl.len().alias("cluster_day_count"),
        pl.col("daily_cluster_return").mean().alias("cluster_mean_net"),
        pl.col("daily_cluster_return").std(ddof=1).alias("cluster_std"),
        (pl.col("daily_cluster_return") > 0).mean().alias("cluster_win_rate"),
    ).with_columns(
        pl.when((pl.col("cluster_day_count") > 1) & (pl.col("cluster_std") > 0))
        .then(
            pl.col("cluster_mean_net")
            / (pl.col("cluster_std") / pl.col("cluster_day_count").sqrt())
        )
        .otherwise(None)
        .alias("cluster_t")
    )
    return event_summary.join(clustered, on=keys, how="left").sort(keys)


def attach_point_in_time_industry(
    outcomes: pl.DataFrame,
    data_dir: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    membership_path = data_dir / "research" / "sw_l1_membership.parquet"
    if not membership_path.is_file():
        return (
            outcomes.with_columns(pl.lit(None).cast(pl.Utf8).alias("industry")),
            {
                "available": False,
                "note": "缺少点时申万一级行业, 候选不得通过集中度门槛",
            },
        )
    membership = (
        pl.read_parquet(membership_path)
        .with_columns(
            pl.col("in_date").cast(pl.Date, strict=False),
            pl.col("out_date").cast(pl.Date, strict=False),
        )
        .select("symbol", "l1_name", "in_date", "out_date")
        .drop_nulls(["symbol", "l1_name", "in_date"])
        .sort(["symbol", "in_date"])
    )
    joined = (
        outcomes.sort(["symbol", "signal_date"])
        .join_asof(
            membership,
            left_on="signal_date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            pl.when(
                pl.col("out_date").is_null()
                | (pl.col("signal_date") <= pl.col("out_date"))
            )
            .then(pl.col("l1_name"))
            .otherwise(None)
            .alias("industry")
        )
        .drop("l1_name", "in_date", "out_date")
    )
    covered = joined.select(pl.col("industry").is_not_null().mean()).item()
    return joined, {
        "available": True,
        "membership_rows": membership.height,
        "outcome_coverage": float(covered or 0.0),
        "note": "点时申万一级行业作为历史题材集中度的保守代理; 概念前向另行审计",
    }


def concentration_audits(long: pl.DataFrame) -> pl.DataFrame:
    keys = ["period", "state", "event_type", "hold_days"]
    tradable = long.filter(pl.col("tradable"))
    if tradable.is_empty():
        return pl.DataFrame()
    daily = (
        tradable.group_by(*keys, "signal_date")
        .agg(pl.col("net_return").mean().alias("daily_cluster_return"))
        .with_columns(pl.col("signal_date").dt.year().alias("year"))
    )
    by_year = daily.group_by(*keys, "year").agg(
        pl.col("daily_cluster_return").sum().alias("year_return_sum")
    )
    year_audit = by_year.group_by(keys).agg(
        (pl.col("year_return_sum") > 0).sum().alias("profitable_year_count"),
        pl.when(pl.col("year_return_sum").clip(lower_bound=0).sum() > 0)
        .then(
            pl.col("year_return_sum").clip(lower_bound=0).max()
            / pl.col("year_return_sum").clip(lower_bound=0).sum()
        )
        .otherwise(None)
        .alias("top_year_positive_share"),
    )
    industry_coverage = tradable.group_by(keys).agg(
        pl.col("industry").is_not_null().mean().alias("industry_coverage")
    )
    by_industry = (
        tradable.drop_nulls("industry")
        .group_by(*keys, "industry")
        .agg(pl.col("net_return").sum().alias("industry_return_sum"))
    )
    industry_audit = by_industry.group_by(keys).agg(
        pl.when(pl.col("industry_return_sum").clip(lower_bound=0).sum() > 0)
        .then(
            pl.col("industry_return_sum").clip(lower_bound=0).max()
            / pl.col("industry_return_sum").clip(lower_bound=0).sum()
        )
        .otherwise(None)
        .alias("top_industry_positive_share"),
        pl.len().alias("industry_count"),
    )
    return (
        year_audit.join(industry_coverage, on=keys, how="full", coalesce=True)
        .join(industry_audit, on=keys, how="full", coalesce=True)
        .sort(keys)
    )


def execution_failure_audits(outcomes: pl.DataFrame) -> dict[str, Any]:
    entry_status = (
        outcomes.group_by("period", "entry_status")
        .len()
        .sort("period", "entry_status")
        .to_dicts()
    )
    exits: list[dict[str, Any]] = []
    for hold in HOLD_DAYS:
        exits.extend(
            outcomes.group_by("period")
            .agg(
                pl.lit(hold).alias("hold_days"),
                (
                    pl.col("entry_valid")
                    & pl.col(f"exit_date_h{hold}").is_null()
                ).sum().alias("exit_unavailable"),
                (
                    pl.col(f"exit_offset_h{hold}").is_not_null()
                    & (pl.col(f"exit_offset_h{hold}") > hold + 1)
                ).sum().alias("exit_deferred"),
            )
            .sort("period")
            .to_dicts()
        )
    return {"entry_status": entry_status, "exit_status": exits}


def evaluate_candidates(
    summary: pl.DataFrame,
    concentration: pl.DataFrame,
) -> list[dict[str, Any]]:
    keys = ["state", "event_type", "hold_days"]
    if not concentration.is_empty():
        summary = summary.join(
            concentration,
            on=["period", *keys],
            how="left",
        )
    candidates = []
    for key_values in summary.select(keys).unique().iter_rows(named=True):
        scoped = summary
        for key, value in key_values.items():
            scoped = scoped.filter(pl.col(key) == value)
        periods = {row["period"]: row for row in scoped.to_dicts()}
        validation = periods.get("validation")
        stress = periods.get("known_stress")
        passed = bool(
            validation
            and stress
            and (validation.get("cluster_mean_net") or -1.0) >= 0.005
            and (stress.get("cluster_mean_net") or -1.0) >= 0.005
            and (validation.get("cluster_day_count") or 0) >= 30
            and (stress.get("cluster_day_count") or 0) >= 30
            and (validation.get("tradable_rate") or 0.0) >= 0.70
            and (stress.get("tradable_rate") or 0.0) >= 0.70
            and (validation.get("cluster_t") or -99.0) > 1.0
            and (stress.get("cluster_t") or -99.0) > 1.0
            and (validation.get("profitable_year_count") or 0)
            >= MIN_PROFITABLE_YEARS
            and (stress.get("profitable_year_count") or 0)
            >= MIN_PROFITABLE_YEARS
            and (validation.get("top_year_positive_share") or 99.0)
            <= MAX_YEAR_POSITIVE_SHARE
            and (stress.get("top_year_positive_share") or 99.0)
            <= MAX_YEAR_POSITIVE_SHARE
            and (validation.get("industry_coverage") or 0.0)
            >= MIN_INDUSTRY_COVERAGE
            and (stress.get("industry_coverage") or 0.0)
            >= MIN_INDUSTRY_COVERAGE
            and (validation.get("top_industry_positive_share") or 99.0)
            <= MAX_INDUSTRY_POSITIVE_SHARE
            and (stress.get("top_industry_positive_share") or 99.0)
            <= MAX_INDUSTRY_POSITIVE_SHARE
        )
        candidates.append({**key_values, "passed": passed, "periods": periods})
    return sorted(candidates, key=lambda row: (not row["passed"], row["state"], row["event_type"], row["hold_days"]))


def load_opening_execution(data_dir: Path, entry_keys: pl.DataFrame) -> dict[str, Any]:
    auction_paths = sorted((data_dir / "tushare_supplemental" / "auction").glob("date=*/part.parquet"))
    if not auction_paths or entry_keys.is_empty():
        return {"auction_available": False, "minute_available": False}
    keys = (
        entry_keys.select(
            "symbol",
            pl.col("entry_date").alias("date"),
            pl.col("entry_raw_price").alias("daily_open"),
        )
        .drop_nulls(["symbol", "date", "daily_open"])
        .unique(subset=["symbol", "date"])
    )
    auction = (
        pl.scan_parquet(auction_paths)
        .filter(pl.col("session") == "open")
        .select(
            "symbol",
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("open").alias("auction_price"),
            pl.col("amount").alias("auction_amount"),
        )
        .collect(engine="streaming")
        .join(keys, on=["symbol", "date"], how="inner")
    )
    auction_metrics = {
        "auction_available": True,
        "matched_entries": auction.height,
        "covered_entry_days": auction.get_column("date").n_unique() if auction.height else 0,
        "median_auction_amount": auction.get_column("auction_amount").median() if auction.height else None,
        "median_auction_vs_daily_open": (
            auction.select((pl.col("auction_price") / pl.col("daily_open") - 1.0).median()).item()
            if auction.height
            else None
        ),
        "capacity_100k_rate": (
            auction.select(
                (pl.col("auction_amount") * OPENING_PARTICIPATION >= POSITION_NOTIONAL).mean()
            ).item()
            if auction.height
            else None
        ),
    }
    minute_paths = sorted((data_dir / "kline_minute").glob("date=*/part.parquet"))
    if not minute_paths:
        return {**auction_metrics, "minute_available": False}
    first_five = (
        pl.scan_parquet(minute_paths)
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .filter(
            (pl.col("datetime").dt.hour() == 9)
            & (pl.col("datetime").dt.minute().is_between(30, 34))
        )
        .select("symbol", "date", "amount", "volume")
        .join(keys.lazy(), on=["symbol", "date"], how="inner")
        .group_by("symbol", "date")
        .agg(
            pl.col("amount").sum().alias("amount_5m"),
            pl.col("volume").sum().alias("volume_5m"),
        )
        .with_columns(
            pl.when(pl.col("volume_5m") > 0)
            .then(pl.col("amount_5m") / (pl.col("volume_5m") * 100.0))
            .otherwise(None)
            .alias("vwap_5m")
        )
        .collect(engine="streaming")
    )
    return {
        **auction_metrics,
        "minute_available": True,
        "minute_matched_entries": first_five.height,
        "median_first_5m_amount": first_five.get_column("amount_5m").median() if first_five.height else None,
        "median_first_5m_vwap_vs_daily_open": (
            first_five.join(keys, on=["symbol", "date"], how="left")
            .select((pl.col("vwap_5m") / pl.col("daily_open") - 1.0).median())
            .item()
            if first_five.height
            else None
        ),
        "first_5m_capacity_100k_rate": (
            first_five.select(
                (pl.col("amount_5m") * OPENING_PARTICIPATION >= POSITION_NOTIONAL).mean()
            ).item()
            if first_five.height
            else None
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path, *, end: date | None = None) -> dict[str, Any]:
    source = load_daily_panel(data_dir, end=end)
    if source.is_empty():
        raise ValueError("no daily enriched data")
    source_rows = source.height
    pit = attach_point_in_time_universe(source, data_dir)
    pit_rows = pit.height
    del source
    gc.collect()
    panel = prepare_market_panel(pit)
    del pit
    gc.collect()
    symbols = panel.get_column("symbol").n_unique()
    first_date = panel.get_column("date").min()
    last_date = panel.get_column("date").max()
    daily_inputs = build_daily_state_inputs(panel)
    thresholds = fit_state_thresholds(daily_inputs)
    daily = classify_states(daily_inputs, thresholds)
    del daily_inputs
    events = build_events(panel, daily)
    event_rows = events.height
    execution_panel = panel.select(
        "symbol",
        "date",
        "_global_index",
        "open",
        "raw_close",
        "raw_open",
        "volume",
        "amount",
        "limit_up_price",
        "limit_down_price",
    )
    del panel
    gc.collect()
    outcomes = attach_event_outcomes(events, execution_panel)
    del events, execution_panel
    gc.collect()
    outcomes, industry_metadata = attach_point_in_time_industry(outcomes, data_dir)
    failure_audits = execution_failure_audits(outcomes)
    long = outcomes_long(outcomes)
    summary = summarize_outcomes(long)
    concentration = concentration_audits(long)
    candidates = evaluate_candidates(summary, concentration)
    entry_keys = outcomes.select("symbol", "entry_date", "entry_raw_price")
    del outcomes, long
    gc.collect()
    execution = load_opening_execution(data_dir, entry_keys)
    passed = [row for row in candidates if row["passed"]]
    payload = {
        "schema_version": "p0-emotion-limit-up-v1",
        "contract": {
            "development_end": DEVELOPMENT_END,
            "validation_end": VALIDATION_END,
            "end": end or last_date,
            "universe": "main_board_pit_non_st_90_calendar_days",
            "signal_clock": "after_close_t",
            "entry": "strict_open_t_plus_1",
            "holds": HOLD_DAYS,
            "max_exit_delay": MAX_EXIT_DELAY,
            "position_notional": POSITION_NOTIONAL,
            "daily_participation": DAILY_PARTICIPATION,
            "concentration_gates": {
                "max_year_positive_share": MAX_YEAR_POSITIVE_SHARE,
                "min_profitable_years": MIN_PROFITABLE_YEARS,
                "min_industry_coverage": MIN_INDUSTRY_COVERAGE,
                "max_industry_positive_share": MAX_INDUSTRY_POSITIVE_SHARE,
            },
            "costs": {
                "commission_pct_each_side": COMMISSION_PCT,
                "slippage_pct_each_side": SLIPPAGE_PCT,
                "stamp_tax_old": STAMP_TAX_OLD,
                "stamp_tax_current": STAMP_TAX_CURRENT,
                "stamp_tax_cut": STAMP_TAX_CUT,
            },
        },
        "data": {
            "source_rows": source_rows,
            "pit_rows": pit_rows,
            "symbols": symbols,
            "first_date": first_date,
            "last_date": last_date,
            "daily_state_rows": daily.height,
            "event_rows": event_rows,
        },
        "industry_metadata": industry_metadata,
        "state_thresholds": thresholds,
        "state_distribution": daily.group_by("period", "state").len().sort("period", "state").to_dicts(),
        "event_summary": summary.to_dicts(),
        "concentration_audits": concentration.to_dicts(),
        "execution_failure_audits": failure_audits,
        "candidate_gates": candidates,
        "execution_calibration": execution,
        "decision": {
            "verdict": "CONTINUE" if passed else "DOWNGRADE",
            "passed_candidate_count": len(passed),
            "reason": (
                "至少一个冻结组合同时通过验证期和已知压力期门槛"
                if passed
                else "没有冻结组合同时通过验证期和已知压力期门槛"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "data": payload["data"],
                "execution_calibration": execution,
                "decision": payload["decision"],
                "passed_candidates": [
                    {key: row[key] for key in ("state", "event_type", "hold_days")}
                    for row in passed
                ],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_emotion_limit_up.json"),
    )
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    run(args.data_dir, args.output, end=args.end)


if __name__ == "__main__":
    main()
