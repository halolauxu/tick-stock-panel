"""Causal weekly micro-cap premium baseline with point-in-time share capital."""
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

from app.price_limits import polars_limit_price  # noqa: E402

START = date(2013, 8, 29)
DEVELOPMENT_END = date(2020, 12, 31)
VALIDATION_END = date(2023, 12, 31)
CHINEXT_LIMIT_CHANGE = date(2020, 8, 24)
STAMP_TAX_CUT = date(2023, 8, 28)

COMMISSION_PCT = 0.0002
SLIPPAGE_PCT = 0.0005
STAMP_TAX_OLD = 0.001
STAMP_TAX_CURRENT = 0.0005
POSITION_NOTIONAL = 15_000.0
DAILY_PARTICIPATION = 0.01
MIN_LISTING_DAYS = 180
MIN_TRADABLE_RATE = 0.80
MIN_ANNUAL_EXCESS = 0.10
MIN_POSITIVE_EXCESS_YEARS = 2

SYMBOL_PATTERN = r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$"


def board_symbol_counts(frame: pl.DataFrame) -> dict[str, int]:
    return {
        row["board"]: row["len"]
        for row in (
            frame.select("symbol")
            .unique()
            .with_columns(
                pl.when(pl.col("symbol").str.starts_with("00"))
                .then(pl.lit("sz_main"))
                .when(pl.col("symbol").str.starts_with("30"))
                .then(pl.lit("chinext"))
                .when(pl.col("symbol").str.starts_with("60"))
                .then(pl.lit("sh_main"))
                .when(pl.col("symbol").str.starts_with("68"))
                .then(pl.lit("star"))
                .otherwise(pl.lit("unexpected"))
                .alias("board")
            )
            .group_by("board")
            .len()
            .sort("board")
            .to_dicts()
        )
    }


def period_expr(value: pl.Expr) -> pl.Expr:
    return (
        pl.when(value <= pl.lit(DEVELOPMENT_END))
        .then(pl.lit("development"))
        .when(value <= pl.lit(VALIDATION_END))
        .then(pl.lit("validation"))
        .otherwise(pl.lit("known_stress"))
    )


def historical_stamp_tax(value: pl.Expr) -> pl.Expr:
    return (
        pl.when(value < pl.lit(STAMP_TAX_CUT))
        .then(pl.lit(STAMP_TAX_OLD))
        .otherwise(pl.lit(STAMP_TAX_CURRENT))
    )


def load_daily(data_dir: Path, end: date | None = None) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        return pl.DataFrame()
    query = (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "close",
            "volume",
            "amount",
            "raw_close",
        )
        .filter(
            (pl.col("date") >= pl.lit(START))
            & pl.col("symbol").str.contains(SYMBOL_PATTERN)
        )
    )
    if end is not None:
        query = query.filter(pl.col("date") <= pl.lit(end))
    return query.collect(engine="streaming")


def load_share_history(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "financials" / "shares").glob("*.parquet"))
    if not paths:
        raise ValueError("historical share capital is required")
    shares = pl.concat(
        [pl.read_parquet(path) for path in paths],
        how="diagonal_relaxed",
    )
    return (
        shares.with_columns(
            pl.coalesce(
                pl.col("announce_date").cast(pl.Utf8).str.to_date(strict=False),
                pl.col("period_end").cast(pl.Utf8).str.to_date(strict=False),
            ).alias("available_date"),
            pl.col("period_end")
            .cast(pl.Utf8)
            .str.to_date(strict=False)
            .alias("period_date"),
            pl.col("total_shares").cast(pl.Float64, strict=False),
            pl.col("float_shares").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("available_date").is_not_null()
            & (pl.col("total_shares") > 0)
            & (pl.col("float_shares") > 0)
            & (pl.col("float_shares") <= pl.col("total_shares"))
        )
        .sort(["symbol", "available_date", "period_date"])
        .unique(subset=["symbol", "available_date"], keep="last")
        .select(
            "symbol",
            "available_date",
            "total_shares",
            "float_shares",
        )
        .sort(["symbol", "available_date"])
    )


def attach_point_in_time_data(panel: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    research = data_dir / "research"
    universe_path = research / "historical_stock_universe_all_a.parquet"
    names_path = research / "historical_stock_names_all_a.parquet"
    if not universe_path.is_file() or not names_path.is_file():
        raise ValueError(
            "all-A PIT security master is required; run "
            "collect_all_a_pit_security_master.py first"
        )
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
    shares = load_share_history(data_dir)
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
            & (
                (pl.col("date") - pl.col("list_date")).dt.total_days()
                >= MIN_LISTING_DAYS
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
            & ~pl.col("name")
            .str.to_uppercase()
            .str.contains(r"(?:\*?ST|退)")
        )
        .join_asof(
            shares,
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            (pl.col("total_shares") > 0)
            & (pl.col("float_shares") > 0)
            & (pl.col("float_shares") <= pl.col("total_shares"))
        )
        .drop(
            "delist_date",
            "start_date",
            "end_date",
            "available_date",
        )
    )


def price_limit_pct() -> pl.Expr:
    is_chinext = pl.col("symbol").str.starts_with("30")
    is_star = pl.col("symbol").str.starts_with("68")
    return (
        pl.when(is_star)
        .then(pl.lit(0.20))
        .when(is_chinext & (pl.col("date") >= pl.lit(CHINEXT_LIMIT_CHANGE)))
        .then(pl.lit(0.20))
        .otherwise(pl.lit(0.10))
    )


def prepare_panel(panel: pl.DataFrame) -> pl.DataFrame:
    dates = panel.select("date").unique().sort("date").with_row_index("_global_index")
    work = panel.join(dates, on="date", how="left").sort(["symbol", "date"])
    work = work.with_columns(
        (pl.col("close") / pl.col("raw_close")).alias("_adj_factor"),
        pl.col("_global_index").shift(1).over("symbol").alias("_prev_index"),
        pl.col("close").shift(1).over("symbol").alias("_prev_close"),
        pl.col("raw_close").shift(1).over("symbol").alias("_prev_raw_close"),
        (
            pl.col("close").shift(1).over("symbol")
            / pl.col("raw_close").shift(1).over("symbol")
        ).alias("_prev_adj_factor"),
    ).with_columns(
        (pl.col("_global_index") == pl.col("_prev_index") + 1).alias("_adjacent"),
        (pl.col("open") / pl.col("_adj_factor")).alias("raw_open"),
    )
    factor_changed = (
        pl.col("_adj_factor") - pl.col("_prev_adj_factor")
    ).abs() > 1e-6
    work = work.with_columns(
        pl.when(pl.col("_adjacent"))
        .then(
            pl.when(factor_changed)
            .then(pl.col("_prev_close"))
            .otherwise(pl.col("_prev_raw_close"))
        )
        .otherwise(None)
        .alias("_reference_close"),
        price_limit_pct().alias("_limit_pct"),
        pl.when(pl.col("_adjacent"))
        .then(pl.col("close") / pl.col("_prev_close") - 1.0)
        .otherwise(None)
        .alias("daily_return"),
    ).with_columns(
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=True)
        .alias("limit_up_price"),
        polars_limit_price(pl.col("_reference_close"), pl.col("_limit_pct"), up=False)
        .alias("limit_down_price"),
        (pl.col("raw_close") * pl.col("total_shares")).alias("market_cap"),
    ).with_columns(
        (
            pl.col("_reference_close").is_not_null()
            & (pl.col("raw_close") <= pl.col("limit_down_price") + 0.005)
        ).alias("is_limit_down")
    )
    return work.select(
        "symbol",
        "date",
        "name",
        "_global_index",
        "list_date",
        "open",
        "raw_open",
        "close",
        "raw_close",
        "volume",
        "amount",
        "total_shares",
        "float_shares",
        "market_cap",
        "daily_return",
        "limit_up_price",
        "limit_down_price",
        "is_limit_down",
    )


def build_weekly_observations(panel: pl.DataFrame) -> pl.DataFrame:
    weekly_dates = (
        panel.select("date", "_global_index")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").dt.strftime("%G-%V").alias("week"))
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("_global_index").max().alias("signal_index"),
        )
        .sort("signal_date")
        .with_columns(
            (pl.col("signal_index") + 1).alias("entry_index"),
            (pl.col("signal_index").shift(-1) + 1).alias("exit_index"),
        )
        .drop_nulls("exit_index")
    )
    signal = (
        panel.join(
            weekly_dates.select(
                "signal_date",
                "signal_index",
                "entry_index",
                "exit_index",
            ),
            left_on=["date", "_global_index"],
            right_on=["signal_date", "signal_index"],
            how="inner",
        )
        .filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("cap_rank") - 1) * 10 / pl.col("universe_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("cap_decile"),
            period_expr(pl.col("date")).alias("period"),
        )
    )
    lookup = panel.select(
        "symbol",
        pl.col("_global_index").alias("lookup_index"),
        pl.col("date").alias("lookup_date"),
        pl.col("open").alias("lookup_open"),
        pl.col("raw_open").alias("lookup_raw_open"),
        pl.col("volume").alias("lookup_volume"),
        pl.col("amount").alias("lookup_amount"),
        pl.col("limit_up_price").alias("lookup_limit_up"),
        pl.col("limit_down_price").alias("lookup_limit_down"),
    )
    work = signal.join(
        lookup,
        left_on=["symbol", "entry_index"],
        right_on=["symbol", "lookup_index"],
        how="left",
    ).rename(
        {
            "lookup_date": "entry_date",
            "lookup_open": "entry_open",
            "lookup_raw_open": "entry_raw_open",
            "lookup_volume": "entry_volume",
            "lookup_amount": "entry_amount",
            "lookup_limit_up": "entry_limit_up",
            "lookup_limit_down": "entry_limit_down",
        }
    )
    work = work.join(
        lookup,
        left_on=["symbol", "exit_index"],
        right_on=["symbol", "lookup_index"],
        how="left",
        suffix="_exit",
    ).rename(
        {
            "lookup_date": "exit_date",
            "lookup_open": "exit_open",
            "lookup_raw_open": "exit_raw_open",
            "lookup_volume": "exit_volume",
            "lookup_amount": "exit_amount",
            "lookup_limit_up": "exit_limit_up",
            "lookup_limit_down": "exit_limit_down",
        }
    )
    mark_valid = (
        pl.col("entry_open").is_not_null()
        & (pl.col("entry_open") > 0)
        & pl.col("exit_open").is_not_null()
        & (pl.col("exit_open") > 0)
    )
    entry_valid = (
        mark_valid
        & (pl.col("entry_volume").fill_null(0) > 0)
        & (
            pl.col("entry_raw_open")
            < pl.col("entry_limit_up") - 0.005
        ).fill_null(False)
        & (pl.col("amount") * DAILY_PARTICIPATION >= POSITION_NOTIONAL)
        & (
            pl.col("entry_amount").fill_null(0) * DAILY_PARTICIPATION
            >= POSITION_NOTIONAL
        )
    )
    exit_valid = (
        (pl.col("exit_volume").fill_null(0) > 0)
        & (
            pl.col("exit_raw_open")
            > pl.col("exit_limit_down") + 0.005
        ).fill_null(False)
    )
    tradable = entry_valid & exit_valid
    return (
        work.with_columns(
            mark_valid.alias("mark_valid"),
            tradable.alias("tradable"),
            pl.when(mark_valid)
            .then(pl.col("exit_open") / pl.col("entry_open") - 1.0)
            .otherwise(None)
            .alias("mark_return"),
            (
                (
                    pl.col("exit_open")
                    * (
                        1.0
                        - COMMISSION_PCT
                        - SLIPPAGE_PCT
                        - historical_stamp_tax(pl.col("exit_date"))
                    )
                )
                / (
                    pl.col("entry_open")
                    * (1.0 + COMMISSION_PCT + SLIPPAGE_PCT)
                )
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


def weekly_portfolios(observations: pl.DataFrame) -> pl.DataFrame:
    return (
        observations.group_by("date", "period")
        .agg(
            pl.col("mark_return").filter(pl.col("cap_decile") == 0).mean().alias("bottom_mark"),
            pl.col("net_return").filter(pl.col("cap_decile") == 0).mean().alias("bottom_net"),
            pl.col("mark_return").mean().alias("market_mark"),
            pl.col("net_return").mean().alias("market_net"),
            pl.col("mark_return").filter(pl.col("cap_decile") == 9).mean().alias("top_mark"),
            pl.col("net_return").filter(pl.col("cap_decile") == 9).mean().alias("top_net"),
            pl.len().filter(pl.col("cap_decile") == 0).alias("bottom_count"),
            pl.col("tradable").filter(pl.col("cap_decile") == 0).mean().alias("bottom_tradable_rate"),
            pl.col("amount").filter(pl.col("cap_decile") == 0).median().alias("bottom_median_amount"),
            pl.col("amount").median().alias("market_median_amount"),
            (pl.col("daily_return") > 0).filter(pl.col("cap_decile") == 0).mean().alias("bottom_breadth"),
            (pl.col("daily_return") > 0).mean().alias("market_breadth"),
            pl.col("is_limit_down").filter(pl.col("cap_decile") == 0).mean().alias("bottom_limit_down_rate"),
            pl.col("is_limit_down").mean().alias("market_limit_down_rate"),
        )
        .sort("date")
        .with_columns(
            (pl.col("bottom_mark") - pl.col("market_mark")).alias("mark_excess"),
            (pl.col("bottom_net") - pl.col("market_net")).alias("net_excess"),
        )
    )


def _compound(values: list[float]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not valid:
        return None
    wealth = 1.0
    for value in valid:
        wealth *= 1.0 + value
    return wealth - 1.0


def _annualized(values: list[float]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(value)]
    total = _compound(valid)
    if total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (52.0 / len(valid)) - 1.0


def _max_drawdown(values: list[float]) -> float | None:
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    found = False
    for value in values:
        if value is None or not math.isfinite(value):
            continue
        found = True
        wealth *= 1.0 + float(value)
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    return drawdown if found else None


def period_metrics(weekly: pl.DataFrame) -> list[dict[str, Any]]:
    output = []
    for period in ("development", "validation", "known_stress"):
        scoped = weekly.filter(pl.col("period") == period).sort("date")
        if scoped.is_empty():
            continue
        bottom = scoped.get_column("bottom_net").to_list()
        market = scoped.get_column("market_net").to_list()
        bottom_mark = scoped.get_column("bottom_mark").to_list()
        market_mark = scoped.get_column("market_mark").to_list()
        bottom_annual = _annualized(bottom)
        market_annual = _annualized(market)
        bottom_mark_annual = _annualized(bottom_mark)
        market_mark_annual = _annualized(market_mark)
        positive_excess_years = 0
        yearly = []
        for year in sorted(scoped.get_column("date").dt.year().unique().to_list()):
            year_frame = scoped.filter(pl.col("date").dt.year() == year)
            year_bottom = _compound(year_frame.get_column("bottom_net").to_list())
            year_market = _compound(year_frame.get_column("market_net").to_list())
            excess = (
                year_bottom - year_market
                if year_bottom is not None and year_market is not None
                else None
            )
            positive_excess_years += int(excess is not None and excess > 0)
            yearly.append(
                {
                    "year": year,
                    "bottom_net_return": year_bottom,
                    "market_net_return": year_market,
                    "excess": excess,
                }
            )
        output.append(
            {
                "period": period,
                "weeks": scoped.height,
                "bottom_annualized_net": bottom_annual,
                "market_annualized_net": market_annual,
                "annualized_net_excess": (
                    bottom_annual - market_annual
                    if bottom_annual is not None and market_annual is not None
                    else None
                ),
                "bottom_annualized_mark": bottom_mark_annual,
                "market_annualized_mark": market_mark_annual,
                "annualized_mark_excess": (
                    bottom_mark_annual - market_mark_annual
                    if bottom_mark_annual is not None
                    and market_mark_annual is not None
                    else None
                ),
                "bottom_max_drawdown": _max_drawdown(bottom),
                "mean_bottom_tradable_rate": scoped.get_column(
                    "bottom_tradable_rate"
                ).mean(),
                "positive_excess_years": positive_excess_years,
                "yearly": yearly,
            }
        )
    return output


def disaster_weeks(weekly: pl.DataFrame) -> list[dict[str, Any]]:
    rows = weekly.sort("date").to_dicts()
    wealth_history: list[float] = []
    wealth = 1.0
    output = []
    prior_excess: list[float] = []
    for row in rows:
        bottom = row.get("bottom_mark")
        excess = row.get("mark_excess")
        if bottom is None or excess is None:
            continue
        wealth *= 1.0 + float(bottom)
        wealth_history.append(wealth)
        rolling_peak = max(wealth_history[-5:])
        drawdown_4w = wealth / rolling_peak - 1.0
        prior_4w_excess = sum(prior_excess[-4:]) if prior_excess else None
        if float(excess) <= -0.05 or drawdown_4w <= -0.15:
            output.append(
                {
                    "signal_date": row["date"],
                    "period": row["period"],
                    "bottom_mark": bottom,
                    "bottom_net": row.get("bottom_net"),
                    "market_mark": row.get("market_mark"),
                    "market_net": row.get("market_net"),
                    "mark_excess": excess,
                    "net_excess": row.get("net_excess"),
                    "mark_drawdown_4w": drawdown_4w,
                    "prior_4w_excess": prior_4w_excess,
                    "bottom_breadth": row.get("bottom_breadth"),
                    "market_breadth": row.get("market_breadth"),
                    "bottom_limit_down_rate": row.get("bottom_limit_down_rate"),
                    "market_limit_down_rate": row.get("market_limit_down_rate"),
                    "bottom_median_amount": row.get("bottom_median_amount"),
                    "market_median_amount": row.get("market_median_amount"),
                }
            )
        prior_excess.append(float(excess))
    return output


def evaluate_gate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_period = {row["period"]: row for row in metrics}
    validation = by_period.get("validation") or {}
    stress = by_period.get("known_stress") or {}
    passed = bool(
        (validation.get("annualized_net_excess") or -99.0) >= MIN_ANNUAL_EXCESS
        and (stress.get("annualized_net_excess") or -99.0) >= MIN_ANNUAL_EXCESS
        and (validation.get("annualized_mark_excess") or -99.0) >= MIN_ANNUAL_EXCESS
        and (stress.get("annualized_mark_excess") or -99.0) >= MIN_ANNUAL_EXCESS
        and (validation.get("mean_bottom_tradable_rate") or 0.0)
        >= MIN_TRADABLE_RATE
        and (stress.get("mean_bottom_tradable_rate") or 0.0)
        >= MIN_TRADABLE_RATE
        and (validation.get("positive_excess_years") or 0)
        >= MIN_POSITIVE_EXCESS_YEARS
        and (stress.get("positive_excess_years") or 0)
        >= MIN_POSITIVE_EXCESS_YEARS
    )
    return {
        "verdict": "CONTINUE" if passed else "DOWNGRADE",
        "passed": passed,
        "reason": (
            "微盘横截面基线同时通过验证期和已知压力期门槛"
            if passed
            else "微盘横截面基线没有同时通过验证期和已知压力期门槛"
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path, *, end: date | None = None) -> dict[str, Any]:
    source = load_daily(data_dir, end=end)
    if source.is_empty():
        raise ValueError("no daily data")
    source_rows = source.height
    source_symbols = source.get_column("symbol").n_unique()
    source_board_counts = board_symbol_counts(source)
    pit = attach_point_in_time_data(source, data_dir)
    pit_rows = pit.height
    pit_symbols = pit.get_column("symbol").n_unique()
    pit_board_counts = board_symbol_counts(pit)
    del source
    gc.collect()
    panel = prepare_panel(pit)
    del pit
    gc.collect()
    data_summary = {
        "source_rows": source_rows,
        "source_symbols": source_symbols,
        "source_board_counts": source_board_counts,
        "pit_share_rows": pit_rows,
        "pit_row_retention_rate": pit_rows / source_rows,
        "symbols": pit_symbols,
        "pit_board_counts": pit_board_counts,
        "first_date": panel.get_column("date").min(),
        "last_date": panel.get_column("date").max(),
    }
    observations = build_weekly_observations(panel)
    del panel
    gc.collect()
    weekly = weekly_portfolios(observations)
    deciles = (
        observations.group_by("date", "period", "cap_decile")
        .agg(
            pl.len().alias("event_count"),
            pl.col("tradable").mean().alias("tradable_rate"),
            pl.col("mark_return").mean().alias("mark_return"),
            pl.col("net_return").mean().alias("net_return"),
        )
        .sort("date", "cap_decile")
    )
    del observations
    gc.collect()
    metrics = period_metrics(weekly)
    disasters = disaster_weeks(weekly)
    decision = evaluate_gate(metrics)
    payload = {
        "schema_version": "p0-microcap-baseline-v1",
        "contract": {
            "development_end": DEVELOPMENT_END,
            "validation_end": VALIDATION_END,
            "end": end or data_summary["last_date"],
            "universe": "sh_sz_pit_non_st_180_calendar_days",
            "market_cap": "raw_close_times_pit_announced_total_shares",
            "rebalance": "weekly_signal_close_next_open_to_next_open",
            "position_notional": POSITION_NOTIONAL,
            "daily_participation": DAILY_PARTICIPATION,
            "minimum_annual_excess": MIN_ANNUAL_EXCESS,
            "minimum_tradable_rate": MIN_TRADABLE_RATE,
        },
        "data": data_summary,
        "weekly_rows": weekly.height,
        "period_metrics": metrics,
        "disaster_weeks": disasters,
        "disaster_week_count": len(disasters),
        "decile_period_summary": (
            deciles.group_by("period", "cap_decile")
            .agg(
                pl.len().alias("weeks"),
                pl.col("tradable_rate").mean().alias("mean_tradable_rate"),
                pl.col("mark_return").mean().alias("mean_weekly_mark"),
                pl.col("net_return").mean().alias("mean_weekly_net"),
            )
            .sort("period", "cap_decile")
            .to_dicts()
        ),
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "data": data_summary,
                "weekly_rows": weekly.height,
                "disaster_week_count": len(disasters),
                "decision": decision,
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
        default=Path("/app/data/research/p0_microcap_baseline.json"),
    )
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    run(args.data_dir, args.output, end=args.end)


if __name__ == "__main__":
    main()
