"""Run the frozen development-only earnings-forecast drift event study."""
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
sys.path.insert(0, str(ROOT / "backend"))

from app.price_limits import (  # noqa: E402
    polars_is_risk_warning_name,
    polars_limit_price,
    polars_price_limit_pct,
)

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 2, 28)
STAMP_TAX_CUT = date(2023, 8, 28)

COMMISSION_PCT = 0.0002
SLIPPAGE_PCT = 0.0005
STAMP_TAX_OLD = 0.001
STAMP_TAX_CURRENT = 0.0005
POSITION_NOTIONAL = 20_000.0
DAILY_PARTICIPATION = 0.01
MIN_LISTING_DAYS = 180
HOLD_TRADING_DAYS = 10
MAX_EXIT_DELAY = 5

CATEGORIES = (
    "turnaround",
    "growth_ge_100",
    "growth_50_100",
    "growth_0_50",
    "negative_control",
)
POSITIVE_CATEGORIES = CATEGORIES[:-1]


def historical_stamp_tax(value: pl.Expr) -> pl.Expr:
    return (
        pl.when(value < pl.lit(STAMP_TAX_CUT))
        .then(pl.lit(STAMP_TAX_OLD))
        .otherwise(pl.lit(STAMP_TAX_CURRENT))
    )


def load_forecasts(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "forecast").glob(
        "year=*/part.parquet"
    ):
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError:
            continue
        if DEVELOPMENT_START.year <= year <= DEVELOPMENT_END.year:
            paths.append(path)
    if len(paths) != DEVELOPMENT_END.year - DEVELOPMENT_START.year + 1:
        raise ValueError("all 2014-2020 forecast yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["ann_date", "symbol", "period_end"])
    )


def categorize_events(events: pl.DataFrame) -> pl.DataFrame:
    event_type = pl.col("type").fill_null("")
    is_first = pl.col("first_ann_date").is_null() | (
        pl.col("first_ann_date") == pl.col("ann_date")
    )
    positive_profit = pl.col("net_profit_min").fill_null(0) > 0
    turnaround = event_type.str.contains("扭亏", literal=True) & positive_profit
    negative = (
        (pl.col("p_change_max").is_not_null() & (pl.col("p_change_max") < 0))
        | event_type.str.contains(r"(?:预减|首亏|续亏|略减)")
    )
    category = (
        pl.when(turnaround)
        .then(pl.lit("turnaround"))
        .when(
            positive_profit
            & pl.col("p_change_min").is_not_null()
            & (pl.col("p_change_min") >= 100)
        )
        .then(pl.lit("growth_ge_100"))
        .when(
            positive_profit
            & pl.col("p_change_min").is_between(50, 100, closed="left")
        )
        .then(pl.lit("growth_50_100"))
        .when(
            positive_profit
            & pl.col("p_change_min").is_between(0, 50, closed="left")
        )
        .then(pl.lit("growth_0_50"))
        .when(negative)
        .then(pl.lit("negative_control"))
        .otherwise(None)
    )
    priority = (
        pl.when(turnaround)
        .then(4)
        .when(negative)
        .then(0)
        .otherwise(1)
    )
    return (
        events.filter(is_first)
        .with_columns(category.alias("category"), priority.alias("_priority"))
        .filter(pl.col("category").is_not_null())
        .sort(
            ["symbol", "ann_date", "_priority", "p_change_min", "period_end"],
            descending=[False, False, True, True, True],
            nulls_last=True,
        )
        .unique(subset=["symbol", "ann_date"], keep="first", maintain_order=True)
        .drop("_priority")
        .sort(["ann_date", "symbol"])
    )


def load_panel(
    data_dir: Path,
    start: date = START,
    panel_end: date = PANEL_END,
) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    panel = (
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
            pl.col("date").is_between(start, panel_end, closed="both")
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
        )
        .collect(engine="streaming")
    )
    return attach_point_in_time_universe(panel, data_dir)


def attach_point_in_time_universe(
    panel: pl.DataFrame, data_dir: Path
) -> pl.DataFrame:
    research = data_dir / "research"
    universe_path = research / "historical_stock_universe_all_a.parquet"
    names_path = research / "historical_stock_names_all_a.parquet"
    if not universe_path.is_file() or not names_path.is_file():
        raise ValueError("all-A point-in-time security master is required")
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
        panel.join(universe, on="symbol", how="left")
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
            & (pl.col("end_date").is_null() | (pl.col("date") <= pl.col("end_date")))
        )
        .with_columns(
            (
                polars_is_risk_warning_name(pl.col("name"))
                | pl.col("name").str.contains("退", literal=True)
            ).alias("excluded_name")
        )
        .drop("delist_date", "start_date", "end_date")
    )


def prepare_panel(panel: pl.DataFrame) -> pl.DataFrame:
    dates = panel.select("date").unique().sort("date").with_row_index("trade_index")
    work = panel.join(dates, on="date", how="left").sort(["symbol", "date"])
    risk = polars_is_risk_warning_name(pl.col("name"))
    work = work.with_columns(
        (pl.col("close") / pl.col("raw_close")).alias("_adj_factor"),
        pl.col("trade_index").shift(1).over("symbol").alias("_previous_index"),
        pl.col("close").shift(1).over("symbol").alias("_previous_close"),
        pl.col("raw_close").shift(1).over("symbol").alias("_previous_raw_close"),
        (pl.col("close").shift(1).over("symbol") / pl.col("raw_close").shift(1).over("symbol")).alias("_previous_factor"),
        polars_price_limit_pct(pl.col("symbol"), pl.col("date"), risk).alias(
            "_limit_pct"
        ),
    ).with_columns(
        (pl.col("trade_index") == pl.col("_previous_index") + 1).alias("_adjacent"),
        (pl.col("open") / pl.col("_adj_factor")).alias("raw_open"),
    )
    factor_changed = (pl.col("_adj_factor") - pl.col("_previous_factor")).abs() > 1e-6
    work = work.with_columns(
        pl.when(pl.col("_adjacent"))
        .then(
            pl.when(factor_changed)
            .then(pl.col("_previous_close"))
            .otherwise(pl.col("_previous_raw_close"))
        )
        .otherwise(None)
        .alias("reference_close")
    ).with_columns(
        polars_limit_price(pl.col("reference_close"), pl.col("_limit_pct"), up=True).alias("limit_up_price"),
        polars_limit_price(pl.col("reference_close"), pl.col("_limit_pct"), up=False).alias("limit_down_price"),
    )
    return work.select(
        "symbol",
        "date",
        "trade_index",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "open",
        "amount",
        "volume",
        "excluded_name",
        "limit_up_price",
        "limit_down_price",
    )


def map_entry_indices(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    holding_trading_days: int = HOLD_TRADING_DAYS,
) -> pl.DataFrame:
    calendar = panel.select("date", "trade_index").unique().sort("date")
    return (
        events.with_columns(
            (pl.col("ann_date") + pl.duration(days=1)).alias("entry_search_date")
        )
        .sort("entry_search_date")
        .join_asof(
            calendar,
            left_on="entry_search_date",
            right_on="date",
            strategy="forward",
        )
        .rename({"date": "mapped_entry_date"})
        .with_columns(
            (pl.col("trade_index") - 1).alias("prior_index"),
            (pl.col("trade_index") + holding_trading_days).alias(
                "planned_exit_index"
            ),
        )
        .sort(["ann_date", "symbol"])
    )


def _lookup(panel: pl.DataFrame, index_column: str, prefix: str) -> pl.DataFrame:
    return panel.select(
        "symbol",
        pl.col("trade_index").alias(index_column),
        *[
            pl.col(column).alias(f"{prefix}_{column}")
            for column in (
                "date",
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "open",
                "amount",
                "volume",
                "excluded_name",
                "limit_up_price",
                "limit_down_price",
            )
        ],
    )


def build_trades(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    holding_trading_days: int = HOLD_TRADING_DAYS,
) -> pl.DataFrame:
    work = map_entry_indices(events, panel, holding_trading_days)
    work = work.join(
        _lookup(panel, "prior_index", "prior"),
        on=["symbol", "prior_index"],
        how="left",
    ).join(
        _lookup(panel, "trade_index", "entry"),
        on=["symbol", "trade_index"],
        how="left",
    )
    for delay in range(MAX_EXIT_DELAY + 1):
        index_name = f"exit_index_{delay}"
        work = work.with_columns(
            (pl.col("planned_exit_index") + delay).alias(index_name)
        ).join(
            _lookup(panel, index_name, f"exit_{delay}"),
            on=["symbol", index_name],
            how="left",
        )

    universe_eligible = (
        pl.col("prior_date").is_not_null()
        & ~pl.col("prior_excluded_name").fill_null(True)
        & pl.col("prior_raw_close").is_between(3.0, 300.0, closed="both")
        & (pl.col("prior_amount") >= 20_000_000.0)
    )
    entry_sealed = (
        (pl.col("entry_raw_open") >= pl.col("entry_limit_up_price") - 0.005)
        & (pl.col("entry_raw_low") >= pl.col("entry_limit_up_price") - 0.005)
    ).fill_null(True)
    entry_valid = (
        universe_eligible
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
            (
                pl.col(f"{prefix}_raw_open")
                <= pl.col(f"{prefix}_limit_down_price") + 0.005
            )
            & (
                pl.col(f"{prefix}_raw_high")
                <= pl.col(f"{prefix}_limit_down_price") + 0.005
            )
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
        [pl.when(condition).then(pl.lit(delay)) for delay, condition in enumerate(sellable)]
    )
    work = work.with_columns(
        universe_eligible.alias("universe_eligible"),
        entry_valid.alias("entry_valid"),
        selected_delay.alias("exit_delay"),
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
            (pl.col("entry_valid") & pl.col("exit_delay").is_not_null()).alias(
                "tradable"
            ),
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
                / (
                    pl.col("entry_open")
                    * (1.0 + COMMISSION_PCT + SLIPPAGE_PCT)
                )
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


def summarize_category(trades: pl.DataFrame, category: str) -> dict[str, Any]:
    scoped = trades.filter(pl.col("category") == category)
    eligible = scoped.filter(pl.col("universe_eligible"))
    tradable = scoped.filter(pl.col("tradable"))
    daily = (
        tradable.group_by("ann_date")
        .agg(pl.col("net_return").mean().alias("daily_return"))
        .sort("ann_date")
    )
    daily_mean = daily.get_column("daily_return").mean() if daily.height else None
    daily_std = daily.get_column("daily_return").std(ddof=1) if daily.height > 1 else None
    daily_t = (
        daily_mean / (daily_std / math.sqrt(daily.height))
        if daily_mean is not None and daily_std not in (None, 0.0)
        else None
    )
    yearly = (
        tradable.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("net_return").mean().alias("mean_return"),
            pl.col("net_return").sum().alias("sum_return"),
        )
        .sort("year")
    )
    positive_years = yearly.filter(pl.col("mean_return") > 0).height
    positive_sums = [
        float(value)
        for value in yearly.get_column("sum_return").to_list()
        if value is not None and value > 0
    ]
    max_year_positive_share = (
        max(positive_sums) / sum(positive_sums) if positive_sums else None
    )
    unresolved = eligible.filter(
        pl.col("entry_valid") & pl.col("exit_delay").is_null()
    ).height
    tradable_rate = tradable.height / eligible.height if eligible.height else 0.0
    result = {
        "events": scoped.height,
        "universe_eligible_events": eligible.height,
        "tradable_events": tradable.height,
        "announcement_days": tradable.get_column("ann_date").n_unique()
        if tradable.height
        else 0,
        "tradable_rate": tradable_rate,
        "unresolved_exits": unresolved,
        "mean_net_return": tradable.get_column("net_return").mean()
        if tradable.height
        else None,
        "median_net_return": tradable.get_column("net_return").median()
        if tradable.height
        else None,
        "win_rate": tradable.filter(pl.col("net_return") > 0).height
        / tradable.height
        if tradable.height
        else None,
        "daily_cluster_t": daily_t,
        "positive_years": positive_years,
        "max_year_positive_share": max_year_positive_share,
        "yearly": yearly.to_dicts(),
    }
    result["promotion_passed"] = bool(
        category in POSITIVE_CATEGORIES
        and result["tradable_events"] >= 500
        and result["announcement_days"] >= 200
        and result["tradable_rate"] >= 0.90
        and result["unresolved_exits"] == 0
        and (result["mean_net_return"] or -math.inf) >= 0.005
        and (result["daily_cluster_t"] or -math.inf) >= 2.5
        and result["positive_years"] >= 5
        and (result["max_year_positive_share"] or math.inf) <= 0.50
    )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_events = load_forecasts(data_dir)
    events = categorize_events(raw_events)
    panel = prepare_panel(load_panel(data_dir))
    trades = build_trades(events, panel)
    summaries = {
        category: summarize_category(trades, category) for category in CATEGORIES
    }
    promoted = [
        category
        for category in POSITIVE_CATEGORIES
        if summaries[category]["promotion_passed"]
    ]
    selected = (
        max(promoted, key=lambda category: summaries[category]["daily_cluster_t"])
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-forecast-drift-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "max_exit_delay": MAX_EXIT_DELAY,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "commission_pct_each_side": COMMISSION_PCT,
            "slippage_pct_each_side": SLIPPAGE_PCT,
        },
        "data": {
            "raw_forecast_rows": raw_events.height,
            "categorized_unique_events": events.height,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "categories": summaries,
        "decision": {
            "promoted_categories": promoted,
            "selected_candidate": selected,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_selected_candidate_before_validation"
                if selected
                else "terminate_forecast_drift_mechanism"
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
        default=Path("/app/data/research/p0_forecast_drift_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
