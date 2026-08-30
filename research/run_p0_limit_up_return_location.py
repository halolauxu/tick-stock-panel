"""Locate where the limit-up ecosystem's returns occur within the next session."""
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

RESEARCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH_ROOT))

import run_p0_emotion_limit_up_study as p0  # noqa: E402

MINUTE_START = date(2025, 8, 27)
LOCATION_END = date(2026, 4, 30)
MINUTE_END = date(2026, 8, 28)
DAILY_BUFFER_START = date(2025, 8, 1)

MIN_LOCATION_DAYS = 80
MIN_CONFIRMATION_DAYS = 30
MIN_TRADABLE_RATE = 0.70
MIN_CLUSTER_NET = 0.005
MIN_CLUSTER_T = 1.0
MIN_POSITIVE_MONTHS = 2
MAX_MONTH_POSITIVE_SHARE = 0.60

COMPONENTS = (
    "signal_close_to_open",
    "open_to_5m_vwap",
    "vwap_5m_to_close",
    "close_to_t1_sell_open",
    "open_to_t1_sell_open_net",
    "vwap_5m_to_t1_sell_open_net",
)


def location_period(value: pl.Expr) -> pl.Expr:
    return (
        pl.when(value <= pl.lit(LOCATION_END))
        .then(pl.lit("location"))
        .otherwise(pl.lit("confirmation"))
    )


def load_thresholds(data_dir: Path) -> dict[str, float]:
    path = data_dir / "research" / "p0_emotion_limit_up.json"
    if not path.is_file():
        raise ValueError("P0-A result is required before running P0-A2")
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("state_thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("P0-A state thresholds are missing")
    return {str(key): float(value) for key, value in thresholds.items()}


def minute_paths(data_dir: Path) -> list[Path]:
    paths = []
    for path in sorted((data_dir / "kline_minute").glob("date=*/part.parquet")):
        try:
            day = date.fromisoformat(path.parent.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if MINUTE_START <= day <= MINUTE_END:
            paths.append(path)
    return paths


def load_first_five_minutes(
    data_dir: Path,
    entry_keys: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = minute_paths(data_dir)
    if not paths:
        raise ValueError("no minute partitions in the frozen range")
    keys = (
        entry_keys.select(
            "symbol",
            pl.col("entry_date").alias("date"),
        )
        .drop_nulls()
        .unique()
    )
    first_five = (
        pl.scan_parquet(paths)
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .filter(
            (pl.col("datetime").dt.hour() == 9)
            & pl.col("datetime").dt.minute().is_between(30, 34)
        )
        .join(keys.lazy(), on=["symbol", "date"], how="inner")
        .sort(["symbol", "date", "datetime"])
        .group_by("symbol", "date", maintain_order=True)
        .agg(
            pl.col("open").first().alias("minute_open"),
            pl.col("close").last().alias("minute_last_5m"),
            pl.col("amount").sum().alias("amount_5m"),
            pl.col("volume").sum().alias("volume_5m"),
            pl.len().alias("minute_bars_5m"),
        )
        .with_columns(
            pl.when(pl.col("volume_5m") > 0)
            .then(pl.col("amount_5m") / (pl.col("volume_5m") * 100.0))
            .otherwise(None)
            .alias("vwap_5m")
        )
        .collect(engine="streaming")
    )
    return first_five, {
        "partition_count": len(paths),
        "first_partition": paths[0].parent.name.removeprefix("date="),
        "last_partition": paths[-1].parent.name.removeprefix("date="),
        "matched_symbol_days": first_five.height,
        "complete_five_bar_rate": (
            first_five.select((pl.col("minute_bars_5m") == 5).mean()).item()
            if first_five.height
            else None
        ),
    }


def attach_return_components(
    outcomes: pl.DataFrame,
    first_five: pl.DataFrame,
    *,
    position_notional: float = p0.POSITION_NOTIONAL,
    opening_participation: float = p0.OPENING_PARTICIPATION,
) -> pl.DataFrame:
    work = outcomes.join(
        first_five,
        left_on=["symbol", "entry_date"],
        right_on=["symbol", "date"],
        how="left",
    )
    adjustment = pl.col("entry_price") / pl.col("entry_raw_price")
    vwap_adjusted = pl.col("vwap_5m") * adjustment
    entry_close_adjusted = pl.col("entry_raw_close") * adjustment
    has_minutes = (
        (pl.col("minute_bars_5m") == 5)
        & pl.col("vwap_5m").is_not_null()
        & (pl.col("vwap_5m") > 0)
    )
    unsealed = pl.col("minute_last_5m") < pl.col("entry_limit_up_price") - 0.005
    opening_capacity = (
        pl.col("amount_5m").fill_null(0) * opening_participation
        >= position_notional
    )
    known_capacity = (
        pl.col("signal_day_amount").fill_null(0) * p0.DAILY_PARTICIPATION
        >= position_notional
    ) & (
        pl.col("entry_day_amount").fill_null(0) * p0.DAILY_PARTICIPATION
        >= position_notional
    )
    has_exit = pl.col("exit_price_h1").is_not_null()
    delayed_valid = has_minutes & unsealed.fill_null(False) & opening_capacity & known_capacity
    delayed_tradable = delayed_valid & has_exit
    sell_cost = (
        p0.COMMISSION_PCT
        + p0.SLIPPAGE_PCT
        + p0.historical_stamp_tax_expr(pl.col("exit_date_h1"))
    )
    return (
        work.with_columns(
            location_period(pl.col("entry_date")).alias("location_period"),
            delayed_valid.alias("delayed_entry_valid"),
            delayed_tradable.alias("delayed_tradable"),
            pl.when(~has_minutes)
            .then(pl.lit("missing_first_five_minutes"))
            .when(~unsealed.fill_null(False))
            .then(pl.lit("still_limit_up_at_09_34"))
            .when(~opening_capacity)
            .then(pl.lit("insufficient_first_five_capacity"))
            .when(~known_capacity)
            .then(pl.lit("insufficient_daily_capacity"))
            .when(~has_exit)
            .then(pl.lit("no_t1_sell_open"))
            .otherwise(pl.lit("ok"))
            .alias("delayed_entry_status"),
            vwap_adjusted.alias("vwap_5m_adjusted"),
            (pl.col("entry_price") / pl.col("close") - 1.0).alias(
                "signal_close_to_open"
            ),
            (vwap_adjusted / pl.col("entry_price") - 1.0).alias(
                "open_to_5m_vwap"
            ),
            (entry_close_adjusted / vwap_adjusted - 1.0).alias(
                "vwap_5m_to_close"
            ),
            (pl.col("exit_price_h1") / entry_close_adjusted - 1.0).alias(
                "close_to_t1_sell_open"
            ),
            pl.col("net_return_h1").alias("open_to_t1_sell_open_net"),
            (
                (
                    pl.col("exit_price_h1")
                    * (1.0 - sell_cost)
                )
                / (
                    vwap_adjusted
                    * (1.0 + p0.COMMISSION_PCT + p0.SLIPPAGE_PCT)
                )
                - 1.0
            ).alias("_delayed_net"),
        )
        .with_columns(
            pl.when(pl.col("delayed_tradable"))
            .then(pl.col("_delayed_net"))
            .otherwise(None)
            .alias("vwap_5m_to_t1_sell_open_net")
        )
        .drop("_delayed_net")
    )


def component_summary(frame: pl.DataFrame, *, include_state: bool) -> pl.DataFrame:
    keys = ["location_period", "event_type"]
    if include_state:
        keys.append("state")
    rows = []
    for metric in COMPONENTS:
        scoped = frame.select(
            *keys,
            "signal_date",
            pl.lit(metric).alias("component"),
            pl.col(metric).alias("value"),
        ).drop_nulls("value")
        event = scoped.group_by(keys).agg(
            pl.len().alias("event_count"),
            pl.col("signal_date").n_unique().alias("signal_day_count"),
            pl.col("value").mean().alias("event_mean"),
            pl.col("value").median().alias("event_median"),
            (pl.col("value") > 0).mean().alias("event_win_rate"),
            pl.col("value").quantile(0.05, "linear").alias("event_p05"),
        )
        daily = (
            scoped.group_by(*keys, "signal_date")
            .agg(pl.col("value").mean().alias("daily_cluster_return"))
            .group_by(keys)
            .agg(
                pl.len().alias("cluster_day_count"),
                pl.col("daily_cluster_return").mean().alias("cluster_mean"),
                pl.col("daily_cluster_return").std(ddof=1).alias("cluster_std"),
            )
            .with_columns(
                pl.when((pl.col("cluster_day_count") > 1) & (pl.col("cluster_std") > 0))
                .then(
                    pl.col("cluster_mean")
                    / (pl.col("cluster_std") / pl.col("cluster_day_count").sqrt())
                )
                .otherwise(None)
                .alias("cluster_t")
            )
        )
        rows.append(event.join(daily, on=keys, how="left").with_columns(pl.lit(metric).alias("component")))
    return pl.concat(rows, how="diagonal_relaxed").sort(*keys, "component")


def delayed_entry_summary(frame: pl.DataFrame) -> pl.DataFrame:
    keys = ["location_period", "event_type"]
    base = frame.group_by(keys).agg(
        pl.len().alias("event_count"),
        pl.col("delayed_tradable").sum().alias("tradable_count"),
    ).with_columns(
        (pl.col("tradable_count") / pl.col("event_count")).alias("tradable_rate")
    )
    daily = (
        frame.filter(pl.col("delayed_tradable"))
        .group_by(*keys, "signal_date")
        .agg(
            pl.col("vwap_5m_to_t1_sell_open_net")
            .mean()
            .alias("daily_cluster_return")
        )
    )
    statistics = daily.group_by(keys).agg(
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
    monthly = (
        daily.with_columns(pl.col("signal_date").dt.truncate("1mo").alias("month"))
        .group_by(*keys, "month")
        .agg(pl.col("daily_cluster_return").sum().alias("month_return_sum"))
        .group_by(keys)
        .agg(
            (pl.col("month_return_sum") > 0).sum().alias("positive_month_count"),
            pl.when(pl.col("month_return_sum").clip(lower_bound=0).sum() > 0)
            .then(
                pl.col("month_return_sum").clip(lower_bound=0).max()
                / pl.col("month_return_sum").clip(lower_bound=0).sum()
            )
            .otherwise(None)
            .alias("top_month_positive_share"),
        )
    )
    return (
        base.join(statistics, on=keys, how="left")
        .join(monthly, on=keys, how="left")
        .sort(keys)
    )


def evaluate_candidates(summary: pl.DataFrame) -> list[dict[str, Any]]:
    candidates = []
    for event_type in sorted(summary.get_column("event_type").unique().to_list()):
        periods = {
            row["location_period"]: row
            for row in summary.filter(pl.col("event_type") == event_type).to_dicts()
        }
        location = periods.get("location") or {}
        confirmation = periods.get("confirmation") or {}
        passed = bool(
            (location.get("cluster_mean_net") or -99.0) >= MIN_CLUSTER_NET
            and (confirmation.get("cluster_mean_net") or -99.0) >= MIN_CLUSTER_NET
            and (location.get("cluster_day_count") or 0) >= MIN_LOCATION_DAYS
            and (confirmation.get("cluster_day_count") or 0) >= MIN_CONFIRMATION_DAYS
            and (location.get("tradable_rate") or 0.0) >= MIN_TRADABLE_RATE
            and (confirmation.get("tradable_rate") or 0.0) >= MIN_TRADABLE_RATE
            and (location.get("cluster_t") or -99.0) > MIN_CLUSTER_T
            and (confirmation.get("cluster_t") or -99.0) > MIN_CLUSTER_T
            and (location.get("positive_month_count") or 0) >= MIN_POSITIVE_MONTHS
            and (confirmation.get("positive_month_count") or 0) >= MIN_POSITIVE_MONTHS
            and (location.get("top_month_positive_share") or 99.0)
            <= MAX_MONTH_POSITIVE_SHARE
            and (confirmation.get("top_month_positive_share") or 99.0)
            <= MAX_MONTH_POSITIVE_SHARE
        )
        candidates.append(
            {"event_type": event_type, "passed": passed, "periods": periods}
        )
    return sorted(candidates, key=lambda row: (not row["passed"], row["event_type"]))


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    thresholds = load_thresholds(data_dir)
    source = p0.load_daily_panel(
        data_dir,
        end=MINUTE_END,
        start=DAILY_BUFFER_START,
    )
    pit = p0.attach_point_in_time_universe(source, data_dir)
    del source
    panel = p0.prepare_market_panel(pit)
    del pit
    daily_inputs = p0.build_daily_state_inputs(panel)
    daily = p0.classify_states(daily_inputs, thresholds)
    del daily_inputs
    events = p0.build_events(panel, daily).filter(
        pl.col("signal_date") >= pl.lit(MINUTE_START)
    )
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
    outcomes = p0.attach_event_outcomes(events, execution_panel)
    del events, execution_panel
    gc.collect()
    first_five, minute_metadata = load_first_five_minutes(
        data_dir,
        outcomes.select("symbol", "entry_date"),
    )
    frame = attach_return_components(outcomes, first_five).filter(
        pl.col("entry_date").is_between(MINUTE_START, MINUTE_END)
    )
    del outcomes, first_five
    primary_summary = delayed_entry_summary(frame)
    candidates = evaluate_candidates(primary_summary)
    passed = [candidate for candidate in candidates if candidate["passed"]]
    component = component_summary(frame, include_state=False)
    state_diagnostic = component_summary(frame, include_state=True)
    payload = {
        "schema_version": "p0-limit-up-return-location-v1",
        "contract": {
            "minute_start": MINUTE_START,
            "location_end": LOCATION_END,
            "minute_end": MINUTE_END,
            "primary_entry": "09:30_to_09:34_vwap",
            "primary_exit": "first_valid_open_after_t_plus_1",
            "minimum_cluster_net": MIN_CLUSTER_NET,
            "minimum_location_days": MIN_LOCATION_DAYS,
            "minimum_confirmation_days": MIN_CONFIRMATION_DAYS,
            "minimum_tradable_rate": MIN_TRADABLE_RATE,
        },
        "data": {
            "event_rows": frame.height,
            "symbols": frame.get_column("symbol").n_unique(),
            "signal_days": frame.get_column("signal_date").n_unique(),
            "first_signal_date": frame.get_column("signal_date").min(),
            "last_signal_date": frame.get_column("signal_date").max(),
        },
        "minute_metadata": minute_metadata,
        "minute_vs_daily_open": {
            "matched_rows": frame.filter(pl.col("minute_open").is_not_null()).height,
            "median_difference": frame.select(
                (pl.col("minute_open") / pl.col("entry_raw_price") - 1.0).median()
            ).item(),
            "p99_absolute_difference": frame.select(
                (pl.col("minute_open") / pl.col("entry_raw_price") - 1.0)
                .abs()
                .quantile(0.99, "linear")
            ).item(),
        },
        "delayed_entry_status": frame.group_by(
            "location_period", "delayed_entry_status"
        ).len().sort("location_period", "delayed_entry_status").to_dicts(),
        "component_summary": component.to_dicts(),
        "state_diagnostic": state_diagnostic.to_dicts(),
        "primary_summary": primary_summary.to_dicts(),
        "candidate_gates": candidates,
        "decision": {
            "verdict": "CONTINUE" if passed else "DOWNGRADE",
            "passed_event_types": [row["event_type"] for row in passed],
            "reason": (
                "至少一个事件类型的延迟入场路径通过冻结门槛"
                if passed
                else "没有事件类型的延迟入场路径同时通过定位期和时间确认期"
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
                "minute_metadata": minute_metadata,
                "decision": payload["decision"],
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
        default=Path("/app/data/research/p0_limit_up_return_location.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
