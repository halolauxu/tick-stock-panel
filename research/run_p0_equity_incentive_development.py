"""Run the frozen development-only equity-incentive announcement study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    _cluster_t,
    attach_market_excess,
    build_market_benchmark,
)

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 20
COOLDOWN_DAYS = 365
CATEGORY = "initial_equity_incentive"

_EXCLUDED = re.compile(
    r"摘要|修订|修正版|修订稿|更正|调整|补充|更新|终止|取消|废止|"
    r"自查|核查意见|法律意见|财务顾问|考核|激励对象|授予|回购注销|"
    r"实施|进展|完成|子公司|员工股权"
)


def classify_title(title: str) -> tuple[str, str] | None:
    text = re.sub(r"\s+", "", str(title or ""))
    if "激励计划" not in text or "草案" not in text or _EXCLUDED.search(text):
        return None
    if "股票期权" in text and "限制性股票" in text:
        instrument = "mixed_option_restricted"
    elif "股票期权" in text:
        instrument = "stock_option"
    elif "限制性股票" in text:
        instrument = "restricted_stock"
    else:
        instrument = "other_equity_incentive"
    return CATEGORY, instrument


def load_announcements(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "equity_incentive").glob(
        "year=*/part.parquet"
    ):
        try:
            year = int(path.parent.name.removeprefix("year="))
        except ValueError:
            continue
        if DEVELOPMENT_START.year <= year <= DEVELOPMENT_END.year:
            paths.append(path)
    expected = DEVELOPMENT_END.year - DEVELOPMENT_START.year + 1
    if len(paths) != expected:
        raise ValueError("all 2014-2020 equity-incentive partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["ann_date", "symbol", "announcement_id"])
    )


def categorize_events(announcements: pl.DataFrame) -> pl.DataFrame:
    classified = [classify_title(title) for title in announcements["title"].to_list()]
    categories = [row[0] if row else None for row in classified]
    instruments = [row[1] if row else None for row in classified]
    work = (
        announcements.with_columns(
            pl.Series("category", categories, dtype=pl.Utf8),
            pl.Series("instrument", instruments, dtype=pl.Utf8),
        )
        .filter(pl.col("category").is_not_null())
        .sort(["symbol", "ann_date", "announcement_id"])
        .unique(subset=["symbol", "ann_date"], keep="first", maintain_order=True)
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
    return work.filter(pl.Series("_keep", keep, dtype=pl.Boolean)).sort(
        ["ann_date", "symbol"]
    )


def summarize(trades: pl.DataFrame) -> dict[str, Any]:
    eligible = trades.filter(pl.col("universe_eligible"))
    tradable = trades.filter(pl.col("tradable"))
    benchmarked = tradable.filter(pl.col("excess_return").is_not_null())
    capacity_base = eligible.filter(
        pl.col("entry_date").is_not_null()
        & (pl.col("entry_amount").fill_null(0) > 0)
        & (pl.col("entry_open").fill_null(0) > 0)
    )
    capacity_feasible = capacity_base.filter(
        pl.col("entry_amount").fill_null(0) * DAILY_PARTICIPATION >= POSITION_NOTIONAL
    )
    yearly = (
        benchmarked.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("net_return").mean().alias("mean_net_return"),
            pl.col("excess_return").mean().alias("mean_excess_return"),
            pl.col("excess_return").sum().alias("sum_excess_return"),
        )
        .sort("year")
    )
    positive_sums = [
        float(value)
        for value in yearly.get_column("sum_excess_return").to_list()
        if value is not None and value > 0
    ]
    result = {
        "events": trades.height,
        "universe_eligible_events": eligible.height,
        "tradable_events": tradable.height,
        "benchmarked_events": benchmarked.height,
        "announcement_days": benchmarked.get_column("ann_date").n_unique()
        if benchmarked.height
        else 0,
        "tradable_rate": tradable.height / eligible.height if eligible.height else 0.0,
        "benchmark_coverage": benchmarked.height / tradable.height
        if tradable.height
        else 0.0,
        "entry_capacity_feasible_rate": capacity_feasible.height / capacity_base.height
        if capacity_base.height
        else 0.0,
        "unresolved_exits": eligible.filter(
            pl.col("entry_valid") & pl.col("exit_delay").is_null()
        ).height,
        "mean_net_return": benchmarked.get_column("net_return").mean()
        if benchmarked.height
        else None,
        "mean_excess_return": benchmarked.get_column("excess_return").mean()
        if benchmarked.height
        else None,
        "median_excess_return": benchmarked.get_column("excess_return").median()
        if benchmarked.height
        else None,
        "excess_daily_cluster_t": _cluster_t(benchmarked, "excess_return"),
        "positive_excess_years": yearly.filter(pl.col("mean_excess_return") > 0).height,
        "max_year_positive_excess_share": max(positive_sums) / sum(positive_sums)
        if positive_sums
        else None,
        "yearly": yearly.to_dicts(),
    }
    result["promotion_passed"] = bool(
        result["tradable_events"] >= 500
        and result["announcement_days"] >= 300
        and result["tradable_rate"] >= 0.90
        and result["benchmark_coverage"] >= 0.99
        and result["entry_capacity_feasible_rate"] >= 0.95
        and result["unresolved_exits"] == 0
        and (result["mean_net_return"] or -math.inf) >= 0.03
        and (result["mean_excess_return"] or -math.inf) >= 0.02
        and (result["excess_daily_cluster_t"] or -math.inf) >= 3.0
        and result["positive_excess_years"] >= 5
        and (result["max_year_positive_excess_share"] or math.inf) <= 0.40
    )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw = load_announcements(data_dir)
    events = categorize_events(raw)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    trades = build_trades(events, panel, HOLD_TRADING_DAYS)
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    summary = summarize(trades)
    instrument_counts = (
        events.group_by("instrument")
        .agg(pl.len().alias("events"), pl.col("symbol").n_unique().alias("symbols"))
        .sort("instrument")
        .to_dicts()
    )
    payload = {
        "schema_version": "p0-equity-incentive-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "holding_trading_days": HOLD_TRADING_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "decision_clock": "announcement date treated as after-close; next trading open entry",
            "family_candidate_cap": 1,
        },
        "data": {
            "raw_search_rows": raw.height,
            "initial_plan_events": events.height,
            "instrument_counts": instrument_counts,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "result": summary,
        "decision": {
            "promoted": summary["promotion_passed"],
            "selected_candidate": CATEGORY if summary["promotion_passed"] else None,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_candidate_before_validation"
                if summary["promotion_passed"]
                else "terminate_equity_incentive_mechanism"
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
        default=Path("/app/data/research/p0_equity_incentive_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
