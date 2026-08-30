"""Run the frozen development-only major-contract underreaction study."""

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

START = date(2015, 12, 1)
DEVELOPMENT_START = date(2016, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
COOLDOWN_DAYS = 90

CATEGORY_HOLDING_DAYS = {
    "transformational_contract": 20,
    "material_contract": 10,
    "low_impact_control": 10,
}
PROMOTABLE_CATEGORIES = tuple(CATEGORY_HOLDING_DAYS)[:2]
ALLOWED_CONTRACT_TYPES = {
    "项目中标",
    "工程建设",
    "项目/产品/技术开发",
    "销售合同",
    "服务/劳务合同",
}
EXCLUDED_TEXT = re.compile(r"框架|意向|备忘录|战略合作|补充协议|终止|解除|废止|取消")
ABOLISHED_VALUES = {"1", "true", "yes", "是", "已废止", "废止"}


def load_contracts(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "major_contract").glob(
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
        raise ValueError("all 2016-2020 major-contract partitions are required")
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["ann_date", "symbol", "event_id"])
    )


def categorize_events(contracts: pl.DataFrame) -> pl.DataFrame:
    text = (
        pl.concat_str(
            ["contract_name", "contents", "stated_effect"],
            separator=" ",
        )
        .fill_null("")
        .alias("_text")
    )
    categorized = (
        contracts.with_columns(text)
        .filter(
            pl.col("contract_type_name").is_in(ALLOWED_CONTRACT_TYPES)
            & pl.col("revenue_ratio_pct").is_not_null()
            & ~pl.col("_text").str.contains(EXCLUDED_TEXT.pattern)
            & ~pl.col("is_abolished").str.to_lowercase().is_in(ABOLISHED_VALUES)
        )
        .with_columns(
            pl.when(pl.col("revenue_ratio_pct") >= 50.0)
            .then(pl.lit("transformational_contract"))
            .when(pl.col("revenue_ratio_pct") >= 20.0)
            .then(pl.lit("material_contract"))
            .when(pl.col("revenue_ratio_pct").is_between(5.0, 10.0, closed="both"))
            .then(pl.lit("low_impact_control"))
            .otherwise(None)
            .alias("category")
        )
        .filter(pl.col("category").is_not_null())
        .sort(["symbol", "category", "ann_date", "event_id"])
        .unique(
            subset=["symbol", "category", "ann_date"], keep="first", maintain_order=True
        )
    )
    last_kept: dict[tuple[str, str], date] = {}
    keep: list[bool] = []
    for row in categorized.iter_rows(named=True):
        key = (row["symbol"], row["category"])
        event_date = row["ann_date"]
        previous = last_kept.get(key)
        accepted = previous is None or (event_date - previous).days >= COOLDOWN_DAYS
        keep.append(accepted)
        if accepted:
            last_kept[key] = event_date
    return (
        categorized.filter(pl.Series("_keep", keep, dtype=pl.Boolean))
        .with_columns(
            pl.col("category")
            .replace_strict(CATEGORY_HOLDING_DAYS, default=None)
            .cast(pl.Int64)
            .alias("holding_trading_days")
        )
        .drop("_text")
        .sort(["ann_date", "symbol", "category"])
    )


def build_all_trades(events: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for category, holding_days in CATEGORY_HOLDING_DAYS.items():
        scoped = events.filter(pl.col("category") == category)
        if scoped.is_empty():
            continue
        trades = build_trades(scoped, panel, holding_days)
        benchmark = build_market_benchmark(panel, holding_days)
        parts.append(attach_market_excess(trades, benchmark))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def summarize_category(trades: pl.DataFrame, category: str) -> dict[str, Any]:
    scoped = trades.filter(pl.col("category") == category)
    eligible = scoped.filter(pl.col("universe_eligible"))
    tradable = scoped.filter(pl.col("tradable"))
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
    return {
        "events": scoped.height,
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


def apply_promotion_gates(summaries: dict[str, dict[str, Any]]) -> None:
    control = summaries["low_impact_control"].get("mean_excess_return")
    for category in CATEGORY_HOLDING_DAYS:
        result = summaries[category]
        result["excess_vs_low_impact_control"] = (
            result["mean_excess_return"] - control
            if result.get("mean_excess_return") is not None and control is not None
            else None
        )
        min_events = 150 if category == "transformational_contract" else 300
        min_net = 0.02 if category == "transformational_contract" else 0.0125
        min_excess = 0.015 if category == "transformational_contract" else 0.0075
        min_control_spread = 0.01 if category == "transformational_contract" else 0.005
        result["promotion_passed"] = bool(
            category in PROMOTABLE_CATEGORIES
            and result["tradable_events"] >= min_events
            and result["announcement_days"] >= 100
            and result["tradable_rate"] >= 0.90
            and result["benchmark_coverage"] >= 0.99
            and result["entry_capacity_feasible_rate"] >= 0.95
            and result["unresolved_exits"] == 0
            and (result["mean_net_return"] or -math.inf) >= min_net
            and (result["mean_excess_return"] or -math.inf) >= min_excess
            and (result["excess_daily_cluster_t"] or -math.inf) >= 2.5
            and result["positive_excess_years"] >= 4
            and (result["max_year_positive_excess_share"] or math.inf) <= 0.50
            and (result["excess_vs_low_impact_control"] or -math.inf)
            >= min_control_spread
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw = load_contracts(data_dir)
    events = categorize_events(raw)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    trades = build_all_trades(events, panel)
    summaries = {
        category: summarize_category(trades, category)
        for category in CATEGORY_HOLDING_DAYS
    }
    apply_promotion_gates(summaries)
    promoted = [
        category
        for category in PROMOTABLE_CATEGORIES
        if summaries[category]["promotion_passed"]
    ]
    selected = (
        max(
            promoted,
            key=lambda category: summaries[category]["excess_daily_cluster_t"],
        )
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-major-contract-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "category_holding_trading_days": CATEGORY_HOLDING_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "allowed_contract_types": sorted(ALLOWED_CONTRACT_TYPES),
            "transformational_ratio_floor_pct": 50.0,
            "material_ratio_floor_pct": 20.0,
            "low_impact_control_band_pct": [5.0, 10.0],
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "decision_clock": "announcement date treated as after-close; next trading open entry",
            "provider_return_fields_used": False,
        },
        "data": {
            "raw_rows": raw.height,
            "ratio_rows": raw.filter(pl.col("revenue_ratio_pct").is_not_null()).height,
            "categorized_events": events.height,
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "categories": summaries,
        "decision": {
            "promoted_categories": promoted,
            "selected_candidate": selected,
            "family_candidate_cap": 1,
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_selected_candidate_before_validation"
                if selected
                else "terminate_major_contract_mechanism"
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
        default=Path("/app/data/research/p0_major_contract_development.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
