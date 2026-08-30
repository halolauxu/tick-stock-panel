"""Run the frozen development-only restructuring announcement event study."""

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
COOLDOWN_DAYS = 365

CATEGORY_HOLDING_DAYS = {
    "initial_major_restructuring": 5,
    "control_transfer": 10,
    "formal_tender_offer": 20,
    "termination_control": 5,
}
PROMOTABLE_CATEGORIES = tuple(CATEGORY_HOLDING_DAYS)[:-1]

_TERMINATION = re.compile(
    r"(?:终止|取消|停止|不再).{0,12}(?:重大资产重组|发行股份购买资产)"
    r"|(?:重大资产重组|发行股份购买资产).{0,12}(?:终止|取消|停止|不再)"
)
_INITIAL_EXCLUSIONS = re.compile(
    r"进展|继续|终止|取消|拟终止|提示性|摘要|修订|回复|说明|补充|更正"
)
_CONTROL_EXCLUSIONS = re.compile(
    r"进展|完成|过户|解除|终止|取消|质押|减持|增持|权益变动报告书"
)
_CONTROL_EVENT = re.compile(
    r"控制权.{0,12}(?:转让|变更|拟发生|将发生)"
    r"|(?:转让|变更).{0,12}控制权"
    r"|(?:实际控制人|控股股东).{0,8}(?:发生|拟发生|将发生)?变更"
)
_TENDER_EXCLUSIONS = re.compile(r"摘要|修订|提示|进展|完成|结果|豁免|变更")


def classify_title(title: str) -> str | None:
    text = re.sub(r"\s+", "", str(title or ""))
    if not text:
        return None
    if _TERMINATION.search(text):
        return "termination_control"
    if "要约收购报告书" in text and not _TENDER_EXCLUSIONS.search(text):
        return "formal_tender_offer"
    if _CONTROL_EVENT.search(text) and not _CONTROL_EXCLUSIONS.search(text):
        return "control_transfer"
    initial = (
        "筹划重大资产重组" in text
        or bool(re.search(r"重大资产重组.{0,8}停牌公告", text))
        or bool(re.search(r"发行股份购买资产.{0,20}预案", text))
    )
    if initial and not _INITIAL_EXCLUSIONS.search(text):
        return "initial_major_restructuring"
    return None


def load_announcements(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "restructuring_announcements").glob(
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
        raise ValueError(
            "all 2014-2020 restructuring announcement partitions are required"
        )
    return (
        pl.read_parquet(sorted(paths))
        .filter(
            pl.col("ann_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["ann_date", "symbol", "art_code"])
    )


def categorize_events(announcements: pl.DataFrame) -> pl.DataFrame:
    categories = [classify_title(title) for title in announcements["title"].to_list()]
    categorized = (
        announcements.with_columns(pl.Series("category", categories, dtype=pl.Utf8))
        .filter(pl.col("category").is_not_null())
        .sort(["symbol", "category", "ann_date", "art_code"])
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
        & (pl.col("entry_volume").fill_null(0) > 0)
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
    result["promotion_passed"] = bool(
        category in PROMOTABLE_CATEGORIES
        and result["tradable_events"] >= 150
        and result["announcement_days"] >= 100
        and result["tradable_rate"] >= 0.90
        and result["benchmark_coverage"] >= 0.99
        and result["entry_capacity_feasible_rate"] >= 0.95
        and result["unresolved_exits"] == 0
        and (result["mean_net_return"] or -math.inf) >= 0.015
        and (result["mean_excess_return"] or -math.inf) >= 0.01
        and (result["excess_daily_cluster_t"] or -math.inf) >= 2.5
        and result["positive_excess_years"] >= 5
        and (result["max_year_positive_excess_share"] or math.inf) <= 0.50
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
    trades = build_all_trades(events, panel)
    summaries = {
        category: summarize_category(trades, category)
        for category in CATEGORY_HOLDING_DAYS
    }
    promoted = [
        category
        for category in PROMOTABLE_CATEGORIES
        if summaries[category]["promotion_passed"]
    ]
    selected = (
        max(
            promoted, key=lambda category: summaries[category]["excess_daily_cluster_t"]
        )
        if promoted
        else None
    )
    payload = {
        "schema_version": "p0-restructuring-announcement-development-v1",
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
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "decision_clock": "announcement date treated as after-close; next trading open entry",
        },
        "data": {
            "raw_rows": raw.height,
            "raw_symbols": raw.get_column("symbol").n_unique(),
            "categorized_events": events.height,
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
                else "terminate_restructuring_announcement_mechanism"
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
            "/app/data/research/p0_restructuring_announcement_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
