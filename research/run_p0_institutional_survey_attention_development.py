"""Run the frozen development-only institutional-survey attention study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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

WARMUP_START = date(2013, 1, 1)
PANEL_START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
HOLD_TRADING_DAYS = 20
LOOKBACK_DAYS = 365
COOLDOWN_DAYS = 60
MIN_INSTITUTIONS = 10
MIN_ATTENTION_MULTIPLE = 2.0
CATEGORY = "institutional_survey_attention_spike"


def load_survey_events(data_dir: Path) -> pl.DataFrame:
    root = data_dir / "event_data" / "institutional_survey"
    paths = []
    missing = []
    for year in range(WARMUP_START.year, DEVELOPMENT_END.year + 1):
        for month in range(1, 13):
            path = root / f"year={year}" / f"month={month:02d}" / "part.parquet"
            if path.is_file():
                paths.append(path)
            else:
                missing.append(f"{year}-{month:02d}")
    if missing:
        raise ValueError(
            "all 2013-2020 institutional-survey partitions are required; missing "
            + ",".join(missing)
        )
    return (
        pl.read_parquet(paths)
        .filter(
            pl.col("notice_date").is_between(
                WARMUP_START, DEVELOPMENT_END, closed="both"
            )
        )
        .sort(["symbol", "notice_date", "event_id"])
    )


def select_attention_spikes(surveys: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for symbol_frame in surveys.partition_by("symbol", maintain_order=True):
        history: list[tuple[date, int]] = []
        last_selected: date | None = None
        for row in symbol_frame.iter_rows(named=True):
            notice_date = row["notice_date"]
            count = int(row["institution_count"])
            history = [
                (past_date, past_count)
                for past_date, past_count in history
                if 0 < (notice_date - past_date).days <= LOOKBACK_DAYS
            ]
            prior_median = (
                float(statistics.median(past_count for _, past_count in history))
                if history
                else None
            )
            multiple = count / prior_median if prior_median and prior_median > 0 else None
            in_development = DEVELOPMENT_START <= notice_date <= DEVELOPMENT_END
            cooldown_clear = (
                last_selected is None
                or (notice_date - last_selected).days >= COOLDOWN_DAYS
            )
            selected = bool(
                in_development
                and count >= MIN_INSTITUTIONS
                and multiple is not None
                and multiple >= MIN_ATTENTION_MULTIPLE
                and cooldown_clear
            )
            if selected:
                rows.append(
                    {
                        **row,
                        "ann_date": notice_date,
                        "category": CATEGORY,
                        "prior_365d_institution_median": prior_median,
                        "attention_multiple": multiple,
                    }
                )
                last_selected = notice_date
            history.append((notice_date, count))
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort(["ann_date", "symbol"])


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
    result["promotion_passed"] = evaluate_gate(result)
    return result


def evaluate_gate(result: dict[str, Any]) -> bool:
    return bool(
        int(result.get("tradable_events") or 0) >= 500
        and int(result.get("announcement_days") or 0) >= 300
        and float(result.get("tradable_rate") or 0.0) >= 0.90
        and float(result.get("benchmark_coverage") or 0.0) >= 0.99
        and float(result.get("entry_capacity_feasible_rate") or 0.0) >= 0.95
        and int(result.get("unresolved_exits") or 0) == 0
        and float(result.get("mean_net_return") or -math.inf) >= 0.03
        and float(result.get("mean_excess_return") or -math.inf) >= 0.02
        and float(result.get("excess_daily_cluster_t") or -math.inf) >= 3.0
        and int(result.get("positive_excess_years") or 0) >= 5
        and float(result.get("max_year_positive_excess_share") or math.inf) <= 0.40
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw = load_survey_events(data_dir)
    events = select_attention_spikes(raw)
    if events.is_empty():
        raise ValueError("institutional-survey attention filter produced no events")
    panel = prepare_panel(load_panel(data_dir, PANEL_START, PANEL_END))
    trades = build_trades(events, panel, HOLD_TRADING_DAYS)
    benchmark = build_market_benchmark(panel, HOLD_TRADING_DAYS)
    trades = attach_market_excess(trades, benchmark)
    summary = summarize(trades)
    payload = {
        "schema_version": "p0-institutional-survey-attention-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "warmup_start": WARMUP_START,
            "development_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "minimum_institutions": MIN_INSTITUTIONS,
            "minimum_attention_multiple": MIN_ATTENTION_MULTIPLE,
            "lookback_calendar_days": LOOKBACK_DAYS,
            "cooldown_calendar_days": COOLDOWN_DAYS,
            "holding_trading_days": HOLD_TRADING_DAYS,
            "position_notional_cny": POSITION_NOTIONAL,
            "daily_participation_rate": DAILY_PARTICIPATION,
            "decision_clock": "notice date treated as after-close; next trading open entry",
            "provider_sum_used_for_signal": False,
        },
        "data": {
            "raw_company_notice_events": raw.height,
            "attention_spike_events": events.height,
            "attention_spike_symbols": events.get_column("symbol").n_unique(),
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
                else "terminate_institutional_survey_attention"
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
            "/app/data/research/p0_institutional_survey_attention_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
