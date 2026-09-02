"""Run the frozen development-only weekly smart-money disagreement study."""
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
    build_trades,
    load_panel,
    prepare_panel,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    attach_market_excess,
    build_market_benchmark,
    summarize_category,
)

START = date(2013, 12, 1)
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
PANEL_END = date(2021, 3, 31)
PRIMARY_HORIZON = 5
DIAGNOSTIC_HORIZONS = (2, 10)
MAX_EXIT_DELAY = 20
TARGETS_PER_WEEK = 10
MIN_DAILY_AMOUNT = 50_000_000.0
MAIN_BOARD_PATTERN = r"^(?:600|601|603|605|000|001|002)\d{3}\.(?:SH|SZ)$"

CANDIDATE = "smart_money_disagreement"
LARGE_CONTROL = "large_inflow_control"
RETAIL_CONTROL = "retail_dominance_control"
CATEGORIES = (CANDIDATE, LARGE_CONTROL, RETAIL_CONTROL)


def load_moneyflow(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "event_data" / "moneyflow").glob(
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
        raise ValueError("all 2014-2020 moneyflow yearly partitions are required")
    return (
        pl.read_parquet(sorted(paths), hive_partitioning=False)
        .filter(
            pl.col("trade_date").is_between(
                DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
            )
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        )
        .sort(["trade_date", "symbol"])
    )


def build_weekly_events(
    moneyflow: pl.DataFrame, event_day_panel: pl.DataFrame
) -> pl.DataFrame:
    large_net = (
        pl.col("buy_lg_cny").fill_null(0)
        + pl.col("buy_elg_cny").fill_null(0)
        - pl.col("sell_lg_cny").fill_null(0)
        - pl.col("sell_elg_cny").fill_null(0)
    )
    small_net = pl.col("buy_sm_cny").fill_null(0) - pl.col(
        "sell_sm_cny"
    ).fill_null(0)
    work = (
        moneyflow.join(event_day_panel, on=["symbol", "trade_date"], how="inner")
        .filter(
            (pl.col("event_daily_amount") >= MIN_DAILY_AMOUNT)
            & pl.col("symbol").str.contains(MAIN_BOARD_PATTERN)
        )
        .with_columns(
            large_net.alias("large_net_flow_cny"),
            small_net.alias("small_net_flow_cny"),
            pl.col("trade_date").dt.truncate("1w").alias("week"),
        )
        .with_columns(
            (
                (pl.col("large_net_flow_cny") - pl.col("small_net_flow_cny"))
                / pl.col("event_daily_amount")
            ).alias("disagreement_ratio"),
            (
                pl.col("large_net_flow_cny") / pl.col("event_daily_amount")
            ).alias("large_net_ratio"),
            pl.col("trade_date").max().over("week").alias("week_last_trade_date"),
        )
        .filter(pl.col("trade_date") == pl.col("week_last_trade_date"))
    )

    smart = (
        work.filter(
            (pl.col("large_net_flow_cny") > 0)
            & (pl.col("small_net_flow_cny") < 0)
        )
        .sort(
            ["trade_date", "disagreement_ratio", "symbol"],
            descending=[False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("trade_date").alias("signal_rank"),
            pl.lit(CANDIDATE).alias("category"),
        )
        .filter(pl.col("signal_rank") <= TARGETS_PER_WEEK)
    )
    large = (
        work.filter(pl.col("large_net_flow_cny") > 0)
        .sort(
            ["trade_date", "large_net_ratio", "symbol"],
            descending=[False, True, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("trade_date").alias("signal_rank"),
            pl.lit(LARGE_CONTROL).alias("category"),
        )
        .filter(pl.col("signal_rank") <= TARGETS_PER_WEEK)
    )
    retail = (
        work.filter(
            (pl.col("large_net_flow_cny") < 0)
            & (pl.col("small_net_flow_cny") > 0)
        )
        .sort(
            ["trade_date", "disagreement_ratio", "symbol"],
            descending=[False, False, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("trade_date").alias("signal_rank"),
            pl.lit(RETAIL_CONTROL).alias("category"),
        )
        .filter(pl.col("signal_rank") <= TARGETS_PER_WEEK)
    )
    return (
        pl.concat([smart, large, retail], how="diagonal_relaxed")
        .rename({"trade_date": "ann_date"})
        .sort(["ann_date", "category", "signal_rank", "symbol"])
    )


def build_event_day_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.select(
        "symbol",
        pl.col("date").alias("trade_date"),
        pl.col("amount").alias("event_daily_amount"),
    )


def _summaries(
    events: pl.DataFrame, panel: pl.DataFrame, horizon: int
) -> tuple[dict[str, dict[str, Any]], pl.DataFrame]:
    trades = build_trades(
        events,
        panel,
        holding_trading_days=horizon,
        max_exit_delay=MAX_EXIT_DELAY,
    )
    benchmark = build_market_benchmark(panel, horizon)
    trades = attach_market_excess(trades, benchmark)
    summaries = {
        category: summarize_category(
            trades,
            category,
            positive_categories=(CANDIDATE,),
            min_tradable_events=1_000,
            min_announcement_days=250,
        )
        for category in CATEGORIES
    }
    return summaries, trades


def evaluate(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = primary[CANDIDATE]
    large = primary[LARGE_CONTROL]
    retail = primary[RETAIL_CONTROL]
    candidate_excess = candidate.get("mean_excess_return")
    large_excess = large.get("mean_excess_return")
    retail_excess = retail.get("mean_excess_return")
    versus_large = (
        candidate_excess - large_excess
        if candidate_excess is not None and large_excess is not None
        else None
    )
    checks = {
        "at_least_1000_tradable_observations": candidate["tradable_events"] >= 1_000,
        "at_least_250_signal_weeks": candidate["announcement_days"] >= 250,
        "tradable_rate_at_least_90pct": candidate["tradable_rate"] >= 0.90,
        "no_unresolved_exits": candidate["unresolved_exits"] == 0,
        "mean_net_return_at_least_30bp": (candidate.get("mean_net_return") or -math.inf)
        >= 0.003,
        "mean_excess_at_least_20bp": (candidate_excess or -math.inf) >= 0.002,
        "weekly_cluster_t_at_least_2_5": (
            candidate.get("excess_daily_cluster_t") or -math.inf
        )
        >= 2.5,
        "at_least_5_positive_years": candidate["positive_excess_years"] >= 5,
        "beats_large_inflow_control_by_10bp": (versus_large or -math.inf) >= 0.001,
        "beats_retail_dominance_control": (
            candidate_excess is not None
            and retail_excess is not None
            and candidate_excess > retail_excess
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_ACCOUNT_CONTRACT" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "candidate_excess_minus_large_control": versus_large,
        "validation_read": False,
        "known_stress_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_flow = load_moneyflow(data_dir)
    panel = prepare_panel(load_panel(data_dir, START, PANEL_END))
    events = build_weekly_events(raw_flow, build_event_day_panel(panel))
    primary, _ = _summaries(events, panel, PRIMARY_HORIZON)
    diagnostics = {
        str(horizon): _summaries(events, panel, horizon)[0]
        for horizon in DIAGNOSTIC_HORIZONS
    }
    decision = evaluate(primary)
    payload = {
        "schema_version": "p0-smart-money-disagreement-discovery-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "signal_frequency": "last_actual_trading_day_of_each_week",
            "targets_per_week": TARGETS_PER_WEEK,
            "primary_holding_trading_days": PRIMARY_HORIZON,
            "diagnostic_holding_trading_days": list(DIAGNOSTIC_HORIZONS),
            "minimum_signal_daily_amount_cny": MIN_DAILY_AMOUNT,
            "maximum_exit_delay_trading_days": MAX_EXIT_DELAY,
        },
        "data": {
            "raw_moneyflow_rows": raw_flow.height,
            "weekly_event_rows": events.height,
            "signal_weeks": events.get_column("ann_date").n_unique(),
            "event_symbols": events.get_column("symbol").n_unique(),
            "panel_rows": panel.height,
            "panel_symbols": panel.get_column("symbol").n_unique(),
        },
        "primary_5d": primary,
        "diagnostics_only": diagnostics,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": digest},
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
            "/app/data/research/p0_smart_money_disagreement_discovery.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
