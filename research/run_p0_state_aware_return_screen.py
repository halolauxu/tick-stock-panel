"""Screen seven frozen state-aware return mechanisms on development data."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_academic_factor_development_screen as academic  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
TARGET_POSITIONS = 10
MIN_LISTING_DAYS = 180
MIN_AMOUNT_STANDARD = 50_000_000.0
MIN_AMOUNT_MICROCAP = 20_000_000.0
MECHANISMS = (
    "revised_momentum_monthly",
    "state_momentum_monthly",
    "healthy_reversal_monthly",
    "attention_continuation_weekly",
    "attention_absorption_weekly",
    "microcap_state_weekly",
    "microcap_reversal_weekly",
)


def _rolling_return(column: str, window: int, group: str | None = None) -> pl.Expr:
    expr = pl.col(column).log1p().rolling_sum(window, min_samples=window)
    if group:
        expr = expr.over(group)
    return expr.exp() - 1.0


def attach_return_features(panel: pl.DataFrame) -> pl.DataFrame:
    work = panel.sort(["symbol", "date"]).with_columns(
        (
            pl.col("raw_close")
            >= pl.col("limit_up_price") - pl.lit(0.005)
        ).alias("is_limit_up"),
        (pl.col("amount") / pl.col("market_cap")).alias("turnover_proxy"),
        pl.col("_global_index").shift(4).over("symbol").alias("_index_5"),
        pl.col("_global_index").shift(19).over("symbol").alias("_index_20"),
        pl.col("_global_index").shift(59).over("symbol").alias("_index_60"),
        pl.col("_global_index").shift(119).over("symbol").alias("_index_120"),
    )
    work = work.with_columns(
        pl.when(pl.col("is_limit_up"))
        .then(0.0)
        .otherwise(pl.col("daily_return"))
        .alias("revised_daily_return")
    )
    work = work.with_columns(
        _rolling_return("daily_return", 5, "symbol").alias("return_5d_raw"),
        _rolling_return("daily_return", 20, "symbol").alias("return_20d_raw"),
        _rolling_return("revised_daily_return", 120, "symbol").alias(
            "revised_momentum_120d_raw"
        ),
        pl.col("turnover_proxy")
        .rolling_mean(5, min_samples=5)
        .over("symbol")
        .alias("turnover_5d"),
        pl.col("turnover_proxy")
        .rolling_mean(60, min_samples=60)
        .over("symbol")
        .alias("turnover_60d"),
        pl.col("amount")
        .rolling_mean(20, min_samples=20)
        .over("symbol")
        .alias("mean_amount_20d"),
    )
    return work.with_columns(
        pl.when(pl.col("_global_index") == pl.col("_index_5") + 4)
        .then(pl.col("return_5d_raw"))
        .otherwise(None)
        .alias("return_5d"),
        pl.when(pl.col("_global_index") == pl.col("_index_20") + 19)
        .then(pl.col("return_20d_raw"))
        .otherwise(None)
        .alias("return_20d"),
        pl.when(pl.col("_global_index") == pl.col("_index_120") + 119)
        .then(pl.col("revised_momentum_120d_raw"))
        .otherwise(None)
        .alias("revised_momentum_120d"),
        pl.when(pl.col("_global_index") == pl.col("_index_60") + 59)
        .then(pl.col("turnover_5d") / pl.col("turnover_60d"))
        .otherwise(None)
        .alias("attention_ratio"),
        (pl.col("date") - pl.col("list_date"))
        .dt.total_days()
        .alias("listing_days"),
    )


def build_market_state(panel: pl.DataFrame) -> pl.DataFrame:
    ranked = panel.filter(pl.col("daily_return").is_not_null()).with_columns(
        pl.len().over("date").alias("universe_count"),
        pl.col("market_cap")
        .rank(method="ordinal")
        .over("date")
        .alias("market_cap_rank"),
    )
    state = (
        ranked.group_by("date")
        .agg(
            pl.col("daily_return").mean().alias("market_daily_return"),
            pl.col("daily_return")
            .filter(
                pl.col("market_cap_rank")
                <= (pl.col("universe_count") * 0.10).ceil()
            )
            .mean()
            .alias("microcap_daily_return"),
        )
        .sort("date")
        .with_columns(
            _rolling_return("market_daily_return", 20).alias(
                "market_return_20d"
            ),
            _rolling_return("market_daily_return", 120).alias(
                "market_return_120d"
            ),
            _rolling_return("microcap_daily_return", 20).alias(
                "microcap_return_20d"
            ),
        )
    )
    return state


def signal_observations(
    panel: pl.DataFrame, frequency: str
) -> tuple[pl.DataFrame, list[date]]:
    if frequency == "monthly":
        period = pl.col("date").dt.strftime("%Y-%m")
    elif frequency == "weekly":
        period = pl.col("date").dt.strftime("%G-%V")
    else:
        raise ValueError(f"unknown frequency: {frequency}")
    calendar = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            period.alias("period"),
        )
        .group_by("period", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .filter(pl.col("signal_date") >= DEVELOPMENT_START)
    )
    observations = panel.join(
        calendar.select("signal_date", "entry_date"),
        left_on="date",
        right_on="signal_date",
        how="inner",
    )
    return observations, calendar.get_column("entry_date").to_list()


def _base_filter(microcap: bool = False) -> pl.Expr:
    minimum = MIN_AMOUNT_MICROCAP if microcap else MIN_AMOUNT_STANDARD
    return (
        (pl.col("listing_days") >= MIN_LISTING_DAYS)
        & (pl.col("mean_amount_20d") >= minimum)
        & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
    )


def build_candidates(observations: pl.DataFrame, mechanism: str) -> pl.DataFrame:
    work = observations.with_columns(
        pl.len().over("date").alias("universe_count"),
        pl.col("market_cap")
        .rank(method="ordinal")
        .over("date")
        .alias("market_cap_rank"),
    ).with_columns(
        (
            ((pl.col("market_cap_rank") - 1) * 10 / pl.col("universe_count"))
            .floor()
            .clip(0, 9)
            .cast(pl.UInt8)
        ).alias("market_cap_decile")
    )
    if mechanism == "revised_momentum_monthly":
        selected = work.filter(
            _base_filter()
            & (pl.col("date").dt.month() != 2)
            & (pl.col("revised_momentum_120d") > 0)
        )
        sort_columns = ["revised_momentum_120d", "market_cap"]
        descending = [True, False]
        score = pl.col("revised_momentum_120d")
    elif mechanism == "state_momentum_monthly":
        selected = work.filter(
            _base_filter()
            & (pl.col("date").dt.month() != 2)
            & (pl.col("market_return_120d") > 0)
            & (pl.col("revised_momentum_120d") > 0)
        )
        sort_columns = ["revised_momentum_120d", "market_cap"]
        descending = [True, False]
        score = pl.col("revised_momentum_120d")
    elif mechanism == "healthy_reversal_monthly":
        selected = work.filter(
            _base_filter()
            & (pl.col("market_return_20d") > 0)
            & pl.col("return_20d").is_between(-0.35, -0.02, closed="both")
        )
        sort_columns = ["return_20d", "market_cap"]
        descending = [False, False]
        score = -pl.col("return_20d")
    elif mechanism == "attention_continuation_weekly":
        selected = work.filter(
            _base_filter()
            & (pl.col("market_return_20d") > 0)
            & (pl.col("attention_ratio") >= 1.5)
            & pl.col("return_20d").is_between(0.02, 0.20, closed="both")
        )
        sort_columns = ["attention_ratio", "return_20d"]
        descending = [True, True]
        score = pl.col("attention_ratio")
    elif mechanism == "attention_absorption_weekly":
        selected = work.filter(
            _base_filter()
            & (pl.col("market_return_20d") > 0)
            & (pl.col("attention_ratio") >= 1.5)
            & pl.col("return_5d").is_between(-0.15, -0.02, closed="both")
        )
        sort_columns = ["attention_ratio", "return_5d"]
        descending = [True, False]
        score = pl.col("attention_ratio")
    elif mechanism == "microcap_state_weekly":
        selected = work.filter(
            _base_filter(microcap=True)
            & (pl.col("market_cap_decile") == 0)
            & (pl.col("market_return_20d") > 0)
            & (pl.col("microcap_return_20d") > pl.col("market_return_20d"))
        )
        sort_columns = ["market_cap", "return_20d"]
        descending = [False, True]
        score = -pl.col("market_cap")
    elif mechanism == "microcap_reversal_weekly":
        selected = work.filter(
            _base_filter(microcap=True)
            & (pl.col("market_cap_decile") == 0)
            & (pl.col("market_return_20d") > 0)
            & pl.col("return_20d").is_between(-0.35, -0.02, closed="both")
        )
        sort_columns = ["return_20d", "market_cap"]
        descending = [False, False]
        score = -pl.col("return_20d")
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")
    return (
        selected.sort(
            ["date", *sort_columns, "symbol"],
            descending=[False, *descending, False],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("date").alias("cap_rank"),
            score.alias("signal_score"),
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            "date",
            "entry_date",
            "symbol",
            "signal_score",
            "market_cap",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def evaluate_gate(
    result: dict[str, Any], benchmark: dict[str, Any], active_fraction: float
) -> dict[str, Any]:
    decision = shared.evaluate_gate(result, benchmark)
    decision["checks"].pop("mean_cash_ratio_at_most_25pct")
    decision["checks"]["active_rebalance_fraction_at_least_35pct"] = (
        active_fraction >= 0.35
    )
    decision["passed"] = all(decision["checks"].values())
    decision["verdict"] = (
        "PROMOTE_TO_VALIDATION" if decision["passed"] else "TERMINATE"
    )
    return decision


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    raw_source = raw_all.filter(pl.col("date") >= DEVELOPMENT_START)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_all, data_dir)
    panel = attach_return_features(baseline.prepare_panel(pit))
    panel = panel.join(build_market_state(panel), on="date", how="left")
    del pit
    gc.collect()
    benchmark = shared.benchmark_metrics(
        panel.filter(pl.col("date") >= DEVELOPMENT_START)
    )
    monthly, monthly_actions = signal_observations(panel, "monthly")
    weekly, weekly_actions = signal_observations(panel, "weekly")
    del panel
    gc.collect()
    results: dict[str, Any] = {}
    promoted: list[str] = []
    for mechanism in MECHANISMS:
        observations, action_dates = (
            (monthly, monthly_actions)
            if mechanism.endswith("monthly")
            else (weekly, weekly_actions)
        )
        candidates = build_candidates(observations, mechanism)
        active_days = candidates.get_column("entry_date").n_unique()
        active_fraction = active_days / len(action_dates)
        result = academic.simulate_factor(
            candidates, raw_source, all_dates, action_dates, data_dir
        )
        decision = evaluate_gate(result, benchmark, active_fraction)
        results[mechanism] = {
            "data": {
                "signal_rows": candidates.height,
                "signal_symbols": candidates.get_column("symbol").n_unique(),
                "scheduled_rebalance_days": len(action_dates),
                "active_rebalance_days": active_days,
                "active_rebalance_fraction": active_fraction,
            },
            "strategy": result,
            "decision": decision,
        }
        if decision["passed"]:
            promoted.append(mechanism)
    payload = {
        "schema_version": "p0-state-aware-return-screen-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "benchmark": benchmark,
        "mechanisms": MECHANISMS,
        "results": results,
        "promoted_to_independent_validation": promoted,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        mechanism: {
            "data": row["data"],
            "metrics": row["strategy"]["metrics"],
            "execution": row["strategy"]["execution"],
            "integrity": row["strategy"]["integrity"],
            "account": row["strategy"]["account"],
            "decision": row["decision"],
        }
        for mechanism, row in results.items()
    }
    print(
        json.dumps(
            {
                "benchmark": benchmark,
                "results": summary,
                "promoted_to_independent_validation": promoted,
                "output": str(output),
                "sha256": digest,
            },
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
        default=Path("/app/data/research/p0_state_aware_return_screen.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
