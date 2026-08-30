"""Run the frozen development-only micro-cap flow-absorption account study."""
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

import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
FLOW_WINDOW = 20
MIN_FLOW_OBSERVATIONS = 15
MIN_TRAILING_AMOUNT = 500_000_000.0
TARGET_POSITIONS = 10
INITIAL_CASH = 200_000.0


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
    return pl.read_parquet(sorted(paths), hive_partitioning=False).filter(
        pl.col("trade_date").is_between(
            DEVELOPMENT_START, DEVELOPMENT_END, closed="both"
        )
    )


def attach_flow_features(
    panel: pl.DataFrame, moneyflow: pl.DataFrame
) -> pl.DataFrame:
    flow = (
        moneyflow.select(
            "symbol",
            pl.col("trade_date").cast(pl.Date).alias("date"),
            (
                pl.col("buy_lg_cny").fill_null(0.0)
                + pl.col("buy_elg_cny").fill_null(0.0)
                - pl.col("sell_lg_cny").fill_null(0.0)
                - pl.col("sell_elg_cny").fill_null(0.0)
            ).alias("large_net_flow_cny"),
        )
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )
    return (
        panel.join(flow, on=["symbol", "date"], how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("large_net_flow_cny")
            .is_not_null()
            .cast(pl.Int16)
            .rolling_sum(window_size=FLOW_WINDOW, min_samples=1)
            .over("symbol")
            .alias("flow_observations_20d"),
            pl.col("large_net_flow_cny")
            .fill_null(0.0)
            .rolling_sum(window_size=FLOW_WINDOW, min_samples=1)
            .over("symbol")
            .alias("large_net_flow_20d_cny"),
            pl.col("amount")
            .fill_null(0.0)
            .rolling_sum(window_size=FLOW_WINDOW, min_samples=1)
            .over("symbol")
            .alias("amount_20d_cny"),
        )
        .with_columns(
            (
                pl.col("large_net_flow_20d_cny") / pl.col("amount_20d_cny")
            ).alias("large_net_flow_ratio_20d")
        )
    )


def _weekly_signal_rows(panel: pl.DataFrame) -> pl.DataFrame:
    weekly = (
        panel.select("date")
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").shift(-1).alias("entry_date"),
            pl.col("date").dt.strftime("%G-%V").alias("week"),
        )
        .group_by("week", maintain_order=True)
        .agg(
            pl.col("date").max().alias("signal_date"),
            pl.col("entry_date").last().alias("entry_date"),
        )
        .drop_nulls("entry_date")
    )
    return (
        panel.join(weekly, left_on="date", right_on="signal_date", how="inner")
        .filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap")
            .rank(method="ordinal")
            .over("date")
            .alias("market_cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("market_cap_rank") - 1) * 10 / pl.col("universe_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("cap_decile")
        )
        .filter(pl.col("cap_decile") == 0)
    )


def build_candidates(panel: pl.DataFrame, *, use_flow: bool) -> pl.DataFrame:
    work = _weekly_signal_rows(panel)
    if use_flow:
        work = (
            work.filter(
                (pl.col("flow_observations_20d") >= MIN_FLOW_OBSERVATIONS)
                & (pl.col("amount_20d_cny") >= MIN_TRAILING_AMOUNT)
                & (pl.col("large_net_flow_ratio_20d") > 0)
            )
            .sort(
                ["date", "large_net_flow_ratio_20d", "market_cap", "symbol"],
                descending=[False, True, False, False],
            )
            .with_columns(
                pl.int_range(1, pl.len() + 1)
                .over("date")
                .alias("selection_rank")
            )
            .filter(pl.col("selection_rank") <= TARGET_POSITIONS)
        )
    else:
        work = (
            work.sort(["date", "market_cap_rank", "symbol"])
            .with_columns(
                pl.int_range(1, pl.len() + 1)
                .over("date")
                .alias("selection_rank")
            )
            .filter(pl.col("selection_rank") <= TARGET_POSITIONS)
        )
    return work.select(
        "date",
        "entry_date",
        "symbol",
        "market_cap",
        pl.col("amount").alias("signal_amount"),
        pl.col("selection_rank").alias("cap_rank"),
        "market_cap_rank",
        "flow_observations_20d",
        "large_net_flow_20d_cny",
        "amount_20d_cny",
        "large_net_flow_ratio_20d",
    ).sort(["entry_date", "cap_rank", "symbol"])


def _annualized(returns: list[float]) -> float | None:
    total = baseline._compound(returns)
    if not returns or total is None or total <= -1.0:
        return None
    return (1.0 + total) ** (252.0 / len(returns)) - 1.0


def summarize_variant(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    data_dir: Path,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quote_source = raw_source.filter(pl.col("symbol").is_in(symbols))
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(quote_source, data_dir)
    )
    grid = account.build_execution_grid(candidates, quotes)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
    )
    daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = (
            daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        result = baseline._compound(values)
        positive_years += int(result is not None and result > 0)
        yearly.append({"year": year, "account_return": result})
    return {
        "metrics": {
            "trading_days": daily.height,
            "account_annualized": _annualized(returns),
            "account_total_return": baseline._compound(returns),
            "account_max_drawdown": baseline._max_drawdown(returns),
            "positive_account_years": positive_years,
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily),
        "signal_rows": candidates.height,
        "rebalance_days": candidates.get_column("entry_date").n_unique(),
        "daily_equity": daily.select(
            "date", "equity", "cash", "position_value", "position_count"
        ).to_dicts(),
        "orders": simulation["orders"],
        "worst_weeks": account.worst_weeks(daily),
    }


def evaluate_gate(
    flow_result: dict[str, Any], control_result: dict[str, Any]
) -> dict[str, Any]:
    flow = flow_result["metrics"]
    control = control_result["metrics"]
    flow_annual = flow.get("account_annualized")
    control_annual = control.get("account_annualized")
    improvement = (
        flow_annual - control_annual
        if flow_annual is not None and control_annual is not None
        else None
    )
    checks = {
        "annualized_at_least_50pct": (flow_annual or -math.inf) >= 0.50,
        "annualized_improvement_at_least_10pp": (improvement or -math.inf)
        >= 0.10,
        "max_drawdown_no_worse_than_35pct": (
            flow.get("account_max_drawdown") or -math.inf
        )
        >= -0.35,
        "at_least_five_positive_years": flow.get("positive_account_years", 0)
        >= 5,
        "mean_cash_ratio_at_most_25pct": (
            flow.get("mean_cash_ratio") or math.inf
        )
        <= 0.25,
        "buy_execution_at_least_80pct": flow_result["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.80,
        "sell_execution_at_least_80pct": flow_result["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.80,
        "ending_positions_resolved": flow_result["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "cash_reconciled": flow_result["integrity"][
            "max_cash_reconciliation_error"
        ]
        <= 0.01,
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION" if passed else "TERMINATE",
        "passed": passed,
        "checks": checks,
        "annualized_improvement_vs_microcap_control": improvement,
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
    raw_source = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START
    )
    if raw_source.is_empty():
        raise ValueError("no development daily data")
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(raw_source, data_dir)
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    panel = attach_flow_features(panel, load_moneyflow(data_dir))
    flow_candidates = build_candidates(panel, use_flow=True)
    control_candidates = build_candidates(panel, use_flow=False)
    del panel
    gc.collect()
    flow_result = summarize_variant(
        flow_candidates, raw_source, all_dates, data_dir
    )
    control_result = summarize_variant(
        control_candidates, raw_source, all_dates, data_dir
    )
    decision = evaluate_gate(flow_result, control_result)
    payload = {
        "schema_version": "p0-microcap-flow-development-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "initial_cash_cny": INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "flow_window_observations": FLOW_WINDOW,
            "minimum_flow_observations": MIN_FLOW_OBSERVATIONS,
            "minimum_trailing_amount_cny": MIN_TRAILING_AMOUNT,
            "signal": "weekly PIT market-cap bottom decile, positive 20-observation large-order net-flow ratio, top 10",
            "control": "weekly PIT market-cap bottom decile, smallest 10",
            "execution": "next trading day open, sells before buys",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "flow_signal_rows": flow_candidates.height,
            "control_signal_rows": control_candidates.height,
        },
        "flow_strategy": flow_result,
        "microcap_control": control_result,
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
            {
                "data": payload["data"],
                "flow_strategy": {
                    "metrics": flow_result["metrics"],
                    "execution": flow_result["execution"],
                    "integrity": flow_result["integrity"],
                    "account": flow_result["account"],
                },
                "microcap_control": {
                    "metrics": control_result["metrics"],
                    "execution": control_result["execution"],
                    "integrity": control_result["integrity"],
                    "account": control_result["account"],
                },
                "decision": decision,
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
        default=Path(
            "/app/data/research/p0_microcap_flow_development.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
