"""Discover a frozen prior-close regime gate for main-board forecast drift."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_forecast_drift_account as base  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CONTROL = "always_on"
STATE_NAMES = (
    "market_5d_positive",
    "market_20d_positive",
    "market_60d_positive",
    "market_120d_positive",
    "market_20d_and_120d_positive",
    "breadth_60d_at_least_half",
)


def _rolling_return(column: str, window: int) -> pl.Expr:
    return pl.col(column).log1p().rolling_sum(window_size=window, min_samples=window).exp() - 1.0


def build_market_states(panel: pl.DataFrame) -> pl.DataFrame:
    """Build close-t state and map it only to the next market open."""
    featured = (
        panel.sort(["symbol", "date"])
        .with_columns(
            pl.col("close")
            .rolling_mean(window_size=60, min_samples=60)
            .over("symbol")
            .alias("ma60"),
            pl.col("_global_index").shift(59).over("symbol").alias("index_60"),
        )
        .with_columns(
            pl.when(pl.col("_global_index") == pl.col("index_60") + 59)
            .then(pl.col("close") > pl.col("ma60"))
            .otherwise(None)
            .alias("above_ma60")
        )
    )
    daily_state = (
        featured.group_by("date")
        .agg(
            pl.col("daily_return")
            .filter(pl.col("daily_return").is_finite())
            .mean()
            .alias("market_daily_return"),
            pl.col("above_ma60").mean().alias("breadth_60d"),
        )
        .sort("date")
        .with_columns(
            _rolling_return("market_daily_return", 5).alias("market_return_5d"),
            _rolling_return("market_daily_return", 20).alias("market_return_20d"),
            _rolling_return("market_daily_return", 60).alias("market_return_60d"),
            _rolling_return("market_daily_return", 120).alias("market_return_120d"),
            pl.col("date").shift(-1).alias("entry_date"),
        )
        .drop_nulls("entry_date")
        .with_columns(
            pl.lit(True).alias(CONTROL),
            (pl.col("market_return_5d") > 0).fill_null(False).alias("market_5d_positive"),
            (pl.col("market_return_20d") > 0).fill_null(False).alias("market_20d_positive"),
            (pl.col("market_return_60d") > 0).fill_null(False).alias("market_60d_positive"),
            (pl.col("market_return_120d") > 0).fill_null(False).alias("market_120d_positive"),
            ((pl.col("market_return_20d") > 0) & (pl.col("market_return_120d") > 0))
            .fill_null(False)
            .alias("market_20d_and_120d_positive"),
            (pl.col("breadth_60d") >= 0.50).fill_null(False).alias("breadth_60d_at_least_half"),
        )
        .rename({"date": "signal_date"})
    )
    return daily_state.select(
        "signal_date",
        "entry_date",
        "market_return_5d",
        "market_return_20d",
        "market_return_60d",
        "market_return_120d",
        "breadth_60d",
        CONTROL,
        *STATE_NAMES,
    )


def gate_candidates(
    candidates: pl.DataFrame, states: pl.DataFrame, state_name: str
) -> pl.DataFrame:
    return (
        candidates.join(states.select("entry_date", state_name), on="entry_date", how="left")
        .filter(pl.col(state_name).fill_null(False))
        .drop(state_name)
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def intent_execution_summary(
    orders: list[dict[str, Any]], action_dates: list[date]
) -> dict[str, Any]:
    """Collapse consecutive daily retries into one executable order intent."""
    date_index = {value: index for index, value in enumerate(action_dates)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in orders:
        grouped[(str(row["side"]), str(row["symbol"]))].append(row)
    totals = {
        "BUY": {"intents": 0, "executed": 0},
        "SELL": {"intents": 0, "executed": 0},
    }
    for (side, _symbol), rows in grouped.items():
        rows.sort(key=lambda row: row["date"])
        episode: list[dict[str, Any]] = []
        previous_index: int | None = None
        for row in rows:
            current_index = date_index[row["date"]]
            if episode and (
                previous_index is None
                or current_index != previous_index + 1
                or episode[-1]["status"] == "FILLED"
            ):
                totals[side]["intents"] += 1
                totals[side]["executed"] += int(any(item["status"] == "FILLED" for item in episode))
                episode = []
            episode.append(row)
            previous_index = current_index
        if episode:
            totals[side]["intents"] += 1
            totals[side]["executed"] += int(any(item["status"] == "FILLED" for item in episode))
    for values in totals.values():
        values["execution_rate"] = (
            values["executed"] / values["intents"] if values["intents"] else 1.0
        )
    return {"buy": totals["BUY"], "sell": totals["SELL"]}


def simulate_variant(
    candidates: pl.DataFrame,
    grid: pl.DataFrame,
    quotes: pl.DataFrame,
    all_dates: list[date],
) -> dict[str, Any]:
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=base.PRIMARY_CAPITAL,
        target_positions=base.TARGET_POSITIONS,
        action_dates=all_dates,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=base.PRIMARY_CAPITAL
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    yearly: list[dict[str, Any]] = []
    positive_years = 0
    for year in range(base.DEVELOPMENT_START.year, base.DEVELOPMENT_END.year + 1):
        values = (
            account_daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        value = baseline._compound(values)
        positive_years += int(value is not None and value > 0)
        yearly.append({"year": year, "account_return": value})
    return {
        "metrics": {
            "annualized": shared._annualized(returns),
            "total_return": baseline._compound(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": account_daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "daily_attempt_execution": account.execution_summary(simulation["orders"]),
        "intent_execution": intent_execution_summary(simulation["orders"], all_dates),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation["max_cash_reconciliation_error"],
        },
        "account": account.account_summary(simulation, account_daily),
    }


def gate_checks(
    result: dict[str, Any], control: dict[str, Any], benchmark: dict[str, Any], active: float
) -> dict[str, bool]:
    metrics = result["metrics"]
    control_metrics = control["metrics"]
    annualized = metrics.get("annualized")
    drawdown = metrics.get("max_drawdown")
    return {
        "annualized_at_least_20pct": (annualized or -math.inf) >= 0.20,
        "annualized_improves_control_by_3pp": (
            (annualized or -math.inf) - (control_metrics.get("annualized") or math.inf) >= 0.03
        ),
        "annualized_excess_at_least_5pp": (
            (annualized or -math.inf) - (benchmark.get("annualized") or math.inf) >= 0.05
        ),
        "max_drawdown_no_worse_than_30pct": (drawdown or -math.inf) >= -0.30,
        "drawdown_improves_control_by_5pp": (
            (drawdown or -math.inf) - (control_metrics.get("max_drawdown") or math.inf) >= 0.05
        ),
        "at_least_5_positive_years": metrics["positive_years"] >= 5,
        "active_ratio_at_least_35pct": active >= 0.35,
        "mean_cash_ratio_at_most_75pct": (metrics.get("mean_cash_ratio") or math.inf) <= 0.75,
        "buy_intent_execution_at_least_90pct": result["intent_execution"]["buy"]["execution_rate"]
        >= 0.90,
        "sell_intent_execution_at_least_90pct": result["intent_execution"]["sell"]["execution_rate"]
        >= 0.90,
        "no_unresolved_positions": result["integrity"]["ending_unresolved_positions"] == 0,
        "cash_reconciled": result["integrity"]["max_cash_reconciliation_error"] <= 0.01,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    raw_all = baseline.load_daily(data_dir, end=base.DEVELOPMENT_END).filter(
        pl.col("date") >= base.DEVELOPMENT_START
    )
    raw_source = main_board.filter_main_board(raw_all)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = baseline.prepare_panel(baseline.attach_point_in_time_data(raw_source, data_dir))
    candidates, signal_audit = base.build_candidates(base.load_events(data_dir), panel, all_dates)
    states = build_market_states(panel)
    benchmark = shared.benchmark_metrics(panel)
    del panel
    gc.collect()

    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir)
    )
    grid = daily.build_action_grid(candidates, quotes, all_dates)
    results: dict[str, Any] = {}
    for state_name in (CONTROL, *STATE_NAMES):
        gated = gate_candidates(candidates, states, state_name)
        active_ratio = states.get_column(state_name).mean()
        result = simulate_variant(gated, grid, quotes, all_dates)
        result["state"] = {
            "active_ratio": active_ratio,
            "candidate_rows": gated.height,
            "candidate_days": gated.get_column("entry_date").n_unique(),
        }
        results[state_name] = result

    control = results[CONTROL]
    qualified: list[str] = []
    checks: dict[str, dict[str, bool]] = {}
    for state_name in STATE_NAMES:
        active_ratio = results[state_name]["state"]["active_ratio"]
        checks[state_name] = gate_checks(results[state_name], control, benchmark, active_ratio)
        if all(checks[state_name].values()):
            qualified.append(state_name)
    order = {name: index for index, name in enumerate(STATE_NAMES)}
    qualified.sort(
        key=lambda name: (
            -(results[name]["metrics"].get("annualized") or -math.inf),
            -(results[name]["metrics"].get("max_drawdown") or -math.inf),
            order[name],
        )
    )
    selected = qualified[0] if qualified else None
    decision = {
        "verdict": "FREEZE_VALIDATION_CONTRACT" if selected else "TERMINATE_FAMILY",
        "selected": selected,
        "qualified": qualified,
        "checks": checks,
        "validation_read": False,
        "known_stress_read": False,
    }
    payload = {
        "schema_version": "p0-main-board-forecast-drift-regime-discovery-v1",
        "contract_frozen": "2026-09-03",
        "period": {
            "start": base.DEVELOPMENT_START,
            "end": base.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "initial_cash_cny": base.PRIMARY_CAPITAL,
            "state_information_cutoff": "previous_trading_day_close",
            "states": list(STATE_NAMES),
        },
        "data": signal_audit,
        "benchmark": benchmark,
        "results": results,
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
                "benchmark": benchmark,
                "summary": {
                    name: {
                        "metrics": result["metrics"],
                        "state": result["state"],
                        "intent_execution": result["intent_execution"],
                        "integrity": result["integrity"],
                    }
                    for name, result in results.items()
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
        default=Path("/app/data/research/p0_main_board_forecast_drift_regime_discovery.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
