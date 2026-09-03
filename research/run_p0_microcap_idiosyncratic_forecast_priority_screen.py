"""Screen a frozen event-priority overlay on the main-board micro-cap account."""

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

import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_idiosyncratic_forecast_surprise_account as event  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

INITIAL_CASH = 200_000.0
EVENT_TARGET_POSITIONS = 5
EVENT_SIGNAL_LIFETIME = 10
PERIODS = {
    "development": (date(2014, 1, 1), date(2020, 12, 31)),
    "validation": (date(2021, 1, 1), date(2023, 12, 31)),
    "known_stress": (date(2024, 1, 1), date(2026, 8, 28)),
}


def _set_event_period(start: date, end: date) -> None:
    event.base.DEVELOPMENT_START = start
    event.base.DEVELOPMENT_END = end
    event.base.SIGNAL_LIFETIME_TRADING_DAYS = EVENT_SIGNAL_LIFETIME
    event.industry.DEVELOPMENT_START = start
    event.industry.DEVELOPMENT_END = end
    event.industry.forecast.DEVELOPMENT_START = start
    event.industry.forecast.DEVELOPMENT_END = end


def build_event_account(
    data_dir: Path, start: date, end: date
) -> tuple[pl.DataFrame, dict[str, Any]]:
    _set_event_period(start, end)
    raw = baseline.load_daily(data_dir, end=end).filter(pl.col("date") >= start)
    raw_source = main_board.filter_main_board(raw)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = baseline.prepare_panel(
        baseline.attach_point_in_time_data(raw_source, data_dir)
    )
    candidates, signal_audit = event.base.build_candidates(
        event.load_events(data_dir), panel, all_dates
    )
    del panel
    gc.collect()

    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    grid = daily.build_action_grid(candidates, quotes, all_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=EVENT_TARGET_POSITIONS,
        action_dates=all_dates,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    detail = {
        "signal_audit": signal_audit,
        "execution": account.execution_summary(simulation["orders"]),
        "intent_execution": event.regime.intent_execution_summary(
            simulation["orders"], all_dates
        ),
        "integrity": {
            **stale,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, account_daily),
    }
    return account_daily.select(
        "date", "equity", "cash", "position_value", "position_count"
    ), detail


def _compound(values: list[float]) -> float | None:
    if not values:
        return None
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result - 1.0


def _annualized(values: list[float]) -> float | None:
    total = _compound(values)
    if total is None or total <= -1.0 or not values:
        return None
    return (1.0 + total) ** (252.0 / len(values)) - 1.0


def _max_drawdown(equity: list[float]) -> float | None:
    if not equity:
        return None
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def combine_curves(
    event_daily: pl.DataFrame, core_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    core = pl.DataFrame(core_rows).with_columns(
        pl.col("date").cast(pl.Date, strict=False)
    )
    joined = (
        event_daily.rename(
            {
                "equity": "event_equity",
                "cash": "event_cash",
                "position_value": "event_position_value",
                "position_count": "event_position_count",
            }
        )
        .join(
            core.select(
                "date",
                pl.col("equity").alias("core_equity"),
            ),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    if joined.height < 2:
        raise ValueError("event and core curves do not overlap")

    rows = joined.to_dicts()
    combined_equity = INITIAL_CASH
    curve: list[dict[str, Any]] = [
        {
            "date": rows[0]["date"],
            "equity": combined_equity,
            "daily_return": 0.0,
            "event_invested_weight": 0.0,
            "core_weight": 1.0,
            "event_position_count": int(rows[0]["event_position_count"]),
        }
    ]
    event_returns: list[float] = []
    core_returns: list[float] = []
    combined_returns: list[float] = []
    invested_weights: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        previous_event_equity = float(previous["event_equity"])
        previous_core_equity = float(previous["core_equity"])
        if previous_event_equity <= 0 or previous_core_equity <= 0:
            raise ValueError("non-positive component equity")
        event_return = float(current["event_equity"]) / previous_event_equity - 1.0
        core_return = float(current["core_equity"]) / previous_core_equity - 1.0
        # Cash after the current open's event orders is available to the core sleeve.
        core_weight = min(
            1.0,
            max(0.0, float(current["event_cash"]) / previous_event_equity),
        )
        event_invested_weight = 1.0 - core_weight
        combined_return = event_return + core_weight * core_return
        combined_equity *= 1.0 + combined_return
        event_returns.append(event_return)
        core_returns.append(core_return)
        combined_returns.append(combined_return)
        invested_weights.append(event_invested_weight)
        curve.append(
            {
                "date": current["date"],
                "equity": combined_equity,
                "daily_return": combined_return,
                "event_invested_weight": event_invested_weight,
                "core_weight": core_weight,
                "event_position_count": int(current["event_position_count"]),
            }
        )

    def summarize(returns: list[float], equities: list[float]) -> dict[str, Any]:
        yearly = []
        positive_years = 0
        for year in range(curve[0]["date"].year, curve[-1]["date"].year + 1):
            values = [
                row["daily_return"] for row in curve[1:] if row["date"].year == year
            ]
            result = _compound(values)
            positive_years += int(result is not None and result > 0)
            yearly.append({"year": year, "return": result})
        return {
            "trading_days": len(returns),
            "total_return": _compound(returns),
            "annualized": _annualized(returns),
            "max_drawdown": _max_drawdown(equities),
            "positive_years": positive_years,
            "yearly": yearly,
        }

    combined = summarize(combined_returns, [float(row["equity"]) for row in curve])
    combined.update(
        {
            "mean_event_invested_weight": sum(invested_weights) / len(invested_weights),
            "active_event_days": sum(weight > 1e-9 for weight in invested_weights),
        }
    )
    core_summary = {
        "trading_days": len(core_returns),
        "total_return": _compound(core_returns),
        "annualized": _annualized(core_returns),
        "max_drawdown": _max_drawdown([float(row["core_equity"]) for row in rows]),
    }
    return curve, combined, core_summary


def evaluate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation = results["validation"]["combined"]
    stress = results["known_stress"]["combined"]
    stress_core = results["known_stress"]["core"]
    event_detail = results["known_stress"]["event"]
    yearly = {int(row["year"]): row["return"] for row in validation["yearly"]}
    intent = event_detail["intent_execution"]
    integrity = event_detail["integrity"]
    complete_round_trips = int(event_detail["account"].get("trade_count") or 0) // 2
    checks = {
        "all_periods_annualized_positive": all(
            (row["combined"].get("annualized") or -math.inf) > 0
            for row in results.values()
        ),
        "validation_all_years_positive": all(
            (yearly.get(year) or -math.inf) > 0 for year in (2021, 2022, 2023)
        ),
        "stress_annualized_improves_core_by_3pp": (
            (stress.get("annualized") or -math.inf)
            - (stress_core.get("annualized") or -math.inf)
            >= 0.03
        ),
        "stress_drawdown_improves_core_by_5pp": (
            (stress.get("max_drawdown") or -math.inf)
            - (stress_core.get("max_drawdown") or -math.inf)
            >= 0.05
        ),
        "stress_drawdown_within_40pct": (stress.get("max_drawdown") or -math.inf)
        >= -0.40,
        "event_buy_intent_execution_at_least_90pct": intent["buy"]["execution_rate"]
        >= 0.90,
        "event_sell_intent_execution_at_least_90pct": intent["sell"]["execution_rate"]
        >= 0.90,
        "event_no_unresolved_positions": integrity["ending_unresolved_positions"] == 0,
        "event_cash_reconciled": integrity["max_cash_reconciliation_error"] <= 0.01,
        "stress_at_least_50_event_round_trips": complete_round_trips >= 50,
    }
    return {
        "verdict": "PROMOTE_TO_UNIFIED_ACCOUNT"
        if all(checks.values())
        else "TERMINATE",
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "stress_annualized_increment": stress.get("annualized")
        - stress_core.get("annualized"),
        "stress_drawdown_increment": stress.get("max_drawdown")
        - stress_core.get("max_drawdown"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, core_result: Path, output: Path) -> dict[str, Any]:
    core_payload = json.loads(core_result.read_text(encoding="utf-8"))
    primary = core_payload["accounts"][str(int(INITIAL_CASH))]["periods"]
    results: dict[str, dict[str, Any]] = {}
    for period, (start, end) in PERIODS.items():
        event_daily, event_detail = build_event_account(data_dir, start, end)
        curve, combined, core = combine_curves(
            event_daily, primary[period]["daily_equity"]
        )
        results[period] = {
            "period": {"start": start, "end": end},
            "combined": combined,
            "core": core,
            "event": event_detail,
            "curve": curve,
        }
    decision = evaluate(results)
    payload = {
        "schema_version": "p0-microcap-idiosyncratic-forecast-priority-screen-v1",
        "contract_frozen": "2026-09-03",
        "assumptions": {
            "capital_cny": INITIAL_CASH,
            "event_target_positions": EVENT_TARGET_POSITIONS,
            "event_position_weight": 1.0 / EVENT_TARGET_POSITIONS,
            "event_signal_lifetime_trading_days": EVENT_SIGNAL_LIFETIME,
            "idle_event_cash": "swept_to_main_board_microcap_core",
            "screen_level": "audited_daily_curve_combination_not_unified_ledger",
        },
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
                "results": {
                    name: {
                        "combined": row["combined"],
                        "core": row["core"],
                        "event": row["event"],
                    }
                    for name, row in results.items()
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
        "--core-result",
        type=Path,
        default=Path("/app/data/research/p0_main_board_microcap_account_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/"
            "p0_microcap_idiosyncratic_forecast_priority_screen_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.core_result, args.output)


if __name__ == "__main__":
    main()
