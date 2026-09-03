"""Run the frozen unified event-priority and main-board micro-cap account."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_idiosyncratic_forecast_surprise_account as event  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_microcap_risk_overlay as risk  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_microcap_idiosyncratic_forecast_priority_screen as screen  # noqa: E402

INITIAL_CASH = 200_000.0
TOTAL_SLOTS = 20
MICROCAP_WEIGHT = 0.05
EVENT_SLOTS = 4
EVENT_WEIGHT = MICROCAP_WEIGHT * EVENT_SLOTS
MAX_EVENT_POSITIONS = TOTAL_SLOTS // EVENT_SLOTS
EVENT_FAMILY = "idiosyncratic_forecast"
MICROCAP_FAMILY = "main_board_microcap"


def build_daily_targets(
    microcap_candidates: pl.DataFrame,
    event_candidates: pl.DataFrame,
    all_dates: list[date],
) -> pl.DataFrame:
    micro_by_date = account._partition_rows(microcap_candidates, "entry_date")
    event_by_date = account._partition_rows(event_candidates, "entry_date")
    current_micro: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for day in all_dates:
        if day in micro_by_date:
            current_micro = micro_by_date[day].sort(["cap_rank", "symbol"]).to_dicts()
        event_rows = (
            event_by_date.get(day, pl.DataFrame())
            .sort(["cap_rank", "symbol"])
            .head(MAX_EVENT_POSITIONS)
            .to_dicts()
            if day in event_by_date
            else []
        )
        event_symbols = {row["symbol"] for row in event_rows}
        rank = 0
        for row in event_rows:
            rank += 1
            output.append(
                {
                    "date": row["date"],
                    "entry_date": day,
                    "symbol": row["symbol"],
                    "signal_amount": row["signal_amount"],
                    "cap_rank": rank,
                    "target_weight": EVENT_WEIGHT,
                    "family": EVENT_FAMILY,
                    "source_rank": row["cap_rank"],
                }
            )
        micro_slots = TOTAL_SLOTS - EVENT_SLOTS * len(event_rows)
        for row in current_micro:
            if row["symbol"] in event_symbols:
                continue
            if micro_slots <= 0:
                break
            rank += 1
            output.append(
                {
                    "date": row["date"],
                    "entry_date": day,
                    "symbol": row["symbol"],
                    "signal_amount": row["signal_amount"],
                    "cap_rank": rank,
                    "target_weight": MICROCAP_WEIGHT,
                    "family": MICROCAP_FAMILY,
                    "source_rank": row["cap_rank"],
                }
            )
            micro_slots -= 1
    return pl.DataFrame(output, infer_schema_length=None).sort(
        ["entry_date", "cap_rank", "symbol"]
    )


def _yearly(daily: pl.DataFrame) -> list[dict[str, Any]]:
    output = []
    for year in sorted(daily.get_column("date").dt.year().unique().to_list()):
        value = baseline._compound(
            daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        output.append({"year": year, "return": value})
    return output


def _metrics(daily: pl.DataFrame) -> dict[str, Any]:
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    total = baseline._compound(returns)
    yearly = _yearly(daily)
    return {
        "trading_days": daily.height,
        "total_return": total,
        "annualized": (
            (1.0 + total) ** (252.0 / len(returns)) - 1.0
            if total is not None and total > -1.0 and returns
            else None
        ),
        "max_drawdown": baseline._max_drawdown(returns),
        "positive_years": sum(
            row["return"] is not None and row["return"] > 0 for row in yearly
        ),
        "yearly": yearly,
        "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
        "mean_position_count": daily.get_column("position_count").mean(),
    }


def _family_execution(orders: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for family in (EVENT_FAMILY, MICROCAP_FAMILY):
        scoped = [row for row in orders if row.get("family") == family]
        by_side = {}
        for side in ("BUY", "SELL"):
            rows = [row for row in scoped if row["side"] == side]
            eligible = [row for row in rows if row["status"] != "PRETRADE_SKIPPED"]
            filled = sum(row["status"] == "FILLED" for row in eligible)
            by_side[side.lower()] = {
                "orders": len(eligible),
                "filled": filled,
                "execution_rate": filled / len(eligible) if eligible else 1.0,
                "rejections": dict(
                    sorted(
                        Counter(
                            row.get("reason")
                            for row in eligible
                            if row.get("reason") is not None
                        ).items()
                    )
                ),
            }
        output[family] = by_side
    return output


def run_period(
    data_dir: Path,
    start: date,
    end: date,
    *,
    event_gate_by_date: dict[date, bool] | None = None,
    event_admission_by_date: dict[date, bool] | None = None,
) -> dict[str, Any]:
    screen._set_event_period(start, end)
    raw = baseline.load_daily(data_dir, end=end).filter(pl.col("date") >= start)
    raw_source = main_board.filter_main_board(raw)
    all_dates = raw_source.get_column("date").unique().sort().to_list()
    panel = baseline.prepare_panel(
        baseline.attach_point_in_time_data(raw_source, data_dir)
    )
    microcap_candidates = account.build_signal_candidates(panel)
    event_candidates, event_audit = event.base.build_candidates(
        event.load_events(data_dir), panel, all_dates
    )
    ungated_event_rows = event_candidates.height
    ungated_event_days = event_candidates.get_column("entry_date").n_unique()
    if event_gate_by_date is not None:
        enabled_dates = [day for day, enabled in event_gate_by_date.items() if enabled]
        event_candidates = event_candidates.filter(
            pl.col("entry_date").is_in(enabled_dates)
        )
    if event_admission_by_date is not None and event_candidates.height:
        admitted_dates = [
            day for day, enabled in event_admission_by_date.items() if enabled
        ]
        admissions = (
            event_candidates.group_by("symbol", "date")
            .agg(pl.col("entry_date").min().alias("initial_entry_date"))
            .filter(pl.col("initial_entry_date").is_in(admitted_dates))
            .select("symbol", "date")
        )
        event_candidates = event_candidates.join(
            admissions, on=["symbol", "date"], how="inner"
        )
    event_audit = {
        **event_audit,
        "ungated_daily_candidate_rows": ungated_event_rows,
        "ungated_active_account_days": ungated_event_days,
        "gated_daily_candidate_rows": event_candidates.height,
        "gated_active_account_days": (
            event_candidates.get_column("entry_date").n_unique()
            if event_candidates.height
            else 0
        ),
        "gate_mode": (
            "daily_holding"
            if event_gate_by_date is not None
            else "initial_entry_admission"
            if event_admission_by_date is not None
            else "none"
        ),
    }
    targets = build_daily_targets(microcap_candidates, event_candidates, all_dates)
    target_counts = (
        targets.group_by("family")
        .agg(
            pl.len().alias("rows"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("entry_date").n_unique().alias("days"),
        )
        .sort("family")
        .to_dicts()
    )
    del panel, microcap_candidates, event_candidates
    gc.collect()

    symbols = targets.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(
            raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir
        )
    )
    execution_grid = account.build_execution_grid(targets, quotes)
    delist_dates = risk.load_delist_dates(data_dir, symbols)
    simulation = account.simulate_account(
        targets,
        execution_grid,
        initial_cash=INITIAL_CASH,
        target_positions=TOTAL_SLOTS,
        action_dates=all_dates,
        delist_dates=delist_dates,
        settle_only_after_delist_date=True,
        candidate_weight_column="target_weight",
    )
    daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    family_execution = _family_execution(simulation["orders"])
    event_round_trips = family_execution[EVENT_FAMILY]["sell"]["filled"]
    return {
        "period": {"start": start, "end": end},
        "data": {
            "event": event_audit,
            "daily_targets": target_counts,
            "target_rows": targets.height,
            "target_symbols": targets.get_column("symbol").n_unique(),
        },
        "metrics": _metrics(daily),
        "execution": account.execution_summary(simulation["orders"]),
        "family_execution": family_execution,
        "event_round_trips": event_round_trips,
        "integrity": {
            **stale,
            "ending_positions": len(simulation["ending_positions"]),
            "delist_write_offs": len(simulation["settlements"]),
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily),
        "daily_equity": daily.select(
            "date",
            "equity",
            "cash",
            "position_value",
            "position_count",
            "stale_positions",
            "cash_ratio",
        ).to_dicts(),
        "orders": simulation["orders"],
        "settlements": simulation["settlements"],
    }


def _core_metrics(core_payload: dict[str, Any], period: str) -> dict[str, Any]:
    row = core_payload["accounts"][str(int(INITIAL_CASH))]["periods"][period]
    metrics = row["metrics"]
    return {
        "annualized": metrics["account_annualized"],
        "max_drawdown": metrics["account_max_drawdown"],
        "yearly": metrics["yearly"],
    }


def evaluate(
    results: dict[str, dict[str, Any]], core: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    validation = results["validation"]
    stress = results["known_stress"]
    validation_yearly = {
        row["year"]: row["return"] for row in validation["metrics"]["yearly"]
    }
    stress_metrics = stress["metrics"]
    stress_core = core["known_stress"]
    checks = {
        "validation_annualized_at_least_30pct": (
            validation["metrics"].get("annualized") or -math.inf
        )
        >= 0.30,
        "validation_all_years_positive": all(
            (validation_yearly.get(year) or -math.inf) > 0
            for year in (2021, 2022, 2023)
        ),
        "validation_drawdown_within_25pct": (
            validation["metrics"].get("max_drawdown") or -math.inf
        )
        >= -0.25,
        "stress_annualized_at_least_15pct": (
            stress_metrics.get("annualized") or -math.inf
        )
        >= 0.15,
        "stress_annualized_loss_vs_core_within_5pp": (
            (stress_metrics.get("annualized") or -math.inf)
            - (stress_core.get("annualized") or -math.inf)
            >= -0.05
        ),
        "stress_drawdown_within_35pct": (
            stress_metrics.get("max_drawdown") or -math.inf
        )
        >= -0.35,
        "stress_drawdown_improves_core_by_10pp": (
            (stress_metrics.get("max_drawdown") or -math.inf)
            - (stress_core.get("max_drawdown") or -math.inf)
            >= 0.10
        ),
        "stress_at_least_two_positive_years": stress_metrics["positive_years"] >= 2,
        "stress_buy_execution_at_least_80pct": stress["execution"]["buy"][
            "execution_rate"
        ]
        >= 0.80,
        "stress_sell_execution_at_least_80pct": stress["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.80,
        "stress_no_unresolved_positions": stress["integrity"][
            "ending_unresolved_positions"
        ]
        == 0,
        "stress_cash_reconciled": stress["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
        "stress_at_least_50_event_round_trips": stress["event_round_trips"] >= 50,
    }
    return {
        "verdict": "FORWARD_ELIGIBLE" if all(checks.values()) else "TERMINATE",
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "stress_annualized_increment": stress_metrics.get("annualized")
        - stress_core.get("annualized"),
        "stress_drawdown_increment": stress_metrics.get("max_drawdown")
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
    core = {period: _core_metrics(core_payload, period) for period in screen.PERIODS}
    results = {
        period: run_period(data_dir, start, end)
        for period, (start, end) in screen.PERIODS.items()
    }
    decision = evaluate(results, core)
    payload = {
        "schema_version": "p0-microcap-idiosyncratic-forecast-unified-account-v1",
        "contract_frozen": "2026-09-03",
        "assumptions": {
            "capital_cny": INITIAL_CASH,
            "total_slots": TOTAL_SLOTS,
            "microcap_slot_weight": MICROCAP_WEIGHT,
            "event_slots_per_position": EVENT_SLOTS,
            "event_position_weight": EVENT_WEIGHT,
            "maximum_event_positions": MAX_EVENT_POSITIONS,
            "event_signal_lifetime_trading_days": screen.EVENT_SIGNAL_LIFETIME,
            "single_cash_order_and_position_ledger": True,
        },
        "core": core,
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
                        key: value
                        for key, value in row.items()
                        if key not in {"daily_equity", "orders"}
                    }
                    for name, row in results.items()
                },
                "core": core,
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
            "p0_microcap_idiosyncratic_forecast_unified_account_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.core_result, args.output)


if __name__ == "__main__":
    main()
