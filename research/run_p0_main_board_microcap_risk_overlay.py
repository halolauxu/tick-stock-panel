"""Run the frozen all-A risk overlay on the main-board micro-cap account."""

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

import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_microcap_escape as escape  # noqa: E402

STUDY_END = date(2026, 8, 28)
PRIMARY_CAPITAL = 200_000.0
CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
FROZEN_THRESHOLDS_SHA256 = "2e1f4425839c047c74243661d0d800850a61a2e6104bcff83c134d202750b2c4"


def _period_bounds(mode: str, last_date: date) -> tuple[date, date, str]:
    if mode == "validate":
        return date(2021, 1, 1), baseline.VALIDATION_END, "validation"
    if mode == "stress":
        return date(2024, 1, 1), last_date, "known_stress"
    raise ValueError(f"unsupported mode: {mode}")


def load_frozen_thresholds(path: Path) -> dict[str, float]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_THRESHOLDS_SHA256:
        raise ValueError(f"frozen threshold hash mismatch: {digest}")
    return escape.load_frozen_thresholds(path)


def load_delist_dates(data_dir: Path, symbols: list[str]) -> dict[str, date]:
    master = data_dir / "research" / "historical_stock_universe_all_a.parquet"
    if not master.is_file():
        raise ValueError("all-A PIT security master is required")
    rows = (
        pl.read_parquet(master)
        .with_columns(pl.col("delist_date").cast(pl.Date, strict=False))
        .filter(pl.col("symbol").is_in(symbols) & pl.col("delist_date").is_not_null())
        .select("symbol", "delist_date")
        .unique(subset=["symbol"], keep="last")
        .to_dicts()
    )
    return {row["symbol"]: row["delist_date"] for row in rows}


def evaluate_period(tiers: list[dict[str, Any]]) -> dict[str, Any]:
    primary = next(row for row in tiers if row["capital"] == PRIMARY_CAPITAL)
    metrics = primary["metrics"]
    execution = primary["execution"]
    integrity = primary["integrity"]
    checks = {
        "annualized_at_least_15pct": metrics["account_annualized"] >= 0.15,
        "annualized_excess_at_least_10pp": (metrics["annualized_excess"] >= 0.10),
        "max_drawdown_within_25pct": (metrics["account_max_drawdown"] >= -0.25),
        "at_least_two_positive_years": (metrics["positive_account_years"] >= 2),
        "buy_execution_at_least_80pct": (execution["buy"]["execution_rate"] >= 0.80),
        "sell_execution_at_least_80pct": (execution["sell"]["execution_rate"] >= 0.80),
        "no_unresolved_positions": (integrity["ending_unresolved_positions"] == 0),
        "cash_reconciled": (integrity["max_cash_reconciliation_error"] <= 0.01),
    }
    return {
        "passed": all(checks.values()),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def _summary_only(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"daily_equity", "orders"}}


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(
    data_dir: Path,
    output: Path,
    *,
    mode: str,
    thresholds_path: Path,
    end: date = STUDY_END,
) -> dict[str, Any]:
    load_end = baseline.VALIDATION_END if mode == "validate" else end
    thresholds = load_frozen_thresholds(thresholds_path)

    full_source = baseline.load_daily(data_dir, end=load_end)
    if full_source.is_empty():
        raise ValueError("no all-A daily data")
    all_dates = full_source.get_column("date").unique().sort().to_list()
    full_pit = baseline.attach_point_in_time_data(full_source, data_dir)
    del full_source
    gc.collect()
    full_panel = baseline.prepare_panel(full_pit)
    del full_pit
    gc.collect()
    features = escape.build_daily_features(full_panel)
    del full_panel
    gc.collect()
    alarms = escape.apply_alarms(features, thresholds)
    risk_by_open, decisions, switches = escape.build_risk_clock(alarms)
    del features, alarms
    gc.collect()

    main_source = main_board.filter_main_board(baseline.load_daily(data_dir, end=load_end))
    if main_source.is_empty():
        raise ValueError("no main-board daily data")
    main_rows = main_source.height
    main_symbols = main_source.get_column("symbol").n_unique()
    main_pit = baseline.attach_point_in_time_data(main_source, data_dir)
    del main_source
    gc.collect()
    main_panel = baseline.prepare_panel(main_pit)
    del main_pit
    gc.collect()
    weekly_candidates = account.build_signal_candidates(main_panel)
    observations = baseline.build_weekly_observations(main_panel)
    weekly_market = baseline.weekly_portfolios(observations).select("date", "period", "market_net")
    candidate_symbols = weekly_candidates.get_column("symbol").unique().to_list()
    del main_panel, observations
    gc.collect()

    start, finish, period = _period_bounds(mode, all_dates[-1])
    scoped_dates = [day for day in all_dates if start <= day <= finish]
    action_candidates, action_dates, naked_dates = escape.build_action_candidates(
        weekly_candidates,
        all_dates,
        risk_by_open,
        start=start,
        end=finish,
    )
    naked_candidates = weekly_candidates.filter(
        (pl.col("entry_date") >= pl.lit(start)) & (pl.col("entry_date") <= pl.lit(finish))
    )
    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=load_end)
    ).filter(pl.col("symbol").is_in(candidate_symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_dates = sorted(set(action_dates) | set(naked_dates))
    execution_grid = escape.build_execution_grid_for_dates(
        weekly_candidates, execution_dates, quotes
    )
    delist_dates = load_delist_dates(data_dir, candidate_symbols)

    raw_tiers = [
        escape.simulate_tier(
            capital=capital,
            period=period,
            action_candidates=action_candidates,
            action_dates=action_dates,
            naked_candidates=naked_candidates,
            naked_dates=naked_dates,
            execution_grid=execution_grid,
            quotes=quotes,
            scoped_dates=scoped_dates,
            weekly_market=weekly_market,
            delist_dates=delist_dates,
        )
        for capital in CAPITALS
    ]
    for result in raw_tiers:
        result["drawdown_episode"] = main_board.drawdown_episode(result["daily_equity"])
    decision = evaluate_period(raw_tiers)
    tiers = [
        result if result["capital"] == PRIMARY_CAPITAL else _summary_only(result)
        for result in raw_tiers
    ]
    scoped_decisions = [row for row in decisions if start <= row["action_date"] <= finish]
    scoped_switches = [row for row in switches if start <= row["action_date"] <= finish]
    payload = {
        "schema_version": "p0-main-board-microcap-risk-overlay-v1",
        "mode": mode,
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "risk_feature_scope": "all_a_pit_microcap_decile",
            "thresholds_sha256": FROZEN_THRESHOLDS_SHA256,
            "capital_ladder": list(CAPITALS),
            "primary_capital": PRIMARY_CAPITAL,
            "delisting_settlement": "zero_recovery_on_first_action_at_or_after_delist_date",
        },
        "data": {
            "first_loaded_date": all_dates[0],
            "last_loaded_date": all_dates[-1],
            "period_start": start,
            "period_end": finish,
            "period_trading_days": len(scoped_dates),
            "main_board_rows": main_rows,
            "main_board_symbols": main_symbols,
            "candidate_symbols": len(candidate_symbols),
            "candidate_board_counts": baseline.board_symbol_counts(
                pl.DataFrame({"symbol": candidate_symbols})
            ),
            "candidate_delist_dates": len(delist_dates),
        },
        "thresholds": thresholds,
        "risk": {
            "risk_off_opens": sum(not row["risk_on"] for row in scoped_decisions),
            "risk_on_opens": sum(row["risk_on"] for row in scoped_decisions),
            "switch_count": len(scoped_switches),
            "switches": scoped_switches,
            "decisions": scoped_decisions,
        },
        "capital_tiers": tiers,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": mode,
                "data": payload["data"],
                "risk": {
                    key: value
                    for key, value in payload["risk"].items()
                    if key not in {"switches", "decisions"}
                },
                "capital_tiers": [
                    {
                        key: value
                        for key, value in result.items()
                        if key
                        in {
                            "capital",
                            "metrics",
                            "naked_metrics",
                            "annualized_delta_vs_naked",
                            "execution",
                            "integrity",
                            "account",
                            "settlements",
                            "drawdown_episode",
                        }
                    }
                    for result in tiers
                ],
                "decision": decision,
                "output": str(output),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "stress"), required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=RESEARCH / "p0_microcap_escape_thresholds.json",
    )
    parser.add_argument("--end", type=date.fromisoformat, default=STUDY_END)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.output,
        mode=args.mode,
        thresholds_path=args.thresholds,
        end=args.end,
    )


if __name__ == "__main__":
    main()
