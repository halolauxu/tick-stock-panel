"""Run the frozen P0-B4 Shanghai/Shenzhen main-board micro-cap study."""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
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

STUDY_END = date(2026, 8, 28)
CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
MAIN_BOARD_PATTERN = (
    r"^(?:(?:000|001|002|003)\d{3}\.SZ|"
    r"(?:600|601|603|605)\d{3}\.SH)$"
)


def is_main_board_symbol(symbol: str) -> bool:
    normalized = str(symbol or "").strip().upper()
    return re.fullmatch(MAIN_BOARD_PATTERN, normalized) is not None


def filter_main_board(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("symbol").str.contains(MAIN_BOARD_PATTERN))


def drawdown_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "drawdown": None,
            "peak_date": None,
            "peak_equity": None,
            "trough_date": None,
            "trough_equity": None,
            "recovery_date": None,
        }
    peak_equity = float(rows[0]["equity"])
    peak_date = rows[0]["date"]
    worst = {
        "drawdown": 0.0,
        "peak_date": peak_date,
        "peak_equity": peak_equity,
        "trough_date": peak_date,
        "trough_equity": peak_equity,
    }
    for row in rows:
        equity = float(row["equity"])
        if equity > peak_equity:
            peak_equity = equity
            peak_date = row["date"]
        drawdown = equity / peak_equity - 1.0
        if drawdown < worst["drawdown"]:
            worst = {
                "drawdown": drawdown,
                "peak_date": peak_date,
                "peak_equity": peak_equity,
                "trough_date": row["date"],
                "trough_equity": equity,
            }
    worst["recovery_date"] = next(
        (
            row["date"]
            for row in rows
            if row["date"] > worst["trough_date"]
            and float(row["equity"]) >= worst["peak_equity"]
        ),
        None,
    )
    return worst


def _period_checks(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["metrics"]
    execution = result["execution"]
    integrity = result["integrity"]
    return {
        "annualized_at_least_15pct": (
            metrics.get("account_annualized") or -99.0
        ) >= 0.15,
        "annualized_excess_at_least_10pp": (
            metrics.get("annualized_excess") or -99.0
        ) >= 0.10,
        "max_drawdown_within_25pct": (
            metrics.get("account_max_drawdown") or -99.0
        ) >= -0.25,
        "at_least_two_positive_years": (
            metrics.get("positive_account_years") or 0
        ) >= 2,
        "buy_execution_at_least_80pct": (
            execution["buy"]["execution_rate"] >= 0.80
        ),
        "sell_execution_at_least_80pct": (
            execution["sell"]["execution_rate"] >= 0.80
        ),
        "no_unresolved_positions": (
            integrity["ending_unresolved_positions"] == 0
        ),
        "cash_reconciled": (
            integrity["max_cash_reconciliation_error"] <= 0.01
        ),
    }


def evaluate_account(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = {
        period: _period_checks(results[period])
        for period in ("validation", "known_stress")
    }
    failures = [
        f"{period}:{name}"
        for period, period_checks in checks.items()
        for name, passed in period_checks.items()
        if not passed
    ]
    return_checks = [
        checks[period]["annualized_at_least_15pct"]
        for period in ("validation", "known_stress")
    ]
    if not failures:
        verdict = "FORWARD_ELIGIBLE"
    elif not all(return_checks):
        verdict = "TERMINATE"
    else:
        verdict = "RESEARCH_ONLY"
    return {
        "verdict": verdict,
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def _summary_only(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"daily_equity", "rebalance_snapshots", "orders"}
    }


def _load_all_board_comparison(
    path: Path,
    main_board_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path), "periods": {}}
    source = json.loads(path.read_text(encoding="utf-8"))
    periods = {}
    for period in ("development", "validation", "known_stress"):
        old = source["independent_accounts"][period]
        new = main_board_results[period]
        periods[period] = {
            "main_board_annualized": new["metrics"]["account_annualized"],
            "all_board_annualized": old["metrics"]["account_annualized"],
            "annualized_difference": (
                new["metrics"]["account_annualized"]
                - old["metrics"]["account_annualized"]
            ),
            "main_board_max_drawdown": new["metrics"][
                "account_max_drawdown"
            ],
            "all_board_max_drawdown": old["metrics"][
                "account_max_drawdown"
            ],
            "drawdown_difference": (
                new["metrics"]["account_max_drawdown"]
                - old["metrics"]["account_max_drawdown"]
            ),
            "main_board_ending_equity": new["account"]["ending_equity"],
            "all_board_ending_equity": old["account"]["ending_equity"],
        }
    return {"available": True, "path": str(path), "periods": periods}


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
    end: date = STUDY_END,
    all_board_result: Path | None = None,
) -> dict[str, Any]:
    source = filter_main_board(baseline.load_daily(data_dir, end=end))
    if source.is_empty():
        raise ValueError("no main-board daily data")
    source_rows = source.height
    source_symbols = source.get_column("symbol").n_unique()
    source_board_counts = baseline.board_symbol_counts(source)
    all_dates = source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    candidates = account.build_signal_candidates(panel)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    del panel, observations
    gc.collect()

    source_quotes = filter_main_board(
        baseline.load_daily(data_dir, end=end)
    ).filter(pl.col("symbol").is_in(candidate_symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(candidates, quotes)

    accounts: dict[str, Any] = {}
    raw_300k: dict[str, dict[str, Any]] | None = None
    for initial_cash in CAPITALS:
        period_results: dict[str, dict[str, Any]] = {}
        for period in ("development", "validation", "known_stress"):
            result = account.run_independent_account(
                period,
                candidates,
                execution_grid,
                quotes,
                all_dates,
                weekly_market,
                initial_cash=initial_cash,
            )
            result["drawdown_episode"] = drawdown_episode(
                result["daily_equity"]
            )
            period_results[period] = (
                result
                if initial_cash == PRIMARY_CAPITAL
                else _summary_only(result)
            )
        key = str(int(initial_cash))
        accounts[key] = {
            "initial_cash": initial_cash,
            "periods": period_results,
            "decision": evaluate_account(period_results),
        }
        if initial_cash == 300_000.0:
            raw_300k = period_results

    comparison_path = all_board_result or (
        data_dir
        / "research"
        / "p0_microcap_account_v3_first_day_reconciled.json"
    )
    payload = {
        "schema_version": "p0-main-board-microcap-account-v1",
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "main_board_pattern": MAIN_BOARD_PATTERN,
            "study_end": end,
            "capital_ladder": list(CAPITALS),
            "primary_capital": PRIMARY_CAPITAL,
            "target_positions": account.TARGET_POSITIONS,
            "lot_size": account.LOT_SIZE,
            "daily_participation": baseline.DAILY_PARTICIPATION,
            "signal": "weekly_main_board_pit_market_cap_bottom_decile",
            "execution": "next_trade_day_open_sells_before_buys",
        },
        "data": {
            "first_date": all_dates[0],
            "last_date": all_dates[-1],
            "trading_days": len(all_dates),
            "source_rows": source_rows,
            "source_symbols": source_symbols,
            "source_board_counts": source_board_counts,
            "candidate_board_counts": (
                baseline.board_symbol_counts(
                    pl.DataFrame({"symbol": candidate_symbols})
                )
                if candidate_symbols
                else {}
            ),
            "candidate_symbols": len(candidate_symbols),
            "signal_rows": candidates.height,
            "rebalance_days": candidates.get_column("entry_date").n_unique(),
        },
        "accounts": accounts,
        "all_board_300k_comparison": _load_all_board_comparison(
            comparison_path,
            raw_300k or {},
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "data": payload["data"],
                "accounts": {
                    capital: {
                        "periods": {
                            period: {
                                "metrics": result["metrics"],
                                "execution": result["execution"],
                                "integrity": result["integrity"],
                                "account": result["account"],
                                "drawdown_episode": result[
                                    "drawdown_episode"
                                ],
                            }
                            for period, result in row["periods"].items()
                        },
                        "decision": row["decision"],
                    }
                    for capital, row in accounts.items()
                },
                "all_board_300k_comparison": payload[
                    "all_board_300k_comparison"
                ],
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_main_board_microcap_account_v1.json"
        ),
    )
    parser.add_argument("--end", type=date.fromisoformat, default=STUDY_END)
    parser.add_argument("--all-board-result", type=Path)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.output,
        end=args.end,
        all_board_result=args.all_board_result,
    )


if __name__ == "__main__":
    main()
