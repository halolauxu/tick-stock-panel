"""Run the frozen double-low CB account with zero-recovery delistments."""
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
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_convertible_bond_double_low_development as cbbase  # noqa: E402
import run_p0_main_board_microcap_account as account_report  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = CAPITALS[0]


def load_delist_dates(
    master: pl.DataFrame, symbols: list[str]
) -> dict[str, date]:
    rows = (
        master.with_columns(
            pl.col("delist_date").cast(pl.Date, strict=False)
        )
        .filter(
            pl.col("symbol").is_in(symbols)
            & pl.col("delist_date").is_not_null()
        )
        .select("symbol", "delist_date")
        .unique(subset=["symbol"], keep="last")
        .to_dicts()
    )
    return {row["symbol"]: row["delist_date"] for row in rows}


def simulate(
    candidates: pl.DataFrame,
    panel: pl.DataFrame,
    all_dates: list[date],
    action_dates: list[date],
    initial_cash: float,
    delist_dates: dict[str, date],
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = cbbase.prepare_quotes(panel, symbols)
    grid = cbbase.build_execution_grid(candidates, quotes, action_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=initial_cash,
        target_positions=cbbase.TARGET_POSITIONS,
        action_dates=action_dates,
        stamp_tax_rate=0.0,
        lot_size=cbbase.LOT_SIZE,
        delist_dates=delist_dates,
        delist_settlement_status="CB_DELISTED_ZERO_RECOVERY",
        settle_only_after_delist_date=True,
    )
    daily, integrity = account.build_daily_equity(
        simulation,
        quotes,
        all_dates,
        initial_cash=initial_cash,
    )
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(
        cbbase.DEVELOPMENT_START.year,
        cbbase.DEVELOPMENT_END.year + 1,
    ):
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
        "capital": initial_cash,
        "metrics": {
            "trading_days": daily.height,
            "total_return": baseline._compound(returns),
            "annualized": cbbase.annualized(returns),
            "max_drawdown": baseline._max_drawdown(returns),
            "positive_years": positive_years,
            "mean_cash_ratio": daily.get_column("cash_ratio").mean(),
            "yearly": yearly,
        },
        "execution": account.execution_summary(simulation["orders"]),
        "integrity": {
            **integrity,
            "max_cash_reconciliation_error": simulation[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(simulation, daily),
        "settlements": simulation["settlements"],
        "drawdown_episode": account_report.drawdown_episode(
            daily.select("date", "equity").to_dicts()
        ),
        "orders": simulation["orders"],
        "daily_equity": daily.select(
            "date",
            "equity",
            "cash",
            "position_value",
            "position_count",
            "stale_positions",
            "cash_ratio",
        ).to_dicts(),
    }


def evaluate(
    tiers: list[dict[str, Any]], benchmark: dict[str, Any]
) -> dict[str, Any]:
    primary = next(row for row in tiers if row["capital"] == PRIMARY_CAPITAL)
    metrics = primary["metrics"]
    execution = primary["execution"]
    integrity = primary["integrity"]
    benchmark_annualized = benchmark.get("annualized")
    annualized = metrics.get("annualized")
    excess = (
        annualized - benchmark_annualized
        if annualized is not None and benchmark_annualized is not None
        else -math.inf
    )
    checks = {
        "annualized_at_least_15pct": (annualized or -math.inf) >= 0.15,
        "excess_at_least_5pp": excess >= 0.05,
        "max_drawdown_within_25pct": (
            metrics.get("max_drawdown") or -math.inf
        )
        >= -0.25,
        "at_least_three_positive_years": metrics["positive_years"] >= 3,
        "buy_execution_at_least_90pct": (
            execution["buy"]["execution_rate"] >= 0.90
        ),
        "sell_execution_at_least_90pct": (
            execution["sell"]["execution_rate"] >= 0.90
        ),
        "no_unresolved_positions": (
            integrity["ending_unresolved_positions"] == 0
        ),
        "cash_reconciled": (
            integrity["max_cash_reconciliation_error"] <= 0.01
        ),
    }
    capacity = {
        str(int(row["capital"])): {
            "buy_execution_at_least_90pct": (
                row["execution"]["buy"]["execution_rate"] >= 0.90
            ),
            "sell_execution_at_least_90pct": (
                row["execution"]["sell"]["execution_rate"] >= 0.90
            ),
            "no_unresolved_positions": (
                row["integrity"]["ending_unresolved_positions"] == 0
            ),
            "cash_reconciled": (
                row["integrity"]["max_cash_reconciliation_error"] <= 0.01
            ),
        }
        for row in tiers
    }
    passed = all(checks.values())
    return {
        "verdict": "PROMOTE_TO_VALIDATION_DATA" if passed else "TERMINATE",
        "passed": passed,
        "annualized_excess": excess,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "capacity_checks": capacity,
        "validation_read": False,
        "known_stress_read": False,
    }


def _summary_only(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"orders", "daily_equity"}
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "convertible_bond"
    master = pl.read_parquet(root / "master.parquet")
    daily = pl.read_parquet(root / "daily.parquet")
    panel = cbbase.prepare_panel(daily, master)
    schedule = cbbase.weekly_schedule(panel)
    candidates = cbbase.build_candidates(panel, schedule)
    action_dates = schedule.get_column("entry_date").to_list()
    all_dates = (
        panel.filter(
            pl.col("date").is_between(
                cbbase.DEVELOPMENT_START,
                cbbase.DEVELOPMENT_END,
                closed="both",
            )
        )
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )
    candidate_symbols = candidates.get_column("symbol").unique().to_list()
    delist_dates = load_delist_dates(master, candidate_symbols)
    raw_tiers = [
        simulate(
            candidates,
            panel,
            all_dates,
            action_dates,
            capital,
            delist_dates,
        )
        for capital in CAPITALS
    ]
    benchmark = cbbase.benchmark_metrics(panel)
    decision = evaluate(raw_tiers, benchmark)
    tiers = [
        row if row["capital"] == PRIMARY_CAPITAL else _summary_only(row)
        for row in raw_tiers
    ]
    payload = {
        "schema_version": "p0-convertible-bond-double-low-conservative-v1",
        "contract_frozen": "2026-09-02",
        "period": {
            "start": cbbase.DEVELOPMENT_START,
            "end": cbbase.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "capital_ladder": list(CAPITALS),
            "primary_capital": PRIMARY_CAPITAL,
            "selection": "unchanged_p0_double_low",
            "settlement": "zero_recovery_first_action_after_delist_date",
        },
        "data": {
            "master_symbols": master.height,
            "daily_symbols": daily.get_column("symbol").n_unique(),
            "signal_rows": candidates.height,
            "signal_symbols": len(candidate_symbols),
            "candidate_delist_dates": len(delist_dates),
            "scheduled_rebalances": len(action_dates),
            "active_rebalances": candidates.get_column(
                "entry_date"
            ).n_unique(),
        },
        "benchmark": benchmark,
        "capital_tiers": tiers,
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
                "benchmark": benchmark,
                "capital_tiers": [
                    _summary_only(row) for row in raw_tiers
                ],
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
            "/app/data/research/p0_convertible_bond_double_low_conservative.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
