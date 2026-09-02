"""Run frozen double-low CB accounts with a CNY 100 redemption floor."""

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

import run_p0_convertible_bond_double_low_conservative as conservative  # noqa: E402
import run_p0_convertible_bond_double_low_development as cbbase  # noqa: E402
import run_p0_main_board_microcap_account as account_report  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

FACE_VALUE = 100.0


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
        delist_settlement_status="CB_DELISTED_FACE_VALUE_RECOVERY",
        settle_only_after_delist_date=True,
        delist_recovery_per_raw_share=FACE_VALUE,
    )
    daily, integrity = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=initial_cash
    )
    returns = daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    positive_years = 0
    for year in range(
        cbbase.DEVELOPMENT_START.year, cbbase.DEVELOPMENT_END.year + 1
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
    raw_daily = pl.read_parquet(root / "daily.parquet")
    panel = cbbase.prepare_panel(raw_daily, master)
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
    symbols = candidates.get_column("symbol").unique().to_list()
    delist_dates = conservative.load_delist_dates(master, symbols)
    early_zero_remain = master.filter(
        pl.col("symbol").is_in(symbols)
        & pl.col("delist_date").is_not_null()
        & pl.col("maturity_date").is_not_null()
        & (pl.col("delist_date") < pl.col("maturity_date"))
        & (pl.col("remain_size") == 0)
    )
    eligible_delist_dates = conservative.load_delist_dates(
        early_zero_remain, symbols
    )
    tiers = [
        simulate(
            candidates,
            panel,
            all_dates,
            action_dates,
            capital,
            eligible_delist_dates,
        )
        for capital in conservative.CAPITALS
    ]
    benchmark = cbbase.benchmark_metrics(panel)
    decision = conservative.evaluate(tiers, benchmark)
    payload = {
        "schema_version": "p0-convertible-bond-face-value-settlement-v1",
        "contract_frozen": "2026-09-02",
        "period": {
            "start": cbbase.DEVELOPMENT_START,
            "end": cbbase.DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "capital_ladder": list(conservative.CAPITALS),
            "primary_capital": conservative.PRIMARY_CAPITAL,
            "selection": "unchanged_p0_double_low",
            "settlement": "cny_100_face_value_first_action_after_delist",
            "accrued_interest": 0.0,
        },
        "data": {
            "signal_rows": candidates.height,
            "signal_symbols": len(symbols),
            "candidate_delist_dates": len(delist_dates),
            "eligible_early_zero_remain_delistments": (
                len(eligible_delist_dates)
            ),
            "scheduled_rebalances": len(action_dates),
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
            "/app/data/research/p0_convertible_bond_face_value_settlement_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
