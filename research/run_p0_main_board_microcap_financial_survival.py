"""Run the frozen main-board micro-cap financial-survival study."""

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

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
MAX_FINANCIAL_AGE_DAYS = 550
MAX_DEBT_RATIO = 0.70
MAX_GOODWILL_RATIO = 0.20
STAGES = {
    "development": (date(2014, 1, 1), baseline.DEVELOPMENT_END),
    "validation": (date(2021, 1, 1), baseline.VALIDATION_END),
    "known_stress": (date(2024, 1, 1), date(2026, 8, 28)),
}


def attach_survival_filter(
    candidates: pl.DataFrame, snapshots: pl.DataFrame
) -> pl.DataFrame:
    return (
        candidates.sort(["symbol", "date"])
        .join_asof(
            snapshots.sort(["symbol", "financial_available_date"]),
            left_on="date",
            right_on="financial_available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("date") - pl.col("financial_available_date"))
            .dt.total_days()
            .alias("financial_age_days")
        )
        .filter(
            pl.col("financial_age_days").is_between(
                0, MAX_FINANCIAL_AGE_DAYS, closed="both"
            )
            & (pl.col("total_assets") > 0)
            & (pl.col("total_equity") > 0)
            & (pl.col("net_income_attributable") > 0)
            & (pl.col("net_operating_cash_flow") > 0)
            & (pl.col("debt_ratio") <= MAX_DEBT_RATIO)
            & (pl.col("goodwill_ratio") <= MAX_GOODWILL_RATIO)
        )
        .select(candidates.columns)
        .sort(["entry_date", "cap_rank", "symbol"])
    )


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"daily_equity", "rebalance_snapshots", "orders"}
    }


def _quality_funnel(control: pl.DataFrame, candidate: pl.DataFrame) -> dict[str, Any]:
    control_days = control["entry_date"].n_unique()
    by_day = (
        candidate.group_by("entry_date")
        .agg(pl.len().alias("eligible"))
        .sort("entry_date")
    )
    return {
        "control_signal_rows": control.height,
        "quality_signal_rows": candidate.height,
        "rebalance_days": control_days,
        "days_with_quality_candidates": by_day.height,
        "mean_quality_candidates": by_day["eligible"].mean() if by_day.height else 0.0,
        "minimum_quality_candidates": by_day["eligible"].min() if by_day.height else 0,
        "days_with_at_least_20": by_day.filter(pl.col("eligible") >= 20).height,
    }


def run_stage(
    data_dir: Path,
    snapshots: pl.DataFrame,
    stage: str,
) -> dict[str, Any]:
    start, end = STAGES[stage]
    source = main_board.filter_main_board(baseline.load_daily(data_dir, end=end))
    all_dates = [
        day for day in source["date"].unique().sort().to_list() if start <= day <= end
    ]
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    control_all = account.build_signal_candidates(panel)
    control = control_all.filter(
        pl.col("entry_date").is_between(start, end, closed="both")
    )
    candidate = attach_survival_filter(control, snapshots)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    symbols = control["symbol"].unique().to_list()
    del panel, observations, control_all
    gc.collect()

    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=end)
    ).filter(pl.col("symbol").is_in(symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(control, quotes)

    accounts: dict[str, Any] = {}
    for capital in CAPITALS:
        control_result = account.run_independent_account(
            stage,
            control,
            execution_grid,
            quotes,
            all_dates,
            weekly_market,
            initial_cash=capital,
        )
        candidate_result = account.run_independent_account(
            stage,
            candidate,
            execution_grid,
            quotes,
            all_dates,
            weekly_market,
            initial_cash=capital,
        )
        accounts[str(int(capital))] = {
            "initial_cash": capital,
            "control": _summary(control_result),
            "financial_survival": _summary(candidate_result),
        }
    return {
        "stage": stage,
        "period": {"start": start, "end": end},
        "trading_days": len(all_dates),
        "funnel": _quality_funnel(control, candidate),
        "accounts": accounts,
    }


def _common_checks(candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "buy_execution_at_least_80pct": candidate["execution"]["buy"]["execution_rate"]
        >= 0.80,
        "sell_execution_at_least_80pct": candidate["execution"]["sell"][
            "execution_rate"
        ]
        >= 0.80,
        "no_unresolved_positions": candidate["integrity"]["ending_unresolved_positions"]
        == 0,
        "cash_reconciled": candidate["integrity"]["max_cash_reconciliation_error"]
        <= 0.01,
    }


def evaluate(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    account_row = result["accounts"][str(int(PRIMARY_CAPITAL))]
    control = account_row["control"]
    candidate = account_row["financial_survival"]
    metrics = candidate["metrics"]
    yearly = {row["year"]: row["account_return"] for row in metrics["yearly"]}
    common = _common_checks(candidate)
    if stage == "development":
        checks = {
            "annualized_at_least_control_plus_5pp": (
                metrics["account_annualized"]
                >= control["metrics"]["account_annualized"] + 0.05
            ),
            "drawdown_improves_at_least_10pp": (
                metrics["account_max_drawdown"]
                >= control["metrics"]["account_max_drawdown"] + 0.10
            ),
            "at_least_6_positive_years": metrics["positive_account_years"] >= 6,
            **common,
        }
    elif stage == "validation":
        checks = {
            "annualized_at_least_30pct": metrics["account_annualized"] >= 0.30,
            "all_2021_2023_years_positive": all(
                (yearly.get(year) or 0.0) > 0 for year in range(2021, 2024)
            ),
            "max_drawdown_within_25pct": metrics["account_max_drawdown"] >= -0.25,
            **common,
        }
    else:
        checks = {
            "all_2024_2026_years_positive": all(
                (yearly.get(year) or 0.0) > 0 for year in range(2024, 2027)
            ),
            "2026_return_above_30pct": (yearly.get(2026) or -99.0) > 0.30,
            "max_drawdown_within_25pct": metrics["account_max_drawdown"] >= -0.25,
            **common,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit_path = (
        data_dir / "research" / "p0_main_board_microcap_financial_survival_data.json"
    )
    data_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if data_audit.get("status") != "DATA_QUALIFIED":
        raise ValueError("financial survival data is not qualified")
    snapshots = pl.read_parquet(data_audit["artifact"])
    stages: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for stage in STAGES:
        if stage == "validation" and not decisions["development"]["passed"]:
            stages[stage] = {"status": "NOT_READ_AFTER_DEVELOPMENT_FAILURE"}
            continue
        if stage == "known_stress" and not decisions.get("validation", {}).get(
            "passed", False
        ):
            stages[stage] = {"status": "NOT_READ_AFTER_VALIDATION_FAILURE"}
            continue
        result = run_stage(data_dir, snapshots, stage)
        decision = evaluate(stage, result)
        result["decision"] = decision
        stages[stage] = result
        decisions[stage] = decision
    passed = all(decisions.get(stage, {}).get("passed", False) for stage in STAGES)
    payload = {
        "schema_version": "p0-main-board-microcap-financial-survival-v1",
        "contract_frozen": "2026-09-03",
        "data_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "target_positions": account.TARGET_POSITIONS,
            "max_financial_age_days": MAX_FINANCIAL_AGE_DAYS,
            "max_debt_ratio": MAX_DEBT_RATIO,
            "max_goodwill_ratio": MAX_GOODWILL_RATIO,
            "requires_positive_net_income": True,
            "requires_positive_operating_cash_flow": True,
            "capital_ladder": list(CAPITALS),
        },
        "stages": stages,
        "decision": (
            "FORWARD_ELIGIBLE" if passed else "TERMINATE_FINANCIAL_SURVIVAL_FILTER"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    development = stages["development"]
    primary = development["accounts"][str(int(PRIMARY_CAPITAL))]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "data_audit_sha256": payload["data_audit_sha256"],
                "development_funnel": development["funnel"],
                "development_gate": development["decision"],
                "primary_account": primary,
                "stage_status": {
                    stage: result.get("status", "READ")
                    for stage, result in stages.items()
                },
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_main_board_microcap_financial_survival_v1.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
