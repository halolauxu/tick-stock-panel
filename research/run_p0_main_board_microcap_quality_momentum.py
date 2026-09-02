"""Run the frozen main-board micro-cap quality-momentum composite."""

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
MOMENTUM_OBSERVATIONS = 60
EXPECTED_AUDIT_SHA256 = "0fc106a8688f7e8cf06a8ba5aaadd0f92414d4f3f473c5c1e1ccdaeede9d9ac5"
STAGES = {
    "development": (date(2014, 1, 1), baseline.DEVELOPMENT_END),
    "validation": (date(2021, 1, 1), baseline.VALIDATION_END),
    "known_stress": (date(2024, 1, 1), date(2026, 8, 28)),
}


def attach_quality_momentum(
    candidates: pl.DataFrame,
    panel: pl.DataFrame,
    snapshots: pl.DataFrame,
) -> pl.DataFrame:
    momentum = (
        panel.sort(["symbol", "date"])
        .with_columns(
            (
                pl.col("close") / pl.col("close").shift(MOMENTUM_OBSERVATIONS).over("symbol") - 1.0
            ).alias("momentum_60d")
        )
        .select("symbol", "date", "momentum_60d")
    )
    work = (
        candidates.join(momentum, on=["symbol", "date"], how="left")
        .sort(["symbol", "date"])
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
            pl.col("financial_age_days").is_between(0, MAX_FINANCIAL_AGE_DAYS, closed="both")
            & (pl.col("total_assets") > 0)
            & (pl.col("total_equity") > 0)
            & pl.col("momentum_60d").is_not_null()
            & pl.col("net_income_attributable").is_not_null()
            & pl.col("net_operating_cash_flow").is_not_null()
            & pl.col("debt_ratio").is_not_null()
        )
        .with_columns(
            (pl.col("net_income_attributable") / pl.col("total_assets")).alias("return_on_assets"),
            (pl.col("net_operating_cash_flow") / pl.col("total_assets")).alias(
                "cash_flow_on_assets"
            ),
            (-pl.col("debt_ratio")).alias("low_debt_score"),
            pl.col("cap_rank").alias("original_cap_rank"),
        )
    )
    score_columns = (
        "return_on_assets",
        "cash_flow_on_assets",
        "low_debt_score",
        "momentum_60d",
    )
    work = work.with_columns(
        *[
            (pl.col(column).rank(method="average").over("date") / pl.len().over("date")).alias(
                f"{column}_rank"
            )
            for column in score_columns
        ]
    ).with_columns(
        pl.sum_horizontal(*(f"{column}_rank" for column in score_columns)).alias(
            "quality_momentum_score"
        )
    )
    return (
        work.sort(
            ["date", "quality_momentum_score", "original_cap_rank", "symbol"],
            descending=[False, True, False, False],
        )
        .with_columns(
            pl.col("quality_momentum_score")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("cap_rank")
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


def stage_trading_dates(all_dates: list[date], stage: str) -> list[date]:
    start, end = STAGES[stage]
    return [day for day in all_dates if start <= day <= end]


def _common_checks(candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "buy_execution_at_least_80pct": candidate["execution"]["buy"]["execution_rate"] >= 0.80,
        "sell_execution_at_least_80pct": candidate["execution"]["sell"]["execution_rate"] >= 0.80,
        "no_unresolved_positions": candidate["integrity"]["ending_unresolved_positions"] == 0,
        "cash_reconciled": candidate["integrity"]["max_cash_reconciliation_error"] <= 0.01,
    }


def evaluate(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    primary = result["accounts"][str(int(PRIMARY_CAPITAL))]["quality_momentum"]
    metrics = primary["metrics"]
    yearly = {row["year"]: row["account_return"] for row in metrics["yearly"]}
    if stage == "development":
        checks = {
            "annualized_at_least_45pct": metrics["account_annualized"] >= 0.45,
            "drawdown_within_35pct": metrics["account_max_drawdown"] >= -0.35,
            "all_7_years_positive": all(yearly.get(year, -1.0) > 0 for year in range(2014, 2021)),
            **_common_checks(primary),
        }
    elif stage == "validation":
        checks = {
            "annualized_at_least_35pct": metrics["account_annualized"] >= 0.35,
            "drawdown_within_25pct": metrics["account_max_drawdown"] >= -0.25,
            "all_3_years_positive": all(yearly.get(year, -1.0) > 0 for year in range(2021, 2024)),
            **_common_checks(primary),
        }
    else:
        checks = {
            "2024_2026_all_positive": all(yearly.get(year, -1.0) > 0 for year in range(2024, 2027)),
            "2026_return_above_30pct": yearly.get(2026, -1.0) > 0.30,
            **_common_checks(primary),
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def run_stage(
    data_dir: Path,
    snapshots: pl.DataFrame,
    stage: str,
) -> dict[str, Any]:
    start, end = STAGES[stage]
    source = main_board.filter_main_board(baseline.load_daily(data_dir, end=end))
    all_dates = source["date"].unique().sort().to_list()
    scoped_dates = stage_trading_dates(all_dates, stage)
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    control_all = account.build_signal_candidates(panel)
    candidate_all = attach_quality_momentum(control_all, panel, snapshots)
    control = control_all.filter(pl.col("entry_date").is_between(start, end, closed="both"))
    candidate = candidate_all.filter(pl.col("entry_date").is_between(start, end, closed="both"))
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select("date", "period", "market_net")
    symbols = control_all["symbol"].unique().to_list()
    del panel, observations
    gc.collect()

    source_quotes = main_board.filter_main_board(baseline.load_daily(data_dir, end=end)).filter(
        pl.col("symbol").is_in(symbols)
    )
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(control_all, quotes)

    accounts: dict[str, Any] = {}
    for capital in CAPITALS:
        control_result = account.run_independent_account(
            stage,
            control,
            execution_grid,
            quotes,
            scoped_dates,
            weekly_market,
            initial_cash=capital,
        )
        candidate_result = account.run_independent_account(
            stage,
            candidate,
            execution_grid,
            quotes,
            scoped_dates,
            weekly_market,
            initial_cash=capital,
        )
        accounts[str(int(capital))] = {
            "initial_cash": capital,
            "control": _summary(control_result),
            "quality_momentum": _summary(candidate_result),
        }
    counts = candidate.group_by("entry_date").agg(pl.len().alias("eligible")).sort("entry_date")
    return {
        "stage": stage,
        "period": {"start": start, "end": end},
        "trading_days": len(scoped_dates),
        "funnel": {
            "control_candidate_rows": control.height,
            "quality_momentum_candidate_rows": candidate.height,
            "rebalance_days": control["entry_date"].n_unique(),
            "days_with_candidates": counts.height,
            "mean_candidates": counts["eligible"].mean() if counts.height else 0.0,
            "minimum_candidates": counts["eligible"].min() if counts.height else 0,
            "days_with_at_least_20": counts.filter(pl.col("eligible") >= 20).height,
        },
        "accounts": accounts,
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit_path = data_dir / "research" / "p0_main_board_microcap_financial_survival_data.json"
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if audit_sha != EXPECTED_AUDIT_SHA256:
        raise ValueError(f"financial audit hash mismatch: {audit_sha}")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "DATA_QUALIFIED":
        raise ValueError("financial data audit did not qualify")
    snapshots = pl.read_parquet(
        data_dir
        / "research"
        / "main_board_microcap_financial_survival"
        / "annual_snapshots.parquet"
    )
    stages: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for stage in STAGES:
        if stage == "validation" and not decisions["development"]["passed"]:
            stages[stage] = {"status": "NOT_READ_AFTER_DEVELOPMENT_FAILURE"}
            continue
        if stage == "known_stress" and not decisions.get("validation", {}).get("passed", False):
            stages[stage] = {"status": "NOT_READ_AFTER_VALIDATION_FAILURE"}
            continue
        stage_result = run_stage(data_dir, snapshots, stage)
        decision = evaluate(stage, stage_result)
        stage_result["decision"] = decision
        stages[stage] = stage_result
        decisions[stage] = decision
    passed = all(decisions.get(stage, {}).get("passed", False) for stage in STAGES)
    payload = {
        "schema_version": "p0-main-board-microcap-quality-momentum-v1",
        "contract_frozen": "2026-09-03",
        "data_audit_sha256": audit_sha,
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "max_financial_age_days": MAX_FINANCIAL_AGE_DAYS,
            "momentum_observations": MOMENTUM_OBSERVATIONS,
            "score": "equal_rank_roa_cfoa_low_debt_momentum_60d",
            "capital_ladder": list(CAPITALS),
        },
        "stages": stages,
        "decision": ("FORWARD_ELIGIBLE" if passed else "TERMINATE_QUALITY_MOMENTUM"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    development = stages["development"]
    primary = development["accounts"][str(int(PRIMARY_CAPITAL))]
    primary_summary = {
        name: {
            "metrics": result["metrics"],
            "execution": result["execution"],
            "integrity": result["integrity"],
            "account": result["account"],
        }
        for name, result in primary.items()
        if name != "initial_cash"
    }
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "data_audit_sha256": audit_sha,
                "development_funnel": development["funnel"],
                "development_gate": development["decision"],
                "primary_account": primary_summary,
                "stage_status": {
                    stage: result.get("status", "READ") for stage, result in stages.items()
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
        default=Path("/app/data/research/p0_main_board_microcap_quality_momentum_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
