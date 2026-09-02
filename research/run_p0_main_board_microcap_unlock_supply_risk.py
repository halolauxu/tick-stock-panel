"""Run the frozen main-board micro-cap scheduled-unlock supply-risk study."""

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

import audit_p0_unlock_supply_risk_data as unlock_audit  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_main_board_microcap_pledge_risk_overlay as pledge  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
PRIMARY_CAPITAL = 200_000.0
MIN_FLOAT_RATIO_PCT = 5.0
UPCOMING_TRADING_DAYS = 20
SYSTEM_HISTORY_WEEKS = 52
SYSTEM_PERCENTILE = 0.90
SYSTEM_MINIMUM_EVENTS = 10
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = baseline.DEVELOPMENT_END
EXPECTED_AUDIT_SHA256 = "6d09fd88acca2ca28ab7e71c1e3efbdcd17fa879c49d2dd72573f4d1789b837d"


def load_eligible_details(data_dir: Path) -> pl.DataFrame:
    details = unlock_audit.load_details(data_dir)
    universe = pl.read_parquet(data_dir / "research" / "historical_stock_universe.parquet").filter(
        pl.col("market") == "主板"
    )
    return (
        details.join(
            universe.select("symbol", "list_date", "delist_date"),
            on="symbol",
            how="inner",
        )
        .filter(
            (pl.col("ann_date") <= pl.col("float_date"))
            & (pl.col("float_date") >= pl.col("list_date"))
            & (pl.col("delist_date").is_null() | (pl.col("float_date") <= pl.col("delist_date")))
            & (pl.col("float_shares") > 0)
            & (pl.col("float_ratio") > 0)
        )
        .select("symbol", "ann_date", "float_date", "float_shares", "float_ratio")
        .sort(["float_date", "symbol", "ann_date"])
    )


def build_weekly_unlock_risk(
    weekly_dates: pl.DataFrame,
    details: pl.DataFrame,
    trading_dates: list[date],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    date_index = {day: index for index, day in enumerate(trading_dates)}
    history: list[int] = []
    exclusion_rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []
    for weekly in weekly_dates.sort("date").iter_rows(named=True):
        signal_date = weekly["date"]
        index = date_index[signal_date]
        future_dates = trading_dates[index + 1 : index + 1 + UPCOMING_TRADING_DAYS]
        if future_dates:
            upcoming = (
                details.filter(
                    (pl.col("ann_date") <= signal_date) & pl.col("float_date").is_in(future_dates)
                )
                .group_by("symbol", "float_date")
                .agg(
                    pl.col("float_shares").sum().alias("float_shares"),
                    pl.col("float_ratio").sum().alias("float_ratio_pct"),
                )
                .filter(
                    (pl.col("float_shares") > 0)
                    & (pl.col("float_ratio_pct") >= MIN_FLOAT_RATIO_PCT)
                    & (pl.col("float_ratio_pct") <= 100)
                )
            )
            symbols = upcoming["symbol"].unique().sort().to_list()
        else:
            symbols = []
        for symbol in symbols:
            exclusion_rows.append({"symbol": symbol, "date": signal_date})
        count = len(symbols)
        threshold = (
            pledge._nearest_rank_percentile(history[-SYSTEM_HISTORY_WEEKS:], SYSTEM_PERCENTILE)
            if len(history) >= SYSTEM_HISTORY_WEEKS
            else None
        )
        risk_off = threshold is not None and count >= max(SYSTEM_MINIMUM_EVENTS, threshold)
        clock_rows.append(
            {
                **weekly,
                "upcoming_material_unlock_symbols_20d": count,
                "historical_threshold": threshold,
                "risk_off": risk_off,
            }
        )
        history.append(count)
    exclusion_schema = {"symbol": pl.Utf8, "date": pl.Date}
    exclusions = (
        pl.DataFrame(exclusion_rows).unique().sort(["date", "symbol"])
        if exclusion_rows
        else pl.DataFrame(schema=exclusion_schema)
    )
    return exclusions, pl.DataFrame(clock_rows)


def build_arms(
    control: pl.DataFrame,
    details: pl.DataFrame,
    trading_dates: list[date],
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    weekly_dates = (
        control.filter(pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both"))
        .select("date", "entry_date")
        .unique()
        .sort("date")
    )
    exclusions, risk_clock = build_weekly_unlock_risk(weekly_dates, details, trading_dates)
    stock_only = control.join(exclusions, on=["symbol", "date"], how="anti")
    risk_off_entries = risk_clock.filter(pl.col("risk_off")).select("entry_date")
    systemic_only = control.join(risk_off_entries, on="entry_date", how="anti")
    combined = stock_only.join(risk_off_entries, on="entry_date", how="anti")
    return (
        {
            "control": control,
            "stock_exclusion": stock_only,
            "systemic_gate": systemic_only,
            "combined": combined,
        },
        exclusions,
        risk_clock,
    )


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"daily_equity", "rebalance_snapshots", "orders"}
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    audit_path = data_dir / "research" / "p0_main_board_microcap_unlock_supply_data.json"
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if audit_sha != EXPECTED_AUDIT_SHA256:
        raise ValueError(f"unlock audit hash mismatch: {audit_sha}")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "DATA_QUALIFIED":
        raise ValueError("unlock supply data audit did not qualify")

    details = load_eligible_details(data_dir)
    source = main_board.filter_main_board(baseline.load_daily(data_dir, end=DEVELOPMENT_END))
    all_dates = source["date"].unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    control_all = account.build_signal_candidates(panel)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select("date", "period", "market_net")
    arms, exclusions, risk_clock = build_arms(control_all, details, all_dates)
    control = arms["control"].filter(
        pl.col("entry_date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
    )
    action_dates = control["entry_date"].unique().sort().to_list()
    symbols = control_all["symbol"].unique().to_list()
    del panel, observations
    gc.collect()

    source_quotes = main_board.filter_main_board(
        baseline.load_daily(data_dir, end=DEVELOPMENT_END)
    ).filter(pl.col("symbol").is_in(symbols))
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_grid = account.build_execution_grid(control_all, quotes)

    accounts: dict[str, Any] = {}
    for capital in CAPITALS:
        accounts[str(int(capital))] = {"initial_cash": capital}
        for name, candidates in arms.items():
            result = pledge.run_arm_account(
                "development",
                candidates,
                action_dates,
                execution_grid,
                quotes,
                all_dates,
                weekly_market,
                initial_cash=capital,
            )
            accounts[str(int(capital))][name] = _summary(result)

    development = {
        "stage": "development",
        "period": {"start": DEVELOPMENT_START, "end": DEVELOPMENT_END},
        "trading_days": len(
            [day for day in all_dates if DEVELOPMENT_START <= day <= DEVELOPMENT_END]
        ),
        "funnel": {
            "eligible_detail_rows": details.height,
            "exclusion_symbol_weeks": exclusions.height,
            "control_candidate_rows": control.height,
            "stock_exclusion_candidate_rows": arms["stock_exclusion"]
            .filter(
                pl.col("entry_date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
            )
            .height,
            "systemic_risk_off_weeks": risk_clock.filter(pl.col("risk_off")).height,
            "rebalance_weeks": risk_clock.height,
        },
        "risk_clock": risk_clock.to_dicts(),
        "accounts": accounts,
    }
    decision = pledge.evaluate("development", development)
    development["decision"] = decision
    payload = {
        "schema_version": "p0-main-board-microcap-unlock-supply-risk-v1",
        "contract_frozen": "2026-09-03",
        "data_audit_sha256": audit_sha,
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "minimum_float_ratio_pct": MIN_FLOAT_RATIO_PCT,
            "upcoming_trading_days": UPCOMING_TRADING_DAYS,
            "system_history_weeks": SYSTEM_HISTORY_WEEKS,
            "system_percentile": SYSTEM_PERCENTILE,
            "system_minimum_events": SYSTEM_MINIMUM_EVENTS,
            "primary_arm": "combined",
            "capital_ladder": list(CAPITALS),
        },
        "stages": {
            "development": development,
            "validation": {
                "status": (
                    "NEEDS_2021_2023_DATA"
                    if decision["passed"]
                    else "NOT_READ_AFTER_DEVELOPMENT_FAILURE"
                )
            },
            "known_stress": {"status": "NOT_READ_BEFORE_VALIDATION"},
        },
        "decision": (
            "DEVELOPMENT_PASSED_NEEDS_VALIDATION_DATA"
            if decision["passed"]
            else "TERMINATE_UNLOCK_SUPPLY_RISK"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    primary = accounts[str(int(PRIMARY_CAPITAL))]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "data_audit_sha256": audit_sha,
                "development_funnel": development["funnel"],
                "development_gate": decision,
                "primary_account": {
                    name: {
                        key: result[key] for key in ("metrics", "execution", "integrity", "account")
                    }
                    for name, result in primary.items()
                    if name != "initial_cash"
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
        default=Path("/app/data/research/p0_main_board_microcap_unlock_supply_risk_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
