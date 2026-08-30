"""Calibrate and validate the preregistered P0-B3 micro-cap risk switch."""
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
sys.path.insert(0, str(RESEARCH))

import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402

DEVELOPMENT_START = baseline.START
DEVELOPMENT_END = baseline.DEVELOPMENT_END
VALIDATION_START = date(2021, 1, 1)
VALIDATION_END = baseline.VALIDATION_END
STRESS_START = date(2024, 1, 1)

CAPITAL_TIERS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
MIN_RISK_OFF_DAYS = 5
RESUME_CLEAN_DAYS = 3

FEATURE_COLUMNS = (
    "microcap_excess_5d",
    "microcap_breadth_3d",
    "microcap_limit_down_3d",
    "microcap_liquidity_5d_60d",
)


def build_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Build close-known daily micro-cap state features without future data."""
    cross_section = (
        panel.filter(
            (pl.col("market_cap") > 0)
            & (pl.col("amount") > 0)
            & pl.col("daily_return").is_not_null()
        )
        .with_columns(
            pl.len().over("date").alias("universe_count"),
            pl.col("market_cap").rank(method="ordinal").over("date").alias("cap_rank"),
        )
        .with_columns(
            (
                ((pl.col("cap_rank") - 1) * 10 / pl.col("universe_count"))
                .floor()
                .clip(0, 9)
                .cast(pl.UInt8)
            ).alias("cap_decile")
        )
        .group_by("date")
        .agg(
            pl.col("daily_return")
            .filter(pl.col("cap_decile") == 0)
            .mean()
            .alias("microcap_daily_return"),
            pl.col("daily_return").mean().alias("market_daily_return"),
            (pl.col("daily_return") > 0)
            .filter(pl.col("cap_decile") == 0)
            .mean()
            .alias("microcap_breadth"),
            pl.col("is_limit_down")
            .filter(pl.col("cap_decile") == 0)
            .mean()
            .alias("microcap_limit_down"),
            pl.col("amount")
            .filter(pl.col("cap_decile") == 0)
            .median()
            .alias("microcap_median_amount"),
            pl.len().filter(pl.col("cap_decile") == 0).alias("microcap_count"),
        )
        .sort("date")
    )
    return cross_section.with_columns(
        (
            (pl.col("microcap_daily_return") + 1.0).rolling_map(
                lambda values: values.product(), window_size=5, min_samples=5
            )
            / (pl.col("market_daily_return") + 1.0).rolling_map(
                lambda values: values.product(), window_size=5, min_samples=5
            )
            - 1.0
        ).alias("microcap_excess_5d"),
        pl.col("microcap_breadth")
        .rolling_mean(window_size=3, min_samples=3)
        .alias("microcap_breadth_3d"),
        pl.col("microcap_limit_down")
        .rolling_mean(window_size=3, min_samples=3)
        .alias("microcap_limit_down_3d"),
        (
            pl.col("microcap_median_amount").rolling_mean(
                window_size=5, min_samples=5
            )
            / pl.col("microcap_median_amount").rolling_mean(
                window_size=60, min_samples=60
            )
        ).alias("microcap_liquidity_5d_60d"),
    )


def calibrate_thresholds(features: pl.DataFrame) -> dict[str, float]:
    scoped = features.filter(
        (pl.col("date") >= pl.lit(DEVELOPMENT_START))
        & (pl.col("date") <= pl.lit(DEVELOPMENT_END))
    ).drop_nulls(FEATURE_COLUMNS)
    if scoped.is_empty():
        raise ValueError("development features are empty")

    def quantile(column: str, probability: float) -> float:
        value = scoped.get_column(column).quantile(
            probability, interpolation="nearest"
        )
        if value is None or not math.isfinite(value):
            raise ValueError(f"invalid threshold for {column}")
        return float(value)

    return {
        "microcap_excess_5d_p10": quantile("microcap_excess_5d", 0.10),
        "microcap_breadth_3d_p10": quantile("microcap_breadth_3d", 0.10),
        "microcap_limit_down_3d_p90": quantile(
            "microcap_limit_down_3d", 0.90
        ),
        "microcap_liquidity_5d_60d_p10": quantile(
            "microcap_liquidity_5d_60d", 0.10
        ),
        "microcap_limit_down_3d_p95": quantile(
            "microcap_limit_down_3d", 0.95
        ),
    }


def apply_alarms(
    features: pl.DataFrame, thresholds: dict[str, float]
) -> pl.DataFrame:
    return features.with_columns(
        (
            pl.col("microcap_excess_5d")
            <= thresholds["microcap_excess_5d_p10"]
        ).fill_null(False).alias("excess_alarm"),
        (
            pl.col("microcap_breadth_3d")
            <= thresholds["microcap_breadth_3d_p10"]
        ).fill_null(False).alias("breadth_alarm"),
        (
            pl.col("microcap_limit_down_3d")
            >= thresholds["microcap_limit_down_3d_p90"]
        ).fill_null(False).alias("limit_down_alarm"),
        (
            pl.col("microcap_liquidity_5d_60d")
            <= thresholds["microcap_liquidity_5d_60d_p10"]
        ).fill_null(False).alias("liquidity_alarm"),
        (
            pl.col("microcap_limit_down_3d")
            >= thresholds["microcap_limit_down_3d_p95"]
        ).fill_null(False).alias("severe_limit_down"),
    ).with_columns(
        pl.sum_horizontal(
            "excess_alarm",
            "breadth_alarm",
            "limit_down_alarm",
            "liquidity_alarm",
        ).alias("ordinary_alarm_count")
    )


def build_risk_clock(
    alarm_features: pl.DataFrame,
) -> tuple[dict[date, bool], list[dict[str, Any]], list[dict[str, Any]]]:
    """Map each next trading open to the state known at the prior close."""
    rows = alarm_features.sort("date").to_dicts()
    next_open_state: dict[date, bool] = {}
    decisions: list[dict[str, Any]] = []
    switches: list[dict[str, Any]] = []
    risk_on = True
    off_days = 0
    clean_days = 0

    for index, row in enumerate(rows[:-1]):
        decision_date = row["date"]
        action_date = rows[index + 1]["date"]
        count = int(row["ordinary_alarm_count"] or 0)
        severe = bool(row["severe_limit_down"])
        next_risk_on = risk_on
        switch = None

        if risk_on:
            if severe or count >= 2:
                next_risk_on = False
                off_days = 0
                clean_days = 0
                switch = "RISK_OFF"
        else:
            off_days += 1
            clean_days = clean_days + 1 if count == 0 else 0
            if off_days >= MIN_RISK_OFF_DAYS and clean_days >= RESUME_CLEAN_DAYS:
                next_risk_on = True
                switch = "RISK_ON"
                off_days = 0
                clean_days = 0

        decision = {
            "decision_date": decision_date,
            "action_date": action_date,
            "risk_on": next_risk_on,
            "switch": switch,
            "ordinary_alarm_count": count,
            "severe_limit_down": severe,
            **{column: row.get(column) for column in FEATURE_COLUMNS},
        }
        decisions.append(decision)
        next_open_state[action_date] = next_risk_on
        if switch:
            switches.append(decision.copy())
        risk_on = next_risk_on

    return next_open_state, decisions, switches


def _candidate_groups(candidates: pl.DataFrame) -> dict[date, list[dict[str, Any]]]:
    output: dict[date, list[dict[str, Any]]] = {}
    for key, group in candidates.partition_by("entry_date", as_dict=True).items():
        day = key[0] if isinstance(key, tuple) else key
        output[day] = group.sort(["cap_rank", "symbol"]).to_dicts()
    return output


def build_action_candidates(
    weekly_candidates: pl.DataFrame,
    all_dates: list[date],
    risk_by_open: dict[date, bool],
    *,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, list[date], list[date]]:
    """Schedule weekly rebalances plus daily exit retries and resume entries."""
    weekly = _candidate_groups(weekly_candidates)
    latest: list[dict[str, Any]] = []
    previous_risk_on = True
    action_rows: list[dict[str, Any]] = []
    action_dates: list[date] = []
    naked_dates: list[date] = []
    for day in all_dates:
        if day in weekly:
            latest = weekly[day]
        risk_on = risk_by_open.get(day, previous_risk_on)
        weekly_action = day in weekly
        resumed = risk_on and not previous_risk_on
        scheduled = weekly_action or not risk_on or resumed
        if start <= day <= end:
            if weekly_action:
                naked_dates.append(day)
            if scheduled:
                action_dates.append(day)
                if risk_on:
                    for row in latest:
                        action_rows.append({**row, "entry_date": day})
        previous_risk_on = risk_on
    if action_rows:
        frame = pl.DataFrame(action_rows, infer_schema_length=None)
    else:
        frame = weekly_candidates.head(0)
    return frame, action_dates, naked_dates


def build_execution_grid_for_dates(
    weekly_candidates: pl.DataFrame,
    action_dates: list[date],
    quotes: pl.DataFrame,
) -> pl.DataFrame:
    symbols = weekly_candidates.select("symbol").unique().sort("symbol")
    dates = pl.DataFrame({"entry_date": action_dates})
    seed = symbols.join(dates, how="cross")
    return account.build_execution_grid(seed, quotes)


def _period_bounds(mode: str, last_date: date) -> tuple[date, date]:
    if mode == "calibrate":
        return DEVELOPMENT_START, DEVELOPMENT_END
    if mode == "validate":
        return VALIDATION_START, VALIDATION_END
    if mode == "stress":
        return STRESS_START, last_date
    raise ValueError(f"unsupported mode: {mode}")


def _single_period_metric(
    daily: pl.DataFrame,
    weekly_market: pl.DataFrame,
    period: str,
) -> dict[str, Any]:
    return next(
        row
        for row in account.account_period_metrics(daily, weekly_market)
        if row["period"] == period
    )


def simulate_tier(
    *,
    capital: float,
    period: str,
    action_candidates: pl.DataFrame,
    action_dates: list[date],
    naked_candidates: pl.DataFrame,
    naked_dates: list[date],
    execution_grid: pl.DataFrame,
    quotes: pl.DataFrame,
    scoped_dates: list[date],
    weekly_market: pl.DataFrame,
) -> dict[str, Any]:
    overlay = account.simulate_account(
        action_candidates,
        execution_grid,
        initial_cash=capital,
        action_dates=action_dates,
    )
    overlay_daily, stale = account.build_daily_equity(
        overlay, quotes, scoped_dates, initial_cash=capital
    )
    naked = account.simulate_account(
        naked_candidates,
        execution_grid,
        initial_cash=capital,
        action_dates=naked_dates,
    )
    naked_daily, naked_stale = account.build_daily_equity(
        naked, quotes, scoped_dates, initial_cash=capital
    )
    metrics = _single_period_metric(overlay_daily, weekly_market, period)
    naked_metrics = _single_period_metric(naked_daily, weekly_market, period)
    execution = account.execution_summary(overlay["orders"])
    integrity = {
        **stale,
        "max_cash_reconciliation_error": overlay[
            "max_cash_reconciliation_error"
        ],
    }
    return {
        "capital": capital,
        "metrics": metrics,
        "naked_metrics": naked_metrics,
        "annualized_delta_vs_naked": (
            metrics["account_annualized"] - naked_metrics["account_annualized"]
        ),
        "execution": execution,
        "integrity": integrity,
        "naked_integrity": {
            **naked_stale,
            "max_cash_reconciliation_error": naked[
                "max_cash_reconciliation_error"
            ],
        },
        "account": account.account_summary(overlay, overlay_daily),
        "naked_account": account.account_summary(naked, naked_daily),
        "daily_equity": overlay_daily.select(
            "date",
            "equity",
            "cash",
            "position_value",
            "position_count",
            "stale_positions",
            "cash_ratio",
        ).to_dicts(),
        "orders": overlay["orders"],
        "worst_weeks": account.worst_weeks(overlay_daily),
    }


def evaluate_period(tiers: list[dict[str, Any]]) -> dict[str, Any]:
    primary = next(row for row in tiers if row["capital"] == CAPITAL_TIERS[0])
    metric = primary["metrics"]
    execution = primary["execution"]
    integrity = primary["integrity"]
    checks = {
        "annualized_at_least_15pct": metric["account_annualized"] >= 0.15,
        "annualized_excess_at_least_10pp": metric["annualized_excess"] >= 0.10,
        "max_drawdown_at_most_25pct": metric["account_max_drawdown"] >= -0.25,
        "annualized_loss_vs_naked_at_most_5pp": (
            primary["annualized_delta_vs_naked"] >= -0.05
        ),
        "buy_execution_at_least_80pct": (
            execution["buy"]["execution_rate"] >= 0.80
        ),
        "sell_execution_at_least_80pct": (
            execution["sell"]["execution_rate"] >= 0.80
        ),
        "no_unresolved_positions": integrity["ending_unresolved_positions"] == 0,
        "cash_error_at_most_one_cent": (
            integrity["max_cash_reconciliation_error"] <= 0.01
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def load_frozen_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "p0-microcap-escape-thresholds-v1":
        raise ValueError("unexpected frozen threshold schema")
    return {key: float(value) for key, value in payload["thresholds"].items()}


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
    thresholds_path: Path | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    load_end = (
        DEVELOPMENT_END
        if mode == "calibrate"
        else VALIDATION_END
        if mode == "validate"
        else end
    )
    source = baseline.load_daily(data_dir, end=load_end)
    if source.is_empty():
        raise ValueError("no daily data")
    all_dates = source.get_column("date").unique().sort().to_list()
    pit = baseline.attach_point_in_time_data(source, data_dir)
    del source
    gc.collect()
    panel = baseline.prepare_panel(pit)
    del pit
    gc.collect()
    features = build_daily_features(panel)
    thresholds = (
        calibrate_thresholds(features)
        if mode == "calibrate"
        else load_frozen_thresholds(
            thresholds_path
            if thresholds_path is not None
            else ROOT / "research" / "p0_microcap_escape_thresholds.json"
        )
    )
    alarms = apply_alarms(features, thresholds)
    risk_by_open, decisions, switches = build_risk_clock(alarms)
    weekly_candidates = account.build_signal_candidates(panel)
    observations = baseline.build_weekly_observations(panel)
    weekly_market = baseline.weekly_portfolios(observations).select(
        "date", "period", "market_net"
    )
    candidate_symbols = weekly_candidates.get_column("symbol").unique().to_list()
    del panel, observations
    gc.collect()

    last_date = all_dates[-1]
    start, finish = _period_bounds(mode, last_date)
    scoped_dates = [day for day in all_dates if start <= day <= finish]
    action_candidates, action_dates, naked_dates = build_action_candidates(
        weekly_candidates,
        all_dates,
        risk_by_open,
        start=start,
        end=finish,
    )
    naked_candidates = weekly_candidates.filter(
        (pl.col("entry_date") >= pl.lit(start))
        & (pl.col("entry_date") <= pl.lit(finish))
    )
    source_quotes = baseline.load_daily(data_dir, end=load_end).filter(
        pl.col("symbol").is_in(candidate_symbols)
    )
    source_quotes = account.attach_quote_names(source_quotes, data_dir)
    quotes = account.prepare_quote_panel(source_quotes)
    del source_quotes
    gc.collect()
    execution_dates = sorted(set(action_dates) | set(naked_dates))
    execution_grid = build_execution_grid_for_dates(
        weekly_candidates, execution_dates, quotes
    )

    period = {
        "calibrate": "development",
        "validate": "validation",
        "stress": "known_stress",
    }[mode]
    tiers = [
        simulate_tier(
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
        )
        for capital in CAPITAL_TIERS
    ]
    decision = evaluate_period(tiers)
    scoped_decisions = [
        row for row in decisions if start <= row["action_date"] <= finish
    ]
    scoped_switches = [
        row for row in switches if start <= row["action_date"] <= finish
    ]
    payload = {
        "schema_version": "p0-microcap-escape-v1",
        "mode": mode,
        "contract": {
            "primary_capital": CAPITAL_TIERS[0],
            "capital_tiers": CAPITAL_TIERS,
            "target_positions": account.TARGET_POSITIONS,
            "threshold_interpolation": "nearest",
            "risk_off": "severe_limit_down_or_at_least_two_ordinary_alarms",
            "minimum_risk_off_trading_days": MIN_RISK_OFF_DAYS,
            "resume": "three_consecutive_zero_alarm_closes_after_minimum_hold",
            "execution": "close_decision_next_trading_open",
            "feature_excess": "five_day_microcap_compound_divided_by_market_compound_minus_one",
        },
        "data": {
            "first_loaded_date": all_dates[0],
            "last_loaded_date": all_dates[-1],
            "period_start": start,
            "period_end": finish,
            "period_trading_days": len(scoped_dates),
            "candidate_symbols": len(candidate_symbols),
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
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, default=_json_default
    )
    output.write_text(serialized, encoding="utf-8")
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "mode": mode,
                "data": payload["data"],
                "thresholds": thresholds,
                "risk": {
                    key: value
                    for key, value in payload["risk"].items()
                    if key not in {"switches", "decisions"}
                },
                "capital_tiers": [
                    {
                        "capital": row["capital"],
                        "metrics": row["metrics"],
                        "naked_metrics": row["naked_metrics"],
                        "annualized_delta_vs_naked": row[
                            "annualized_delta_vs_naked"
                        ],
                        "execution": row["execution"],
                        "integrity": row["integrity"],
                        "account": row["account"],
                    }
                    for row in tiers
                ],
                "decision": decision,
                "output": str(output),
                "sha256": sha256,
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
    parser.add_argument(
        "--mode", choices=("calibrate", "validate", "stress"), required=True
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--end", type=date.fromisoformat)
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
