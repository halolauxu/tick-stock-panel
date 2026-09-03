"""Run the frozen R2-02 development account for operating forecast events."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import run_p0_daily_momentum_development as daily  # noqa: E402
import run_p0_forecast_drift_development as forecast  # noqa: E402
import run_p0_industry_confirmed_forecast_drift_discovery as industry  # noqa: E402
import run_p0_industry_momentum_development as shared  # noqa: E402
import run_p0_main_board_forecast_drift_regime_discovery as regime  # noqa: E402
import run_p0_main_board_microcap_account as main_board  # noqa: E402
import run_p0_microcap_account as account  # noqa: E402
import run_p0_microcap_baseline as baseline  # noqa: E402
import run_p0_short_horizon_baseline_audit as lifecycle  # noqa: E402

SCHEMA_VERSION = "p0-short-horizon-event-account-development-v1"
DEVELOPMENT_START = date(2014, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
HORIZONS = (2, 5, 10)
INITIAL_CASH = 200_000.0
TARGET_POSITIONS = 5
COOLDOWN_SESSIONS = 10
MAX_EXIT_DELAY = 20
MIN_SIGNAL_AMOUNT = 50_000_000.0
EVENT_FAMILY = "company_specific_operating_forecast"
MICROCAP_BASELINE = "p0_main_board_microcap_account_v1.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _cluster_t(frame: pl.DataFrame, column: str) -> float | None:
    daily_frame = (
        frame.drop_nulls(column).group_by("ann_date").agg(pl.col(column).mean().alias("value"))
    )
    if daily_frame.height <= 1:
        return None
    mean = daily_frame.get_column("value").mean()
    standard_deviation = daily_frame.get_column("value").std(ddof=1)
    if mean is None or standard_deviation in (None, 0.0):
        return None
    return mean / (standard_deviation / math.sqrt(daily_frame.height))


def load_qualified_events(
    data_dir: Path, start: date, end: date
) -> tuple[pl.DataFrame, dict[str, Any]]:
    result_path = data_dir / "research" / "p0_short_horizon_event_facts_audit_v1.json"
    event_path = result_path.with_suffix(".events.parquet")
    if not result_path.is_file() or not event_path.is_file():
        raise ValueError("passing R2-01 fact audit and event table are required")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("decision", {}).get("verdict") != "PASS_TO_EVENT_ACCOUNT":
        raise ValueError("R2-01 fact audit has not passed")
    events = pl.read_parquet(event_path)
    scoped = events.filter(pl.col("ann_date").is_between(start, end, closed="both"))
    qualified = (
        scoped.filter(
            (pl.col("effective_reason_class") == "OPERATING")
            & (pl.col("prior_roe") > 0)
            & (pl.col("prior_operating_cash_to_revenue") > 0)
            & (pl.col("prior_net_operating_cash_flow") > 0)
        )
        .with_columns(pl.lit(EVENT_FAMILY).alias("category"))
        .sort(["ann_date", "symbol"])
    )
    audit = {
        "scoped_events": scoped.height,
        "qualified_events": qualified.height,
        "qualified_symbols": qualified.get_column("symbol").n_unique(),
        "qualified_announcement_days": qualified.get_column("ann_date").n_unique(),
        "fact_audit_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "event_table_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }
    return qualified, audit


def build_candidates(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    all_dates: list[date],
    horizon: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    calendar = pl.DataFrame({"entry_date": all_dates}).with_row_index("action_index")
    last_entry_index = len(all_dates) - horizon - MAX_EXIT_DELAY - 1
    signal_quotes = (
        events.sort(["symbol", "ann_date"])
        .join_asof(
            panel.select("symbol", "date", "raw_close", "amount").sort(["symbol", "date"]),
            left_on="ann_date",
            right_on="date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .rename({"date": "signal_quote_date"})
        .with_columns((pl.col("ann_date") + pl.duration(days=1)).alias("available_after"))
        .sort("available_after")
        .join_asof(
            calendar.sort("entry_date"),
            left_on="available_after",
            right_on="entry_date",
            strategy="forward",
        )
        .drop_nulls("entry_date")
        .filter(
            (pl.col("action_index") <= last_entry_index)
            & (pl.col("amount") >= MIN_SIGNAL_AMOUNT)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
        )
    )
    expanded = (
        signal_quotes.with_columns(
            pl.int_ranges(pl.col("action_index"), pl.col("action_index") + horizon).alias(
                "_active_indices"
            )
        )
        .explode("_active_indices")
        .drop("entry_date", "action_index")
        .join(
            calendar.rename({"action_index": "_active_indices"}),
            on="_active_indices",
            how="inner",
        )
        .sort(
            ["entry_date", "symbol", "ann_date", "p_change_min", "p_change_max"],
            descending=[False, False, True, True, True],
            nulls_last=True,
        )
        .unique(subset=["entry_date", "symbol"], keep="first")
        .sort(
            ["entry_date", "l1_code", "p_change_min", "p_change_max", "symbol"],
            descending=[False, False, True, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["entry_date", "l1_code"]).alias("industry_rank")
        )
        .filter(pl.col("industry_rank") == 1)
        .sort(
            ["entry_date", "p_change_min", "p_change_max", "symbol"],
            descending=[False, True, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("entry_date").alias("cap_rank"),
            pl.lit(EVENT_FAMILY).alias("family"),
        )
        .filter(pl.col("cap_rank") <= TARGET_POSITIONS)
        .select(
            pl.col("ann_date").alias("date"),
            "entry_date",
            "symbol",
            "l1_code",
            "l1_name",
            "p_change_min",
            "p_change_max",
            "net_profit_min",
            "net_profit_max",
            pl.col("amount").alias("signal_amount"),
            "cap_rank",
            "family",
        )
        .sort(["entry_date", "cap_rank", "symbol"])
    )
    return expanded, {
        "eligible_unique_events": signal_quotes.height,
        "eligible_event_symbols": signal_quotes.get_column("symbol").n_unique(),
        "daily_candidate_rows": expanded.height,
        "active_account_days": expanded.get_column("entry_date").n_unique(),
        "candidate_symbols": expanded.get_column("symbol").n_unique(),
        "last_accepted_entry_index": last_entry_index,
    }


def build_benchmarks(panel: pl.DataFrame, membership: pl.DataFrame, horizon: int) -> pl.DataFrame:
    entry = panel.select(
        "symbol",
        "trade_index",
        pl.col("date").alias("entry_date"),
        pl.col("open").alias("benchmark_entry_open"),
        pl.col("raw_open").alias("benchmark_entry_raw_open"),
        pl.col("amount").alias("benchmark_entry_amount"),
        pl.col("volume").alias("benchmark_entry_volume"),
        pl.col("excluded_name").alias("benchmark_entry_excluded"),
    )
    exit_prices = panel.select(
        "symbol",
        (pl.col("trade_index") - horizon).alias("trade_index"),
        pl.col("open").alias("benchmark_exit_open"),
    )
    returns = (
        entry.join(exit_prices, on=["symbol", "trade_index"], how="inner")
        .filter(
            ~pl.col("benchmark_entry_excluded").fill_null(True)
            & pl.col("benchmark_entry_raw_open").is_between(3.0, 300.0, closed="both")
            & (pl.col("benchmark_entry_amount").fill_null(0) >= MIN_SIGNAL_AMOUNT)
            & (pl.col("benchmark_entry_volume").fill_null(0) > 0)
            & (pl.col("benchmark_entry_open").fill_null(0) > 0)
            & (pl.col("benchmark_exit_open").fill_null(0) > 0)
        )
        .with_columns(
            (pl.col("benchmark_exit_open") / pl.col("benchmark_entry_open") - 1.0).alias(
                "benchmark_return"
            )
        )
    )
    mapped = (
        returns.sort(["symbol", "entry_date"])
        .join_asof(
            membership,
            left_on="entry_date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(
            pl.col("l1_code").is_not_null()
            & (pl.col("out_date").is_null() | (pl.col("entry_date") <= pl.col("out_date")))
        )
    )
    market = mapped.group_by("trade_index").agg(
        pl.col("benchmark_return").median().alias("market_median_return")
    )
    industry_benchmark = mapped.group_by("trade_index", "l1_code").agg(
        pl.col("benchmark_return").median().alias("industry_median_return")
    )
    return market.join(industry_benchmark, on="trade_index", how="inner")


def summarize_event_study(
    events: pl.DataFrame,
    panel: pl.DataFrame,
    membership: pl.DataFrame,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trades = forecast.build_trades(
        events, panel, holding_trading_days=horizon, max_exit_delay=MAX_EXIT_DELAY
    )
    eligible = pl.col("universe_eligible") & (pl.col("prior_amount") >= MIN_SIGNAL_AMOUNT)
    tradable = pl.col("tradable") & eligible
    trades = trades.with_columns(
        eligible.alias("universe_eligible"),
        tradable.alias("tradable"),
        pl.when(tradable).then(pl.col("net_return")).otherwise(None).alias("net_return"),
    ).join(
        build_benchmarks(panel, membership, horizon),
        on=["trade_index", "l1_code"],
        how="left",
    )
    trades = trades.with_columns(
        pl.when(pl.col("tradable") & pl.col("market_median_return").is_not_null())
        .then(pl.col("net_return") - pl.col("market_median_return"))
        .otherwise(None)
        .alias("market_excess"),
        pl.when(pl.col("tradable") & pl.col("industry_median_return").is_not_null())
        .then(pl.col("net_return") - pl.col("industry_median_return"))
        .otherwise(None)
        .alias("industry_excess"),
    )
    eligible_frame = trades.filter(pl.col("universe_eligible"))
    tradable_frame = trades.filter(pl.col("tradable"))
    benchmarked = tradable_frame.filter(
        pl.col("market_excess").is_not_null() & pl.col("industry_excess").is_not_null()
    )
    yearly = (
        benchmarked.with_columns(pl.col("ann_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("net_return").mean().alias("mean_net_return"),
            pl.col("market_excess").mean().alias("mean_market_excess"),
            pl.col("industry_excess").mean().alias("mean_industry_excess"),
            pl.col("market_excess").sum().alias("sum_market_excess"),
            pl.len().alias("events"),
        )
        .sort("year")
    )
    positive_sums = [
        float(value)
        for value in yearly.get_column("sum_market_excess").to_list()
        if value is not None and value > 0
    ]
    summary = {
        "events": trades.height,
        "universe_eligible_events": eligible_frame.height,
        "tradable_events": tradable_frame.height,
        "announcement_days": benchmarked.get_column("ann_date").n_unique(),
        "tradable_rate": (
            tradable_frame.height / eligible_frame.height if eligible_frame.height else 0.0
        ),
        "market_benchmark_coverage": (
            tradable_frame.get_column("market_excess").is_not_null().sum() / tradable_frame.height
            if tradable_frame.height
            else 0.0
        ),
        "industry_benchmark_coverage": (
            tradable_frame.get_column("industry_excess").is_not_null().sum() / tradable_frame.height
            if tradable_frame.height
            else 0.0
        ),
        "unresolved_exits": eligible_frame.filter(
            pl.col("entry_valid") & pl.col("exit_delay").is_null()
        ).height,
        "mean_net_return": benchmarked.get_column("net_return").mean(),
        "median_net_return": benchmarked.get_column("net_return").median(),
        "win_rate": (
            benchmarked.filter(pl.col("net_return") > 0).height / benchmarked.height
            if benchmarked.height
            else None
        ),
        "mean_market_excess": benchmarked.get_column("market_excess").mean(),
        "mean_industry_excess": benchmarked.get_column("industry_excess").mean(),
        "market_excess_cluster_t": _cluster_t(benchmarked, "market_excess"),
        "positive_market_excess_years": yearly.filter(pl.col("mean_market_excess") > 0).height,
        "max_year_positive_share": (
            max(positive_sums) / sum(positive_sums) if positive_sums else None
        ),
        "yearly": yearly.to_dicts(),
    }
    detail_columns = [
        "ann_date",
        "symbol",
        "l1_code",
        "l1_name",
        "mapped_entry_date",
        "actual_exit_date",
        "exit_delay",
        "universe_eligible",
        "tradable",
        "net_return",
        "market_median_return",
        "industry_median_return",
        "market_excess",
        "industry_excess",
    ]
    return summary, trades.select(detail_columns).sort(["ann_date", "symbol"]).to_dicts()


def _microcap_correlation(
    daily_equity: pl.DataFrame, baseline_path: Path, period_name: str
) -> float | None:
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = baseline_payload["accounts"][str(int(INITIAL_CASH))]["periods"][period_name][
        "daily_equity"
    ]
    microcap = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False),
        pl.col("equity").cast(pl.Float64),
    )
    microcap = microcap.with_columns(pl.col("equity").pct_change().alias("microcap_return"))
    joined = daily_equity.select("date", "daily_return").join(
        microcap.select("date", "microcap_return"), on="date", how="inner"
    )
    if joined.height < 3:
        return None
    return joined.select(pl.corr("daily_return", "microcap_return")).item()


def _max_industry_share(
    simulation: dict[str, Any],
    candidates: pl.DataFrame,
    quotes: pl.DataFrame,
    daily_equity: pl.DataFrame,
    all_dates: list[date],
) -> float:
    industry_lookup = {
        (row["entry_date"], row["symbol"]): row["l1_code"]
        for row in candidates.select("entry_date", "symbol", "l1_code").to_dicts()
    }
    date_index = {day: index for index, day in enumerate(all_dates)}
    wanted: list[dict[str, Any]] = []
    for interval in simulation["intervals"]:
        start_date = interval["start_date"]
        l1_code = industry_lookup.get((start_date, interval["symbol"]))
        if l1_code is None:
            continue
        start_index = date_index[start_date]
        end_index = date_index.get(interval.get("end_date"), len(all_dates))
        for index in range(start_index, end_index):
            wanted.append(
                {
                    "position_id": interval["position_id"],
                    "symbol": interval["symbol"],
                    "date": all_dates[index],
                    "units": interval["units"],
                    "l1_code": l1_code,
                }
            )
    if not wanted:
        return 0.0
    marked = (
        pl.DataFrame(wanted, infer_schema_length=None)
        .join(quotes.select("symbol", "date", "close"), on=["symbol", "date"], how="left")
        .sort(["position_id", "date"])
        .with_columns(pl.col("close").forward_fill().over("position_id").alias("mark"))
        .with_columns((pl.col("units") * pl.col("mark")).alias("market_value"))
        .group_by("date", "l1_code")
        .agg(pl.col("market_value").sum())
        .join(daily_equity.select("date", "equity"), on="date", how="left")
        .with_columns((pl.col("market_value") / pl.col("equity")).alias("share"))
    )
    return float(marked.get_column("share").max() or 0.0)


def _unexpected_over_horizon(
    cycles: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    all_dates: list[date],
    horizon: int,
) -> int:
    date_index = {day: index for index, day in enumerate(all_dates)}
    delayed_keys = {
        (str(row["symbol"]), row["date"])
        for row in orders
        if row.get("side") == "SELL"
        and row.get("status") == "REJECTED"
        and row.get("exit_trigger") == "max_holding_sessions"
    }
    unexpected = 0
    for row in cycles:
        if not row["closed"] or int(row["holding_sessions"]) <= horizon:
            continue
        entry_date = date.fromisoformat(row["entry_date"])
        expiry_index = date_index[entry_date] + horizon - 1
        expiry_date = all_dates[expiry_index]
        if (row["symbol"], expiry_date) not in delayed_keys:
            unexpected += 1
    return unexpected


def simulate_account_horizon(
    candidates: pl.DataFrame,
    raw_source: pl.DataFrame,
    all_dates: list[date],
    data_dir: Path,
    horizon: int,
    baseline_path: Path,
    period_name: str,
) -> dict[str, Any]:
    symbols = candidates.get_column("symbol").unique().to_list()
    quotes = account.prepare_quote_panel(
        account.attach_quote_names(raw_source.filter(pl.col("symbol").is_in(symbols)), data_dir)
    )
    grid = daily.build_action_grid(candidates, quotes, all_dates)
    simulation = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=all_dates,
        max_holding_sessions=horizon,
        cooldown_sessions=COOLDOWN_SESSIONS,
    )
    account_daily, stale = account.build_daily_equity(
        simulation, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    doubled = account.simulate_account(
        candidates,
        grid,
        initial_cash=INITIAL_CASH,
        target_positions=TARGET_POSITIONS,
        action_dates=all_dates,
        max_holding_sessions=horizon,
        cooldown_sessions=COOLDOWN_SESSIONS,
        cost_multiplier=2.0,
    )
    doubled_daily, _ = account.build_daily_equity(
        doubled, quotes, all_dates, initial_cash=INITIAL_CASH
    )
    returns = account_daily.get_column("daily_return").drop_nulls().to_list()
    yearly = []
    for year in range(DEVELOPMENT_START.year, DEVELOPMENT_END.year + 1):
        values = (
            account_daily.filter(pl.col("date").dt.year() == year)
            .get_column("daily_return")
            .drop_nulls()
            .to_list()
        )
        yearly.append({"year": year, "account_return": baseline._compound(values)})
    reconstruction = lifecycle.reconstruct_lifecycles(
        simulation["orders"], simulation["settlements"], all_dates, default_family=EVENT_FAMILY
    )
    cycle_summary = lifecycle.summarize_lifecycles(reconstruction["cycles"])
    positive_profits = sorted(
        [
            float(row["cash_pnl"])
            for row in reconstruction["cycles"]
            if row.get("closed") and float(row.get("cash_pnl") or 0.0) > 0
        ],
        reverse=True,
    )
    top5_share = sum(positive_profits[:5]) / sum(positive_profits) if positive_profits else None
    intents = regime.intent_execution_summary(simulation["orders"], all_dates)
    metrics = {
        "complete_round_trips": cycle_summary["closed_cycles"],
        "annualized": shared._annualized(returns),
        "total_return": baseline._compound(returns),
        "max_drawdown": baseline._max_drawdown(returns),
        "positive_years": sum(
            row["account_return"] is not None and row["account_return"] > 0 for row in yearly
        ),
        "yearly": yearly,
        "mean_cash_ratio": account_daily.get_column("cash_ratio").mean(),
        "buy_intent_execution": intents["buy"]["execution_rate"],
        "sell_intent_execution": intents["sell"]["execution_rate"],
        "ending_unresolved_positions": stale["ending_unresolved_positions"],
        "max_cash_reconciliation_error": simulation["max_cash_reconciliation_error"],
        "unexpected_over_horizon_cycles": _unexpected_over_horizon(
            reconstruction["cycles"], simulation["orders"], all_dates, horizon
        ),
        "delayed_over_horizon_cycles": cycle_summary["over_10_cycles"]
        if horizon == 10
        else sum(
            row["closed"] and int(row["holding_sessions"]) > horizon
            for row in reconstruction["cycles"]
        ),
        "max_holding_sessions": cycle_summary["max_holding_sessions"],
        "top5_positive_profit_share": top5_share,
        "max_industry_asset_share": _max_industry_share(
            simulation, candidates, quotes, account_daily, all_dates
        ),
        "microcap_daily_correlation": _microcap_correlation(
            account_daily, baseline_path, period_name
        ),
        "double_cost_total_return": baseline._compound(
            doubled_daily.get_column("daily_return").drop_nulls().to_list()
        ),
        "account_summary": account.account_summary(simulation, account_daily),
        "daily_attempt_execution": account.execution_summary(simulation["orders"]),
        "lifecycle": cycle_summary,
        "lifecycle_issues": reconstruction["issues"],
    }
    return {
        "metrics": metrics,
        "orders": simulation["orders"],
        "settlements": simulation["settlements"],
        "cycles": reconstruction["cycles"],
        "daily_equity": account_daily.to_dicts(),
    }


def _checks(result: dict[str, Any]) -> dict[str, bool]:
    event = result["event_study"]
    account_result = result["account"]
    correlation = account_result.get("microcap_daily_correlation")
    return {
        "at_least_50_complete_round_trips": account_result["complete_round_trips"] >= 50,
        "tradable_rate_at_least_90pct": event["tradable_rate"] >= 0.90,
        "market_benchmark_coverage_at_least_99pct": event["market_benchmark_coverage"] >= 0.99,
        "industry_benchmark_coverage_at_least_99pct": event["industry_benchmark_coverage"] >= 0.99,
        "no_unresolved_event_exits": event["unresolved_exits"] == 0,
        "mean_net_return_at_least_50bp": (event.get("mean_net_return") or -math.inf) >= 0.005,
        "mean_market_excess_at_least_30bp": (event.get("mean_market_excess") or -math.inf) >= 0.003,
        "mean_industry_excess_at_least_30bp": (event.get("mean_industry_excess") or -math.inf)
        >= 0.003,
        "market_excess_cluster_t_at_least_2": (event.get("market_excess_cluster_t") or -math.inf)
        >= 2.0,
        "at_least_5_positive_market_excess_years": event["positive_market_excess_years"] >= 5,
        "max_year_positive_share_at_most_50pct": (event.get("max_year_positive_share") or math.inf)
        <= 0.50,
        "account_annualized_positive": (account_result.get("annualized") or -math.inf) > 0,
        "account_max_drawdown_within_25pct": (account_result.get("max_drawdown") or -math.inf)
        >= -0.25,
        "at_least_5_positive_account_years": account_result["positive_years"] >= 5,
        "buy_intent_execution_at_least_90pct": account_result["buy_intent_execution"] >= 0.90,
        "sell_intent_execution_at_least_90pct": account_result["sell_intent_execution"] >= 0.90,
        "no_unresolved_account_positions": account_result["ending_unresolved_positions"] == 0,
        "cash_reconciled": account_result["max_cash_reconciliation_error"] <= 0.01,
        "holding_contract_satisfied": account_result["unexpected_over_horizon_cycles"] == 0,
        "top5_positive_profit_share_at_most_30pct": (
            account_result.get("top5_positive_profit_share") or math.inf
        )
        <= 0.30,
        "max_industry_asset_share_at_most_25pct": account_result["max_industry_asset_share"]
        <= 0.25,
        "absolute_microcap_daily_correlation_below_0_4": correlation is not None
        and abs(correlation) < 0.4,
        "double_cost_total_return_positive": (
            account_result.get("double_cost_total_return") or -math.inf
        )
        > 0,
    }


def evaluate_development(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions = {}
    promoted = []
    for horizon, result in sorted(results.items(), key=lambda item: int(item[0])):
        checks = _checks(result)
        decisions[horizon] = {
            "passed": all(checks.values()),
            "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed],
        }
        if decisions[horizon]["passed"]:
            promoted.append(int(horizon))
    selected = (
        sorted(
            promoted,
            key=lambda value: (
                -float(results[str(value)]["event_study"]["mean_industry_excess"]),
                value,
            ),
        )[0]
        if promoted
        else None
    )
    return {
        "verdict": ("PROMOTE_HORIZON_TO_VALIDATION" if selected else "REJECT_EVENT_ACCOUNT"),
        "selected_horizon": selected,
        "horizons": decisions,
        "validation_read": False,
        "known_stress_read": False,
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    events, data_audit = load_qualified_events(data_dir, DEVELOPMENT_START, DEVELOPMENT_END)
    raw_all = baseline.load_daily(data_dir, end=DEVELOPMENT_END).filter(
        pl.col("date") >= DEVELOPMENT_START - timedelta(days=45)
    )
    raw_source = main_board.filter_main_board(raw_all)
    all_dates = (
        raw_source.filter(pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END))
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )
    event_panel = forecast.prepare_panel(
        forecast.load_panel(
            data_dir,
            start=DEVELOPMENT_START - timedelta(days=45),
            panel_end=DEVELOPMENT_END,
        ).filter(pl.col("symbol").str.contains(main_board.MAIN_BOARD_PATTERN))
    )
    account_panel = baseline.prepare_panel(baseline.attach_point_in_time_data(raw_source, data_dir))
    membership = industry.load_point_in_time_membership(data_dir)
    baseline_path = data_dir / "research" / MICROCAP_BASELINE
    if not baseline_path.is_file():
        raise ValueError("frozen main-board microcap baseline is required")
    results = {}
    for horizon in HORIZONS:
        candidates, candidate_audit = build_candidates(events, account_panel, all_dates, horizon)
        event_summary, event_details = summarize_event_study(
            events, event_panel, membership, horizon
        )
        account_details = simulate_account_horizon(
            candidates,
            raw_source,
            all_dates,
            data_dir,
            horizon,
            baseline_path,
            "development",
        )
        results[str(horizon)] = {
            "candidate_audit": candidate_audit,
            "event_study": event_summary,
            "event_details": event_details,
            "account": account_details["metrics"],
            "orders": account_details["orders"],
            "settlements": account_details["settlements"],
            "cycles": account_details["cycles"],
            "daily_equity": account_details["daily_equity"],
        }
        gc.collect()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_frozen": "2026-09-04",
        "period": {
            "name": "development",
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "validation_read": False,
            "known_stress_read": False,
        },
        "assumptions": {
            "board_scope": "sh_sz_main_board_only",
            "initial_cash_cny": INITIAL_CASH,
            "target_positions": TARGET_POSITIONS,
            "horizons": list(HORIZONS),
            "cooldown_sessions": COOLDOWN_SESSIONS,
            "minimum_signal_amount_cny": MIN_SIGNAL_AMOUNT,
            "financial_quality": "prior_roe_and_operating_cash_ratios_strictly_positive",
            "reason_class": "OPERATING_only",
            "maximum_one_position_per_sw_l1_industry": True,
        },
        "data": {
            **data_audit,
            "microcap_baseline_path": str(baseline_path),
            "microcap_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        },
        "horizons": results,
        "decision": evaluate_development(results),
    }
    _atomic_json(payload, output)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "period": payload["period"],
                "data": payload["data"],
                "horizons": {
                    key: {
                        "candidate_audit": value["candidate_audit"],
                        "event_study": value["event_study"],
                        "account": value["account"],
                    }
                    for key, value in results.items()
                },
                "decision": payload["decision"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
