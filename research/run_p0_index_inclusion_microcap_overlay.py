"""Test the frozen main-board index-inclusion cash-window engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import date, timedelta
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
MAX_EXIT_ATTEMPTS = 20
STAGES = {
    "development": (date(2013, 12, 1), date(2020, 12, 31)),
    "validation": (date(2021, 1, 1), date(2023, 12, 31)),
    "known_stress": (date(2024, 1, 1), date(2026, 8, 28)),
}


def filter_main_board_additions(additions: pl.DataFrame) -> pl.DataFrame:
    return additions.filter(
        pl.col("symbol").str.contains(main_board.MAIN_BOARD_PATTERN)
    )


def available_dates(data_dir: Path, start: date, end: date) -> list[date]:
    output = []
    for path in (data_dir / "kline_daily_enriched").glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= day <= end:
            output.append(day)
    return sorted(set(output))


def _first_after(days: list[date], boundary: date) -> date | None:
    return next((day for day in days if day > boundary), None)


def build_cycles(
    additions: pl.DataFrame,
    all_dates: list[date],
    *,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    cycles: list[dict[str, Any]] = []
    scoped = filter_main_board_additions(additions).filter(
        pl.col("announcement_date").is_between(start, end, closed="both")
    )
    for key, frame in scoped.partition_by("cycle_month", as_dict=True).items():
        cycle_month = key[0] if isinstance(key, tuple) else key
        announcement_date = frame["announcement_date"][0]
        effective_date = frame["effective_date"][0]
        entry_date = _first_after(all_dates, announcement_date)
        exit_date = _first_after(all_dates, effective_date)
        if entry_date is None or exit_date is None:
            continue
        cycles.append(
            {
                "cycle_month": cycle_month,
                "announcement_date": announcement_date,
                "effective_date": effective_date,
                "entry_date": entry_date,
                "planned_exit_date": exit_date,
                "symbols": sorted(frame["symbol"].unique().to_list()),
            }
        )
    return sorted(cycles, key=lambda row: row["entry_date"])


def load_quotes(
    data_dir: Path,
    symbols: list[str],
    *,
    start: date,
    end: date,
) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "kline_daily_enriched").glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if start - timedelta(days=10) <= day <= end:
            paths.append(path)
    if not paths or not symbols:
        return pl.DataFrame()
    source = (
        pl.scan_parquet(sorted(paths))
        .select("symbol", "date", "open", "close", "volume", "amount", "raw_close")
        .filter(pl.col("symbol").is_in(symbols))
        .collect(engine="streaming")
    )
    return account.prepare_quote_panel(account.attach_quote_names(source, data_dir))


def _quote_rows(quotes: pl.DataFrame) -> dict[tuple[date, str], dict[str, Any]]:
    return {(row["date"], row["symbol"]): row for row in quotes.to_dicts()}


def _buy_rejection(quote: dict[str, Any] | None, gross: float) -> str | None:
    if quote is None:
        return "missing_market_data"
    if quote.get("is_excluded_name"):
        return "risk_warning"
    if not quote.get("volume") or quote["volume"] <= 0:
        return "suspended"
    raw_open = quote.get("raw_open")
    if raw_open is None or raw_open <= 0:
        return "missing_open"
    limit_up = quote.get("limit_up_price")
    if limit_up is not None and raw_open >= limit_up - 0.005:
        return "limit_up"
    if gross > float(quote.get("amount") or 0.0) * baseline.DAILY_PARTICIPATION:
        return "insufficient_capacity"
    return None


def _sell_rejection(
    position: dict[str, Any], quote: dict[str, Any] | None
) -> str | None:
    if quote is None:
        return "missing_market_data"
    if not quote.get("volume") or quote["volume"] <= 0:
        return "suspended"
    raw_open = quote.get("raw_open")
    if raw_open is None or raw_open <= 0:
        return "missing_open"
    limit_down = quote.get("limit_down_price")
    if limit_down is not None and raw_open <= limit_down + 0.005:
        return "limit_down"
    gross = float(position["units"]) * float(quote["open"])
    if gross > float(quote.get("amount") or 0.0) * baseline.DAILY_PARTICIPATION:
        return "insufficient_capacity"
    return None


def simulate(
    cycles: list[dict[str, Any]],
    quotes: pl.DataFrame,
    all_dates: list[date],
    *,
    initial_cash: float,
) -> dict[str, Any]:
    quote_map = _quote_rows(quotes)
    entries = {row["entry_date"]: row for row in cycles}
    positions: dict[str, dict[str, Any]] = {}
    orders: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    cash = float(initial_cash)
    cash_ledger = float(initial_cash)
    max_cash_error = 0.0

    for day in all_dates:
        for symbol, position in positions.items():
            quote = quote_map.get((day, symbol))
            if quote is not None and quote.get("close") is not None:
                position["last_mark"] = float(quote["close"])

        for symbol in list(positions):
            position = positions[symbol]
            if day < position["planned_exit_date"]:
                continue
            if position["exit_attempts"] >= MAX_EXIT_ATTEMPTS:
                continue
            position["exit_attempts"] += 1
            quote = quote_map.get((day, symbol))
            reason = _sell_rejection(position, quote)
            order = {
                "date": day,
                "cycle_month": position["cycle_month"],
                "symbol": symbol,
                "side": "SELL",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
                "attempt": position["exit_attempts"],
            }
            if reason:
                orders.append(order)
                continue
            gross = float(position["units"]) * float(quote["open"])
            commission_fee = account.commission(gross)
            stamp_rate = (
                baseline.STAMP_TAX_OLD
                if day < baseline.STAMP_TAX_CUT
                else baseline.STAMP_TAX_CURRENT
            )
            stamp_tax = gross * stamp_rate
            slippage = gross * baseline.SLIPPAGE_PCT
            cash_delta = gross - commission_fee - stamp_tax - slippage
            cash += cash_delta
            cash_ledger += cash_delta
            order.update(
                gross=gross,
                commission=commission_fee,
                stamp_tax=stamp_tax,
                slippage=slippage,
                cash_delta=cash_delta,
            )
            orders.append(order)
            del positions[symbol]

        cycle = entries.get(day)
        if cycle is not None:
            symbols = cycle["symbols"]
            pre_open_equity = cash + sum(
                float(position["units"])
                * float(
                    quote_map.get((day, symbol), {}).get("open")
                    or position["last_mark"]
                )
                for symbol, position in positions.items()
            )
            target = pre_open_equity / len(symbols) if symbols else 0.0
            for symbol in symbols:
                quote = quote_map.get((day, symbol))
                raw_open = (
                    float(quote["raw_open"])
                    if quote is not None and quote.get("raw_open")
                    else 0.0
                )
                shares = account.affordable_shares(raw_open, target, cash)
                gross = shares * raw_open
                reason = (
                    "already_held"
                    if symbol in positions
                    else "zero_lot_or_cash"
                    if shares <= 0
                    else _buy_rejection(quote, gross)
                )
                order = {
                    "date": day,
                    "signal_date": cycle["announcement_date"],
                    "cycle_month": cycle["cycle_month"],
                    "symbol": symbol,
                    "side": "BUY",
                    "status": "REJECTED" if reason else "FILLED",
                    "reason": reason,
                    "target_notional": target,
                }
                if reason:
                    orders.append(order)
                    continue
                commission_fee = account.commission(gross)
                slippage = gross * baseline.SLIPPAGE_PCT
                cash_delta = -(gross + commission_fee + slippage)
                cash += cash_delta
                cash_ledger += cash_delta
                units = gross / float(quote["open"])
                positions[symbol] = {
                    "cycle_month": cycle["cycle_month"],
                    "units": units,
                    "raw_shares": shares,
                    "planned_exit_date": cycle["planned_exit_date"],
                    "exit_attempts": 0,
                    "last_mark": float(quote["close"]),
                }
                order.update(
                    raw_shares=shares,
                    gross=gross,
                    commission=commission_fee,
                    stamp_tax=0.0,
                    slippage=slippage,
                    cash_delta=cash_delta,
                )
                orders.append(order)

        position_value = sum(
            float(position["units"]) * float(position["last_mark"])
            for position in positions.values()
        )
        equity = cash + position_value
        daily.append(
            {
                "date": day,
                "cash": cash,
                "position_value": position_value,
                "position_count": len(positions),
                "equity": equity,
                "cash_ratio": cash / equity if equity > 0 else 0.0,
            }
        )
        max_cash_error = max(max_cash_error, abs(cash - cash_ledger))

    return {
        "daily": daily,
        "orders": orders,
        "ending_cash": cash,
        "ending_positions": positions,
        "max_cash_reconciliation_error": max_cash_error,
    }


def execution_summary(simulation: dict[str, Any]) -> dict[str, Any]:
    orders = simulation["orders"]
    output: dict[str, Any] = {}
    for side in ("BUY", "SELL"):
        scoped = [row for row in orders if row["side"] == side]
        intent_keys = {(row["cycle_month"], row["symbol"]) for row in scoped}
        filled_keys = {
            (row["cycle_month"], row["symbol"])
            for row in scoped
            if row["status"] == "FILLED"
        }
        output[side.lower()] = {
            "intents": len(intent_keys),
            "filled": len(filled_keys),
            "execution_rate": len(filled_keys) / len(intent_keys)
            if intent_keys
            else 1.0,
            "attempts": len(scoped),
            "rejection_reasons": dict(
                sorted(
                    Counter(
                        row["reason"] for row in scoped if row.get("reason") is not None
                    ).items()
                )
            ),
        }
    return output


def metrics(simulation: dict[str, Any], initial_cash: float) -> dict[str, Any]:
    daily = pl.DataFrame(simulation["daily"], infer_schema_length=None).with_columns(
        (
            pl.col("equity") / pl.col("equity").shift(1).fill_null(initial_cash) - 1.0
        ).alias("daily_return")
    )
    returns = daily["daily_return"].to_list()
    total = baseline._compound(returns)
    annualized = (
        (1.0 + total) ** (252.0 / len(returns)) - 1.0
        if total is not None and total > -1.0 and returns
        else None
    )
    yearly = []
    for year in sorted(daily["date"].dt.year().unique().to_list()):
        value = baseline._compound(
            daily.filter(pl.col("date").dt.year() == year)["daily_return"].to_list()
        )
        yearly.append({"year": year, "return": value})
    costs = sum(
        float(row.get("commission") or 0.0)
        + float(row.get("stamp_tax") or 0.0)
        + float(row.get("slippage") or 0.0)
        for row in simulation["orders"]
        if row["status"] == "FILLED"
    )
    return {
        "total_return": total,
        "annualized": annualized,
        "max_drawdown": baseline._max_drawdown(returns),
        "positive_years": sum((row["return"] or 0.0) > 0 for row in yearly),
        "yearly": yearly,
        "ending_equity": daily["equity"][-1],
        "mean_cash_ratio": daily["cash_ratio"].mean(),
        "total_costs": costs,
    }


def run_stage(
    additions: pl.DataFrame,
    data_dir: Path,
    stage: str,
) -> dict[str, Any]:
    start, end = STAGES[stage]
    all_dates = available_dates(data_dir, start, end)
    cycles = build_cycles(additions, all_dates, start=start, end=end)
    symbols = sorted({symbol for cycle in cycles for symbol in cycle["symbols"]})
    quotes = load_quotes(data_dir, symbols, start=start, end=end)
    accounts: dict[str, Any] = {}
    for capital in CAPITALS:
        simulation = simulate(cycles, quotes, all_dates, initial_cash=capital)
        accounts[str(int(capital))] = {
            "initial_cash": capital,
            "metrics": metrics(simulation, capital),
            "execution": execution_summary(simulation),
            "integrity": {
                "ending_unresolved_positions": len(simulation["ending_positions"]),
                "max_cash_reconciliation_error": simulation[
                    "max_cash_reconciliation_error"
                ],
            },
            "orders": simulation["orders"] if capital == PRIMARY_CAPITAL else [],
        }
    return {
        "stage": stage,
        "period": {"start": start, "end": end},
        "cycles": cycles,
        "symbols": len(symbols),
        "trading_days": len(all_dates),
        "accounts": accounts,
    }


def evaluate(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    primary = result["accounts"][str(int(PRIMARY_CAPITAL))]
    row = primary["metrics"]
    execution = primary["execution"]
    integrity = primary["integrity"]
    yearly = {item["year"]: item["return"] for item in row["yearly"]}
    common = {
        "buy_execution_at_least_80pct": execution["buy"]["execution_rate"] >= 0.80,
        "sell_execution_at_least_80pct": execution["sell"]["execution_rate"] >= 0.80,
        "no_unresolved_positions": integrity["ending_unresolved_positions"] == 0,
        "cash_reconciled": integrity["max_cash_reconciliation_error"] <= 0.01,
    }
    if stage == "development":
        checks = {
            "annualized_at_least_5pct": (row["annualized"] or -99.0) >= 0.05,
            "max_drawdown_within_15pct": (row["max_drawdown"] or -99.0) >= -0.15,
            "at_least_6_positive_years": row["positive_years"] >= 6,
            **common,
        }
    elif stage == "validation":
        checks = {
            "annualized_at_least_5pct": (row["annualized"] or -99.0) >= 0.05,
            "all_3_years_positive": all(
                (yearly.get(year) or 0.0) > 0 for year in range(2021, 2024)
            ),
            "max_drawdown_within_15pct": (row["max_drawdown"] or -99.0) >= -0.15,
            **common,
        }
    else:
        checks = {
            "2024_2025_2026_positive": all(
                (yearly.get(year) or 0.0) > 0 for year in range(2024, 2027)
            ),
            "2026_return_at_least_25pct": (yearly.get(2026) or -99.0) >= 0.25,
            "max_drawdown_within_20pct": (row["max_drawdown"] or -99.0) >= -0.20,
            **common,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    research_dir = data_dir / "research"
    audit_path = research_dir / "p0_index_inclusion_notice_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "NOTICE_MATCH_SUFFICIENT":
        raise ValueError("extended official notice audit is not sufficient")
    additions = pl.read_parquet(audit["artifact"])
    stages: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for stage in ("development", "validation", "known_stress"):
        if stage == "validation" and not decisions["development"]["passed"]:
            stages[stage] = {"status": "NOT_READ_AFTER_DEVELOPMENT_FAILURE"}
            continue
        if stage == "known_stress" and not decisions.get("validation", {}).get(
            "passed", False
        ):
            stages[stage] = {"status": "NOT_READ_AFTER_VALIDATION_FAILURE"}
            continue
        stage_result = run_stage(additions, data_dir, stage)
        decision = evaluate(stage, stage_result)
        stage_result["decision"] = decision
        stages[stage] = stage_result
        decisions[stage] = decision
    passed = all(decisions.get(stage, {}).get("passed", False) for stage in STAGES)
    payload = {
        "schema_version": "p0-index-inclusion-microcap-overlay-v1",
        "contract_frozen": "2026-09-03",
        "notice_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "contract": {
            "board_scope": "sh_sz_main_board_only",
            "entry": "first_trading_day_after_official_notice_at_open",
            "exit": "first_trading_day_after_effective_close_at_open_with_20_day_retry",
            "allocation": "all_eligible_additions_equal_weight",
            "capital_ladder": list(CAPITALS),
            "daily_participation": baseline.DAILY_PARTICIPATION,
        },
        "notice_data": {
            "matched_cycles": audit["matched_cycles"],
            "rejected_cycles": audit["rejected_cycles"],
            "matched_additions": audit["matched_additions"],
        },
        "stages": stages,
        "decision": (
            "ELIGIBLE_FOR_70_30_MICROCAP_COMBINATION"
            if passed
            else "TERMINATE_INDEX_INCLUSION_ENHANCER"
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
                **payload,
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
        default=Path("/app/data/research/p0_index_inclusion_microcap_overlay_v1.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
