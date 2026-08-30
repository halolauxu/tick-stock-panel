"""Run the frozen discovery-only intraday failed-board reclaim study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from research import run_p0_emotion_limit_up_study as emotion  # noqa: E402
from research.run_p0_forecast_drift_development import (  # noqa: E402
    COMMISSION_PCT,
    SLIPPAGE_PCT,
    STAMP_TAX_CURRENT,
    STAMP_TAX_CUT,
    STAMP_TAX_OLD,
)

CONTEXT_START = date(2025, 7, 1)
DISCOVERY_START = date(2025, 8, 27)
DISCOVERY_END = date(2026, 2, 13)
EXECUTION_END = date(2026, 3, 31)
POSITION_NOTIONAL = 20_000.0
DAILY_PARTICIPATION = 0.01
LOT_SIZE = 100
COOLDOWN_DAYS = 20
MAX_EXIT_DELAY = 20
MIN_PREVIOUS_AMOUNT = 50_000_000.0
MIN_BREAK_PCT_OF_LIMIT = 0.98
MIN_RECLAIM_PCT_OF_LIMIT = 0.99


def stamp_tax_rate(day: date) -> float:
    return STAMP_TAX_OLD if day < STAMP_TAX_CUT else STAMP_TAX_CURRENT


def prepare_context(data_dir: Path) -> pl.DataFrame:
    raw = emotion.load_daily_panel(data_dir, end=EXECUTION_END, start=CONTEXT_START)
    pit = emotion.attach_point_in_time_universe(raw, data_dir)
    panel = emotion.prepare_market_panel(pit).sort(["symbol", "date"])
    return panel.with_columns(
        pl.col("amount").shift(1).over("symbol").alias("previous_amount"),
        pl.col("_global_index").shift(1).over("symbol").alias("previous_global_index"),
        (pl.col("close") / pl.col("raw_close")).alias("adj_factor"),
    ).with_columns(
        (
            (pl.col("previous_global_index") == pl.col("_global_index") - 1)
            & ~pl.col("_is_excluded")
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("previous_amount") >= MIN_PREVIOUS_AMOUNT)
        ).alias("universe_eligible")
    )


def _minute_path(data_dir: Path, day: date) -> Path:
    return data_dir / "kline_minute" / f"date={day.isoformat()}" / "part.parquet"


def _next_minute(previous: datetime, current: datetime) -> bool:
    return current - previous == timedelta(minutes=1)


def detect_reclaim(rows: list[dict[str, Any]], limit_up: float) -> dict[str, Any] | None:
    if not rows or not math.isfinite(limit_up) or limit_up <= 0:
        return None
    ordered = sorted(rows, key=lambda row: row["datetime"])
    touch = None
    for index, row in enumerate(ordered):
        clock = row["datetime"].time()
        if (
            time(9, 31) <= clock <= time(14, 30)
            and float(row.get("high") or 0.0) >= limit_up - 0.005
        ):
            touch = index
            break
    if touch is None:
        return None
    broken = None
    for index in range(touch + 1, len(ordered)):
        if float(ordered[index].get("low") or math.inf) <= (limit_up * MIN_BREAK_PCT_OF_LIMIT):
            broken = index
            break
    if broken is None:
        return None
    reclaim = None
    for index in range(broken + 1, len(ordered) - 1):
        row = ordered[index]
        clock = row["datetime"].time()
        close = float(row.get("close") or 0.0)
        previous_close = float(ordered[index - 1].get("close") or 0.0)
        if (
            clock <= time(14, 49)
            and limit_up * MIN_RECLAIM_PCT_OF_LIMIT <= close <= limit_up - 0.005
            and close > previous_close
        ):
            reclaim = index
            break
    if reclaim is None:
        return None
    entry = ordered[reclaim + 1]
    if not _next_minute(ordered[reclaim]["datetime"], entry["datetime"]):
        return None
    return {
        "first_touch_datetime": ordered[touch]["datetime"],
        "break_datetime": ordered[broken]["datetime"],
        "signal_datetime": ordered[reclaim]["datetime"],
        "entry_datetime": entry["datetime"],
        "entry_open": entry.get("open"),
        "entry_high": entry.get("high"),
        "entry_low": entry.get("low"),
        "entry_close": entry.get("close"),
        "entry_volume": entry.get("volume"),
        "entry_amount": entry.get("amount"),
    }


def load_intraday_events(
    data_dir: Path, context: pl.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = context.filter(
        pl.col("date").is_between(DISCOVERY_START, DISCOVERY_END, closed="both")
        & pl.col("universe_eligible")
        & (pl.col("is_limit_up") | pl.col("is_broken_board"))
    )
    by_day = {
        key[0] if isinstance(key, tuple) else key: group
        for key, group in candidates.partition_by("date", as_dict=True).items()
    }
    detected = []
    missing_partitions = []
    for day, daily in sorted(by_day.items()):
        path = _minute_path(data_dir, day)
        if not path.is_file():
            missing_partitions.append(day)
            continue
        symbols = daily.get_column("symbol").to_list()
        minute = (
            pl.read_parquet(path)
            .filter(pl.col("symbol").is_in(symbols))
            .select(
                "symbol",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            )
            .sort(["symbol", "datetime"])
        )
        groups = {
            key[0] if isinstance(key, tuple) else key: group.to_dicts()
            for key, group in minute.partition_by("symbol", as_dict=True).items()
        }
        for row in daily.to_dicts():
            event = detect_reclaim(groups.get(row["symbol"], []), float(row["limit_up_price"]))
            if event is not None:
                detected.append(
                    {
                        **event,
                        "symbol": row["symbol"],
                        "date": day,
                        "trade_index": row["_global_index"],
                        "limit_up_price": row["limit_up_price"],
                        "entry_adj_factor": row["adj_factor"],
                    }
                )
    last_kept: dict[str, date] = {}
    cooled = []
    for row in sorted(detected, key=lambda value: (value["date"], value["symbol"])):
        previous = last_kept.get(row["symbol"])
        if previous is None or (row["date"] - previous).days >= COOLDOWN_DAYS:
            cooled.append(row)
            last_kept[row["symbol"]] = row["date"]
    return cooled, {
        "daily_touch_candidates": candidates.height,
        "detected_before_cooldown": len(detected),
        "detected_after_cooldown": len(cooled),
        "missing_minute_partitions": [day.isoformat() for day in missing_partitions],
    }


def load_0931(data_dir: Path) -> pl.DataFrame:
    paths = []
    for path in (data_dir / "kline_minute").glob("date=*/part.parquet"):
        try:
            day = date.fromisoformat(path.parent.name.removeprefix("date="))
        except ValueError:
            continue
        if DISCOVERY_START <= day <= EXECUTION_END:
            paths.append(path)
    if not paths:
        raise ValueError("minute execution partitions are required")
    return (
        pl.scan_parquet(sorted(paths), hive_partitioning=False)
        .filter((pl.col("datetime").dt.hour() == 9) & (pl.col("datetime").dt.minute() == 31))
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .select("symbol", "date", "open", "volume", "amount")
        .unique(subset=["symbol", "date"], keep="last")
        .collect(engine="streaming")
    )


def attach_exits(
    events: list[dict[str, Any]],
    minute_0931: pl.DataFrame,
    context: pl.DataFrame,
) -> list[dict[str, Any]]:
    quotes = {(row["symbol"], row["date"]): row for row in minute_0931.to_dicts()}
    daily = {
        (row["symbol"], row["date"]): row
        for row in context.select("symbol", "date", "limit_down_price", "adj_factor").to_dicts()
    }
    calendar = context.select("date", "_global_index").unique().sort("date")
    by_index = {row["_global_index"]: row["date"] for row in calendar.to_dicts()}
    output = []
    for source in events:
        row = dict(source)
        entry_open = float(row.get("entry_open") or 0.0)
        entry_amount = float(row.get("entry_amount") or 0.0)
        entry_valid = (
            entry_open > 0
            and float(row.get("entry_volume") or 0.0) > 0
            and entry_open < float(row["limit_up_price"]) - 0.005
        )
        shares = (
            math.floor(POSITION_NOTIONAL / entry_open / LOT_SIZE) * LOT_SIZE
            if entry_open > 0
            else 0
        )
        entry_capacity = entry_amount * DAILY_PARTICIPATION >= shares * entry_open
        row.update(
            entry_valid=entry_valid,
            shares=shares,
            entry_capacity=entry_capacity,
        )
        exit_row = None
        exit_reason = "missing_or_blocked_exit"
        corporate_action = False
        for delay in range(MAX_EXIT_DELAY + 1):
            exit_date = by_index.get(row["trade_index"] + 1 + delay)
            if exit_date is None:
                continue
            quote = quotes.get((row["symbol"], exit_date))
            ctx = daily.get((row["symbol"], exit_date))
            if quote is None or ctx is None:
                continue
            if abs(float(ctx["adj_factor"]) - float(row["entry_adj_factor"])) > 1e-6:
                corporate_action = True
                exit_reason = "corporate_action"
                break
            price = float(quote.get("open") or 0.0)
            if (
                price <= 0
                or float(quote.get("volume") or 0.0) <= 0
                or price <= float(ctx.get("limit_down_price") or 0.0) + 0.005
            ):
                continue
            if float(quote.get("amount") or 0.0) * DAILY_PARTICIPATION < shares * price:
                exit_reason = "exit_capacity"
                continue
            exit_row = {
                "exit_date": exit_date,
                "exit_open": price,
                "exit_delay": delay,
            }
            exit_reason = "filled"
            break
        row.update(
            **(exit_row or {}),
            exit_reason=exit_reason,
            corporate_action=corporate_action,
        )
        tradable = bool(
            entry_valid
            and entry_capacity
            and shares > 0
            and exit_row is not None
            and not corporate_action
        )
        row["tradable"] = tradable
        if tradable:
            entry_gross = shares * entry_open
            exit_gross = shares * float(row["exit_open"])
            entry_cost = entry_gross * (COMMISSION_PCT + SLIPPAGE_PCT)
            exit_cost = exit_gross * (
                COMMISSION_PCT + SLIPPAGE_PCT + stamp_tax_rate(row["exit_date"])
            )
            row["net_return"] = (exit_gross - exit_cost) / (entry_gross + entry_cost) - 1.0
        else:
            row["net_return"] = None
        output.append(row)
    return output


def build_same_minute_benchmark(
    data_dir: Path,
    events: list[dict[str, Any]],
    minute_0931: pl.DataFrame,
    context: pl.DataFrame,
) -> dict[datetime, float]:
    entry_by_day: dict[date, set[datetime]] = defaultdict(set)
    for row in events:
        entry_by_day[row["date"]].add(row["entry_datetime"])
    quotes = {(row["symbol"], row["date"]): row for row in minute_0931.to_dicts()}
    daily_rows = context.select(
        "symbol",
        "date",
        "_global_index",
        "universe_eligible",
        "adj_factor",
    ).to_dicts()
    daily = {(row["symbol"], row["date"]): row for row in daily_rows}
    calendar = {
        row["_global_index"]: row["date"]
        for row in context.select("date", "_global_index").unique().to_dicts()
    }
    returns: dict[datetime, list[float]] = defaultdict(list)
    for day, timestamps in sorted(entry_by_day.items()):
        path = _minute_path(data_dir, day)
        minute = (
            pl.read_parquet(path)
            .filter(pl.col("datetime").is_in(list(timestamps)))
            .select("symbol", "datetime", "open", "volume")
        )
        for row in minute.to_dicts():
            ctx = daily.get((row["symbol"], day))
            if ctx is None or not ctx["universe_eligible"]:
                continue
            exit_date = calendar.get(ctx["_global_index"] + 1)
            quote = quotes.get((row["symbol"], exit_date))
            exit_ctx = daily.get((row["symbol"], exit_date))
            entry_open = float(row.get("open") or 0.0)
            if (
                quote is None
                or exit_ctx is None
                or entry_open <= 0
                or float(row.get("volume") or 0.0) <= 0
                or float(quote.get("open") or 0.0) <= 0
                or abs(float(exit_ctx["adj_factor"]) - float(ctx["adj_factor"])) > 1e-6
            ):
                continue
            returns[row["datetime"]].append(float(quote["open"]) / entry_open - 1.0)
    return {key: statistics.median(values) for key, values in returns.items() if values}


def summarize(events: list[dict[str, Any]], benchmark: dict[datetime, float]) -> dict[str, Any]:
    tradable = [row for row in events if row["tradable"]]
    benchmarked = []
    for row in tradable:
        market = benchmark.get(row["entry_datetime"])
        if market is None:
            continue
        benchmarked.append(
            {**row, "market_return": market, "excess_return": row["net_return"] - market}
        )
    daily: dict[date, list[dict[str, Any]]] = defaultdict(list)
    monthly: dict[str, list[float]] = defaultdict(list)
    for row in benchmarked:
        daily[row["date"]].append(row)
        monthly[row["date"].strftime("%Y-%m")].append(row["excess_return"])
    daily_excess = [
        statistics.mean(row["excess_return"] for row in values) for values in daily.values()
    ]
    mean_excess = statistics.mean(daily_excess) if daily_excess else None
    t_value = None
    if len(daily_excess) >= 2:
        std = statistics.stdev(daily_excess)
        t_value = mean_excess / (std / math.sqrt(len(daily_excess))) if std else None
    month_means = {key: statistics.mean(values) for key, values in monthly.items()}
    positive = {key: value for key, value in month_means.items() if value > 0}
    positive_total = sum(positive.values())
    largest_positive_share = max(positive.values()) / positive_total if positive_total > 0 else None
    unresolved = sum(
        row["entry_valid"]
        and row["entry_capacity"]
        and row["exit_reason"] == "missing_or_blocked_exit"
        for row in events
    )
    metrics = {
        "events": len(events),
        "tradable_events": len(tradable),
        "benchmarked_events": len(benchmarked),
        "event_days": len(daily),
        "tradable_rate": len(tradable) / len(events) if events else 0.0,
        "entry_capacity_rate": (
            sum(row["entry_capacity"] for row in events) / len(events) if events else 0.0
        ),
        "benchmark_coverage": (len(benchmarked) / len(tradable) if tradable else 0.0),
        "mean_net_return": (
            statistics.mean(row["net_return"] for row in benchmarked) if benchmarked else None
        ),
        "mean_market_return": (
            statistics.mean(row["market_return"] for row in benchmarked) if benchmarked else None
        ),
        "mean_excess_return": mean_excess,
        "daily_cluster_t": t_value,
        "positive_excess_months": len(positive),
        "largest_positive_month_share": largest_positive_share,
        "unresolved_exits": unresolved,
        "rejection_reasons": dict(
            sorted(Counter(row["exit_reason"] for row in events if not row["tradable"]).items())
        ),
        "monthly_excess": month_means,
    }
    checks = {
        "at_least_200_tradable_events": len(tradable) >= 200,
        "at_least_80_event_days": len(daily) >= 80,
        "tradable_rate_at_least_80pct": metrics["tradable_rate"] >= 0.80,
        "entry_capacity_at_least_90pct": metrics["entry_capacity_rate"] >= 0.90,
        "benchmark_coverage_at_least_99pct": metrics["benchmark_coverage"] >= 0.99,
        "mean_net_at_least_50bp": (metrics["mean_net_return"] or -math.inf) >= 0.005,
        "mean_excess_at_least_30bp": (mean_excess or -math.inf) >= 0.003,
        "daily_cluster_t_at_least_2": (t_value or -math.inf) >= 2.0,
        "at_least_four_positive_months": len(positive) >= 4,
        "largest_positive_month_at_most_50pct": (
            largest_positive_share is not None and largest_positive_share <= 0.50
        ),
        "no_unresolved_exit": unresolved == 0,
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "verdict": "PROMOTE_TO_CONFIRMATION" if all(checks.values()) else "TERMINATE",
        "confirmation_read": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    context = prepare_context(data_dir)
    events, event_audit = load_intraday_events(data_dir, context)
    minute_0931 = load_0931(data_dir)
    executed = attach_exits(events, minute_0931, context)
    benchmark = build_same_minute_benchmark(data_dir, executed, minute_0931, context)
    result = summarize(executed, benchmark)
    payload = {
        "schema_version": "p0-intraday-board-reclaim-discovery-v1",
        "contract_frozen": "2026-08-30",
        "period": {
            "discovery_start": DISCOVERY_START,
            "discovery_end": DISCOVERY_END,
            "confirmation_read": False,
        },
        "assumptions": {
            "first_touch_latest": "14:30",
            "minimum_break_from_limit_pct": MIN_BREAK_PCT_OF_LIMIT,
            "minimum_reclaim_of_limit_pct": MIN_RECLAIM_PCT_OF_LIMIT,
            "entry": "next continuous minute open after reclaim confirmation",
            "exit": "next A-share trading-day 09:31 open, delayed up to 20 days",
            "position_notional_cny": POSITION_NOTIONAL,
            "participation_rate": DAILY_PARTICIPATION,
            "cooldown_calendar_days": COOLDOWN_DAYS,
        },
        "data": {
            **event_audit,
            "benchmark_timestamps": len(benchmark),
        },
        "result": result,
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
                "result": result,
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
        default=Path("/app/data/research/p0_intraday_board_reclaim_discovery.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
