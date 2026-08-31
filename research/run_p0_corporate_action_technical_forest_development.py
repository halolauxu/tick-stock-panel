"""Run the frozen corporate-action-pool technical random-forest study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from research.run_p0_equity_incentive_development import (  # noqa: E402
    categorize_events as categorize_incentives,
)
from research.run_p0_equity_incentive_development import (  # noqa: E402
    load_announcements,
)
from research.run_p0_forecast_drift_development import (  # noqa: E402
    attach_point_in_time_universe,
    prepare_panel,
)
from research.run_p0_holder_increase_development import (  # noqa: E402
    aggregate_events as aggregate_holder_events,
)
from research.run_p0_holder_increase_development import (  # noqa: E402
    load_holder_trades,
)
from research.run_p0_microcap_account import (  # noqa: E402
    _buy_rejection,
    _sell_rejection,
    affordable_shares,
    commission,
)
from research.run_p0_microcap_baseline import (  # noqa: E402
    COMMISSION_PCT,
    DAILY_PARTICIPATION,
    SLIPPAGE_PCT,
    STAMP_TAX_CURRENT,
    STAMP_TAX_CUT,
    STAMP_TAX_OLD,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    categorize_events as categorize_repurchase_events,
)
from research.run_p0_repurchase_drift_development import (  # noqa: E402
    load_repurchase_events,
)

DATA_START = date(2013, 8, 29)
EVENT_START = date(2014, 1, 1)
DEVELOPMENT_START = date(2015, 1, 1)
DEVELOPMENT_END = date(2020, 12, 31)
INITIAL_CAPITALS = (200_000.0, 300_000.0, 500_000.0, 1_000_000.0)
HORIZONS = (3, 5, 10, 20, 40, 60)
ACTIVE_DAYS = 20
TRAINING_DAYS = 60
REFIT_DAYS = 20
MIN_TRAINING_ROWS = 2_000
MIN_TRAINING_DATES = 40
TARGET_POSITIONS = 10
MAX_EXIT_DELAY = 5
RANDOM_SEED = 20_260_831

FEATURE_GROUPS = (
    "macd",
    "rsi",
    "kdj",
    "boll",
    "skew",
    "return",
    "amount",
)
FEATURE_COLUMNS = tuple(
    f"{group}_{h}" for h in HORIZONS for group in FEATURE_GROUPS
)


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_positive_events(data_dir: Path) -> pl.DataFrame:
    """Load the three previously frozen positive corporate-action event pools."""
    holder = (
        aggregate_holder_events(load_holder_trades(data_dir))
        .filter(
            pl.col("category").is_in(
                ["management_increase", "corporate_increase", "personal_increase"]
            )
        )
        .select("symbol", "ann_date", "category")
        .with_columns(pl.lit("holder_increase").alias("family"))
    )
    repurchase = (
        categorize_repurchase_events(load_repurchase_events(data_dir))
        .filter(pl.col("category").is_in(["proposal_approved", "completion"]))
        .select("symbol", "ann_date", "category")
        .with_columns(pl.lit("repurchase").alias("family"))
    )
    incentive = (
        categorize_incentives(load_announcements(data_dir))
        .select("symbol", "ann_date", "category")
        .with_columns(pl.lit("equity_incentive").alias("family"))
    )
    return (
        pl.concat([holder, repurchase, incentive], how="vertical")
        .filter(pl.col("ann_date").is_between(EVENT_START, DEVELOPMENT_END, closed="both"))
        .unique(subset=["symbol", "ann_date", "family"], maintain_order=True)
        .sort(["ann_date", "symbol", "family"])
    )


def load_market_panel(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / "kline_daily_enriched").glob("date=*/part.parquet"))
    if not paths:
        raise ValueError("daily enriched data is required")
    panel = (
        pl.scan_parquet(paths)
        .select(
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover_rate",
            "raw_close",
            "raw_high",
            "raw_low",
        )
        .filter(
            pl.col("date").is_between(DATA_START, DEVELOPMENT_END, closed="both")
            & pl.col("symbol").str.contains(r"^\d{6}\.(?:SH|SZ|BJ)$")
        )
        .collect(engine="streaming")
    )
    return attach_point_in_time_universe(panel, data_dir)


def attach_trade_index(panel: pl.DataFrame) -> pl.DataFrame:
    calendar = panel.select("date").unique().sort("date").with_row_index("trade_index")
    return panel.join(calendar, on="date", how="left").sort(["symbol", "date"])


def build_active_pool(events: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Map after-close announcements to causal 20-decision-day active windows."""
    calendar = calendar.select("date", "trade_index").unique().sort("trade_index")
    dates = calendar.get_column("date").to_list()
    indices = calendar.get_column("trade_index").to_list()
    by_index = dict(zip(indices, dates, strict=True))
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for event in events.iter_rows(named=True):
        announced = event["ann_date"]
        insertion = bisect_left(dates, announced)
        if insertion < len(dates) and dates[insertion] == announced:
            start_position = insertion
        else:
            start_position = insertion - 1
        if start_position < 0:
            continue
        start_index = int(indices[start_position])
        for position in range(start_position, min(start_position + ACTIVE_DAYS, len(dates))):
            current_index = int(indices[position])
            key = (str(event["symbol"]), current_index)
            existing = rows.get(key)
            if existing is None:
                rows[key] = {
                    "symbol": str(event["symbol"]),
                    "trade_index": current_index,
                    "date": by_index[current_index],
                    "event_start_index": start_index,
                    "families": {str(event["family"])},
                }
            else:
                existing["event_start_index"] = min(
                    int(existing["event_start_index"]), start_index
                )
                existing["families"].add(str(event["family"]))
    output = []
    for row in rows.values():
        output.append(
            {
                **row,
                "families": ",".join(sorted(row["families"])),
            }
        )
    if not output:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "trade_index": pl.UInt32,
                "date": pl.Date,
                "event_start_index": pl.Int64,
                "families": pl.Utf8,
            }
        )
    return pl.DataFrame(output, infer_schema_length=None).sort(
        ["trade_index", "event_start_index", "symbol"]
    )


def _rolling_feature_expressions(horizon: int) -> list[pl.Expr]:
    close = pl.col("close")
    ret = pl.col("_daily_return")
    gain = pl.when(ret > 0).then(ret).otherwise(0.0)
    loss = pl.when(ret < 0).then(-ret).otherwise(0.0)
    short = max(2, horizon // 3)
    ema_short = close.ewm_mean(span=short, adjust=False, min_samples=short).over("symbol")
    ema_long = close.ewm_mean(
        span=horizon, adjust=False, min_samples=horizon
    ).over("symbol")
    gain_sum = gain.rolling_sum(window_size=horizon, min_samples=horizon).over("symbol")
    loss_sum = loss.rolling_sum(window_size=horizon, min_samples=horizon).over("symbol")
    rolling_low = pl.col("low").rolling_min(
        window_size=horizon, min_samples=horizon
    ).over("symbol")
    rolling_high = pl.col("high").rolling_max(
        window_size=horizon, min_samples=horizon
    ).over("symbol")
    rolling_mean = close.rolling_mean(
        window_size=horizon, min_samples=horizon
    ).over("symbol")
    rolling_std = close.rolling_std(
        window_size=horizon, min_samples=horizon, ddof=1
    ).over("symbol")
    amount_mean = pl.col("amount").rolling_mean(
        window_size=horizon, min_samples=horizon
    ).over("symbol")
    return [
        ((ema_short - ema_long) / close).alias(f"macd_{horizon}"),
        (gain_sum / (gain_sum + loss_sum)).alias(f"rsi_{horizon}"),
        ((close - rolling_low) / (rolling_high - rolling_low)).alias(
            f"kdj_{horizon}"
        ),
        ((close - rolling_mean) / rolling_std).alias(f"boll_{horizon}"),
        ret.rolling_skew(window_size=horizon, bias=False)
        .over("symbol")
        .alias(f"skew_{horizon}"),
        (close / close.shift(horizon).over("symbol") - 1.0).alias(
            f"return_{horizon}"
        ),
        (pl.col("amount") / amount_mean - 1.0).alias(f"amount_{horizon}"),
    ]


def build_feature_panel(panel: pl.DataFrame, active_pool: pl.DataFrame) -> pl.DataFrame:
    """Build causal features and next-open-to-next-open labels for active rows."""
    work = attach_trade_index(panel).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias(
            "_daily_return"
        ),
        pl.col("trade_index").shift(-1).over("symbol").alias("_next_index_1"),
        pl.col("trade_index").shift(-2).over("symbol").alias("_next_index_2"),
        pl.col("open").shift(-1).over("symbol").alias("_next_open_1"),
        pl.col("open").shift(-2).over("symbol").alias("_next_open_2"),
    )
    valid_label = (
        (pl.col("_next_index_1") == pl.col("trade_index") + 1)
        & (pl.col("_next_index_2") == pl.col("trade_index") + 2)
        & (pl.col("_next_open_1") > 0)
        & (pl.col("_next_open_2") > 0)
    )
    work = work.with_columns(
        pl.when(valid_label)
        .then(pl.col("_next_open_2") / pl.col("_next_open_1") - 1.0)
        .otherwise(None)
        .alias("forward_open_return")
    )
    market = (
        work.filter(
            ~pl.col("excluded_name").fill_null(True)
            & pl.col("raw_close").is_between(3.0, 300.0, closed="both")
            & (pl.col("amount").fill_null(0) >= 20_000_000.0)
            & pl.col("forward_open_return").is_not_null()
        )
        .group_by("trade_index")
        .agg(
            pl.col("forward_open_return").median().alias("market_forward_median"),
            pl.len().alias("market_symbols"),
        )
    )
    active = active_pool.select(
        "symbol", "trade_index", "date", "event_start_index", "families"
    )
    base = active.join(
        work.select(
            "symbol",
            "trade_index",
            "forward_open_return",
            "excluded_name",
            "raw_close",
            "amount",
        ),
        on=["symbol", "trade_index"],
        how="left",
    ).join(market, on="trade_index", how="left")
    for horizon in HORIZONS:
        full = work.select(
            "symbol", "trade_index", *_rolling_feature_expressions(horizon)
        )
        base = base.join(full, on=["symbol", "trade_index"], how="left")
    base = base.with_columns(
        pl.when(pl.col(column).is_finite())
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in FEATURE_COLUMNS
    )
    rank_expressions = []
    for column in FEATURE_COLUMNS:
        count = pl.col(column).count().over("trade_index")
        rank = pl.col(column).rank(method="average").over("trade_index")
        rank_expressions.append(
            pl.when(pl.col(column).is_not_null() & (count > 1))
            .then((rank - 1.0) / (count - 1.0) - 0.5)
            .when(pl.col(column).is_not_null())
            .then(0.0)
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias(column)
        )
    return (
        base.with_columns(rank_expressions)
        .with_columns(
            (pl.col("forward_open_return") - pl.col("market_forward_median")).alias(
                "label_excess"
            )
        )
        .sort(["trade_index", "symbol"])
    )


def training_window(current_index: int) -> tuple[int, int]:
    """Return inclusive indices; t-1 is excluded because its T+1 exit is unknown."""
    return current_index - TRAINING_DAYS - 1, current_index - 2


def _rank_correlation(scores: np.ndarray, labels: np.ndarray) -> float | None:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if int(mask.sum()) < 5:
        return None
    scores = scores[mask]
    labels = labels[mask]
    score_ranks = pl.Series(scores).rank(method="average").to_numpy()
    label_ranks = pl.Series(labels).rank(method="average").to_numpy()
    if float(np.std(score_ranks)) == 0 or float(np.std(label_ranks)) == 0:
        return None
    value = float(np.corrcoef(score_ranks, label_ranks)[0, 1])
    return value if math.isfinite(value) else None


def default_model_factory() -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=30,
        max_features=1.0 / 3.0,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def generate_predictions(
    features: pl.DataFrame,
    *,
    model_factory: Callable[[], Any] = default_model_factory,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Generate truly walk-forward scores; each fit only sees fully known labels."""
    development = features.filter(
        pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
    )
    indices = sorted(development.get_column("trade_index").unique().to_list())
    if not indices:
        return pl.DataFrame(), {}
    # Reserve the next-open entry, T+1 exit, and the full frozen sell-delay
    # window inside development. This is a calendar boundary, not an outcome
    # filter, and prevents incomplete final-period trades from entering evidence.
    evaluation_end_index = int(indices[-1]) - (MAX_EXIT_DELAY + 2)
    development = development.filter(pl.col("trade_index") <= evaluation_end_index)
    indices = [int(index) for index in indices if int(index) <= evaluation_end_index]
    if not indices:
        return pl.DataFrame(), {}
    first_index = int(indices[0])
    model: Any | None = None
    fit_id = -1
    next_refit = first_index
    predictions: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    importance_total = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    importance_fits = 0
    for current_index in indices:
        current_index = int(current_index)
        if current_index >= next_refit:
            lower, upper = training_window(current_index)
            training = features.filter(
                pl.col("trade_index").is_between(lower, upper, closed="both")
                & pl.col("label_excess").is_not_null()
            )
            fit_id += 1
            model = None
            if (
                training.height >= MIN_TRAINING_ROWS
                and training.get_column("trade_index").n_unique()
                >= MIN_TRAINING_DATES
            ):
                x = training.select(FEATURE_COLUMNS).to_numpy()
                y = training.get_column("label_excess").to_numpy()
                model = model_factory()
                model.fit(x, y)
                importance = getattr(model, "feature_importances_", None)
                if importance is not None and len(importance) == len(FEATURE_COLUMNS):
                    importance_total += np.asarray(importance, dtype=float)
                    importance_fits += 1
            fits.append(
                {
                    "fit_id": fit_id,
                    "decision_index": current_index,
                    "training_start_index": lower,
                    "training_end_index": upper,
                    "training_rows": training.height,
                    "training_dates": training.get_column("trade_index").n_unique()
                    if training.height
                    else 0,
                    "fitted": model is not None,
                }
            )
            next_refit = current_index + REFIT_DAYS
        if model is None:
            continue
        current = development.filter(pl.col("trade_index") == current_index)
        if current.is_empty():
            continue
        scores = model.predict(current.select(FEATURE_COLUMNS).to_numpy())
        for row, score in zip(current.iter_rows(named=True), scores, strict=True):
            predictions.append(
                {
                    "date": row["date"],
                    "trade_index": current_index,
                    "symbol": row["symbol"],
                    "event_start_index": row["event_start_index"],
                    "families": row["families"],
                    "score": float(score),
                    "label_excess": row["label_excess"],
                    "fit_id": fit_id,
                }
            )
    prediction_frame = (
        pl.DataFrame(predictions, infer_schema_length=None)
        if predictions
        else pl.DataFrame()
    )
    daily_ics: list[dict[str, Any]] = []
    if not prediction_frame.is_empty():
        for key, day in prediction_frame.partition_by("trade_index", as_dict=True).items():
            value = _rank_correlation(
                day.get_column("score").to_numpy(),
                day.get_column("label_excess").fill_null(float("nan")).to_numpy(),
            )
            if value is not None:
                daily_ics.append(
                    {
                        "trade_index": key[0] if isinstance(key, tuple) else key,
                        "fit_id": int(day.get_column("fit_id")[0]),
                        "rank_ic": value,
                    }
                )
    block_ics: list[float] = []
    if daily_ics:
        block_frame = (
            pl.DataFrame(daily_ics)
            .group_by("fit_id")
            .agg(pl.col("rank_ic").mean().alias("block_ic"))
        )
        block_ics = [float(value) for value in block_frame["block_ic"].to_list()]
    mean_ic = float(np.mean(block_ics)) if block_ics else None
    std_ic = float(np.std(block_ics, ddof=1)) if len(block_ics) > 1 else None
    importance = (
        importance_total / importance_fits
        if importance_fits
        else np.zeros(len(FEATURE_COLUMNS))
    )
    group_importance = {
        group: float(
            sum(
                importance[index]
                for index, column in enumerate(FEATURE_COLUMNS)
                if column.startswith(f"{group}_")
            )
        )
        for group in FEATURE_GROUPS
    }
    return prediction_frame, {
        "fits": fits,
        "fitted_models": sum(bool(row["fitted"]) for row in fits),
        "prediction_rows": prediction_frame.height,
        "prediction_dates": prediction_frame.get_column("trade_index").n_unique()
        if not prediction_frame.is_empty()
        else 0,
        "daily_ic_observations": len(daily_ics),
        "block_ic_mean": mean_ic,
        "block_ic_std": std_ic,
        "block_ic_ir": mean_ic / std_ic if mean_ic is not None and std_ic else None,
        "group_feature_importance": group_importance,
        "maximum_group_feature_importance": max(group_importance.values(), default=None),
    }


def rank_candidates(predictions: pl.DataFrame, *, control: bool) -> pl.DataFrame:
    if predictions.is_empty():
        return predictions
    if control:
        sort_columns = ["trade_index", "event_start_index", "symbol"]
        descending = [False, False, False]
    else:
        sort_columns = ["trade_index", "score", "symbol"]
        descending = [False, True, False]
    return (
        predictions.sort(sort_columns, descending=descending)
        .with_columns(pl.int_range(pl.len()).over("trade_index").alias("rank"))
        .filter(pl.col("rank") < TARGET_POSITIONS)
        .with_columns((pl.col("trade_index") + 1).alias("entry_index"))
    )


def build_quote_lookup(
    panel: pl.DataFrame,
    ranked_frames: list[pl.DataFrame],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[date]]:
    execution = prepare_panel(panel)
    calendar = execution.select("date", "trade_index").unique().sort("trade_index")
    all_dates = calendar.get_column("date").to_list()
    required: list[dict[str, Any]] = []
    for frame in ranked_frames:
        if frame.is_empty():
            continue
        for row in frame.select("symbol", "entry_index").unique().iter_rows(named=True):
            for offset in range(MAX_EXIT_DELAY + 2):
                required.append(
                    {
                        "symbol": row["symbol"],
                        "trade_index": int(row["entry_index"]) + offset,
                    }
                )
    if not required:
        return {}, all_dates
    wanted = pl.DataFrame(required, infer_schema_length=None).unique()
    quotes = wanted.join(execution, on=["symbol", "trade_index"], how="left")
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in quotes.iter_rows(named=True):
        output[(str(row["symbol"]), int(row["trade_index"]))] = {
            **row,
            "exact_quote": row.get("date") is not None,
            "entry_amount": row.get("amount"),
            "entry_volume": row.get("volume"),
            "is_excluded_name": row.get("excluded_name"),
        }
    return output, all_dates


def _stamp_tax(trade_date: date) -> float:
    return STAMP_TAX_OLD if trade_date < STAMP_TAX_CUT else STAMP_TAX_CURRENT


def simulate_account(
    ranked: pl.DataFrame,
    quote_lookup: dict[tuple[str, int], dict[str, Any]],
    calendar_dates: list[date],
    initial_capital: float,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Run sell-first daily T+1 execution with immutable cash/trade records."""
    candidates: dict[int, list[dict[str, Any]]] = {}
    if not ranked.is_empty():
        for key, frame in ranked.partition_by("entry_index", as_dict=True).items():
            index = int(key[0] if isinstance(key, tuple) else key)
            candidates[index] = frame.sort("rank").to_dicts()
    positions: dict[str, dict[str, Any]] = {}
    cash = float(initial_capital)
    orders: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    position_id = 0
    start_index = min(candidates, default=0)
    end_index = min(len(calendar_dates) - 1, max(candidates, default=-1) + MAX_EXIT_DELAY + 2)
    for trade_index in range(start_index, end_index + 1):
        trade_date = calendar_dates[trade_index]
        for symbol, position in positions.items():
            quote = quote_lookup.get((symbol, trade_index))
            if quote and quote.get("close") is not None:
                position["last_mark"] = float(quote["close"])
        pre_open_equity = cash
        for symbol, position in positions.items():
            quote = quote_lookup.get((symbol, trade_index))
            mark = (
                float(quote["open"])
                if quote and quote.get("exact_quote") and quote.get("open")
                else float(position["last_mark"])
            )
            pre_open_equity += position["units"] * mark

        for symbol in list(positions):
            position = positions[symbol]
            if trade_index < position["planned_exit_index"] or position.get("unresolved"):
                continue
            quote = quote_lookup.get((symbol, trade_index))
            reason = _sell_rejection(position, quote)
            delay = trade_index - int(position["planned_exit_index"])
            if reason:
                unresolved = delay >= MAX_EXIT_DELAY
                position["unresolved"] = unresolved
                orders.append(
                    {
                        "date": trade_date,
                        "trade_index": trade_index,
                        "position_id": position["position_id"],
                        "symbol": symbol,
                        "side": "SELL",
                        "status": "UNRESOLVED" if unresolved else "REJECTED",
                        "reason": reason,
                        "delay": delay,
                    }
                )
                continue
            gross = float(position["units"]) * float(quote["open"])
            commission_fee = commission(gross)
            stamp_tax = gross * _stamp_tax(trade_date)
            slippage = gross * SLIPPAGE_PCT
            cash_delta = gross - commission_fee - stamp_tax - slippage
            cash += cash_delta
            net_pnl = cash_delta - float(position["cash_out"])
            record = {
                "date": trade_date,
                "trade_index": trade_index,
                "position_id": position["position_id"],
                "symbol": symbol,
                "side": "SELL",
                "status": "FILLED",
                "reason": None,
                "delay": delay,
                "gross": gross,
                "commission": commission_fee,
                "stamp_tax": stamp_tax,
                "slippage": slippage,
                "cash_delta": cash_delta,
                "net_pnl": net_pnl,
                "entry_date": position["entry_date"],
            }
            orders.append(record)
            completed.append(record)
            del positions[symbol]

        slots = max(0, TARGET_POSITIONS - len(positions))
        target_notional = pre_open_equity / TARGET_POSITIONS if pre_open_equity > 0 else 0.0
        for candidate in candidates.get(trade_index, []):
            if slots <= 0:
                break
            symbol = str(candidate["symbol"])
            if symbol in positions:
                continue
            quote = quote_lookup.get((symbol, trade_index))
            raw_open = (
                float(quote["raw_open"])
                if quote and quote.get("raw_open") is not None
                else 0.0
            )
            shares = affordable_shares(raw_open, target_notional, cash)
            gross = shares * raw_open
            reason = "zero_lot_or_cash" if shares <= 0 else _buy_rejection(quote, gross)
            order = {
                "date": trade_date,
                "trade_index": trade_index,
                "symbol": symbol,
                "side": "BUY",
                "status": "REJECTED" if reason else "FILLED",
                "reason": reason,
                "rank": int(candidate["rank"]),
                "score": candidate.get("score"),
            }
            if reason:
                orders.append(order)
                continue
            commission_fee = commission(gross)
            slippage = gross * SLIPPAGE_PCT
            cash_out = gross + commission_fee + slippage
            cash -= cash_out
            position_id += 1
            adjusted_open = float(quote["open"])
            positions[symbol] = {
                "position_id": position_id,
                "symbol": symbol,
                "entry_date": trade_date,
                "entry_index": trade_index,
                "planned_exit_index": trade_index + 1,
                "units": gross / adjusted_open,
                "raw_shares": shares,
                "cash_out": cash_out,
                "last_mark": float(quote["close"]),
                "unresolved": False,
            }
            order.update(
                position_id=position_id,
                raw_shares=shares,
                gross=gross,
                commission=commission_fee,
                stamp_tax=0.0,
                slippage=slippage,
                cash_delta=-cash_out,
            )
            orders.append(order)
            slots -= 1

        close_equity = cash
        stale_positions = 0
        for symbol, position in positions.items():
            quote = quote_lookup.get((symbol, trade_index))
            if quote and quote.get("close") is not None:
                mark = float(quote["close"])
                position["last_mark"] = mark
            else:
                mark = float(position["last_mark"])
                stale_positions += 1
            close_equity += float(position["units"]) * mark
        daily.append(
            {
                "date": trade_date,
                "trade_index": trade_index,
                "cash": cash,
                "equity": close_equity,
                "positions": len(positions),
                "stale_positions": stale_positions,
            }
        )
    daily_frame = pl.DataFrame(daily, infer_schema_length=None).sort("trade_index")
    if daily_frame.height:
        daily_frame = daily_frame.with_columns(
            (
                pl.col("equity")
                / pl.col("equity").shift(1).fill_null(initial_capital)
                - 1.0
            ).alias("daily_return")
        )
    attempted_buys = [row for row in orders if row["side"] == "BUY"]
    filled_buys = [row for row in attempted_buys if row["status"] == "FILLED"]
    total_return = (
        float(daily_frame.get_column("equity")[-1]) / initial_capital - 1.0
        if daily_frame.height
        else 0.0
    )
    annualized = (
        (1.0 + total_return) ** (252.0 / daily_frame.height) - 1.0
        if daily_frame.height and total_return > -1.0
        else None
    )
    peak = float(initial_capital)
    max_drawdown = 0.0
    for equity in daily_frame.get_column("equity").to_list() if daily_frame.height else []:
        peak = max(peak, float(equity))
        max_drawdown = min(max_drawdown, float(equity) / peak - 1.0)
    yearly = []
    positive_years = 0
    if daily_frame.height:
        for year in sorted(daily_frame.get_column("date").dt.year().unique().to_list()):
            scoped = daily_frame.filter(pl.col("date").dt.year() == year)
            previous = (
                initial_capital
                if scoped.get_column("trade_index")[0] == daily_frame.get_column("trade_index")[0]
                else float(
                    daily_frame.filter(
                        pl.col("trade_index") < scoped.get_column("trade_index")[0]
                    ).get_column("equity")[-1]
                )
            )
            year_return = float(scoped.get_column("equity")[-1]) / previous - 1.0
            positive_years += int(year_return > 0)
            yearly.append({"year": int(year), "return": year_return})
    positive_year_pnl: dict[int, float] = defaultdict(float)
    for row in completed:
        if float(row["net_pnl"]) > 0:
            positive_year_pnl[row["date"].year] += float(row["net_pnl"])
    total_positive = sum(positive_year_pnl.values())
    largest_year_share = (
        max(positive_year_pnl.values()) / total_positive if total_positive else None
    )
    costs = sum(
        float(row.get("commission") or 0.0)
        + float(row.get("stamp_tax") or 0.0)
        + float(row.get("slippage") or 0.0)
        for row in orders
        if row["status"] == "FILLED"
    )
    cash_deltas_by_index: dict[int, float] = defaultdict(float)
    for row in orders:
        if row["status"] == "FILLED":
            cash_deltas_by_index[int(row["trade_index"])] += float(row["cash_delta"])
    reconciled_cash = float(initial_capital)
    max_ledger_error = 0.0
    for row in daily:
        reconciled_cash += cash_deltas_by_index[int(row["trade_index"])]
        max_ledger_error = max(
            max_ledger_error, abs(reconciled_cash - float(row["cash"]))
        )
    attempted_orders = [
        row for row in orders if row["side"] in {"BUY", "SELL"}
    ]
    filled_orders = [row for row in attempted_orders if row["status"] == "FILLED"]
    capacity_rejects = sum(
        row.get("reason") == "insufficient_capacity" for row in attempted_orders
    )
    summary = {
        "initial_capital": initial_capital,
        "ending_equity": float(daily_frame.get_column("equity")[-1])
        if daily_frame.height
        else initial_capital,
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": max_drawdown,
        "positive_years": positive_years,
        "yearly": yearly,
        "attempted_buys": len(attempted_buys),
        "filled_buys": len(filled_buys),
        "buy_execution_rate": len(filled_buys) / len(attempted_buys)
        if attempted_buys
        else 0.0,
        "attempted_orders": len(attempted_orders),
        "filled_orders": len(filled_orders),
        "execution_rate": len(filled_orders) / len(attempted_orders)
        if attempted_orders
        else 0.0,
        "capacity_rejects": capacity_rejects,
        "capacity_rejection_rate": capacity_rejects / len(attempted_orders)
        if attempted_orders
        else 0.0,
        "completed_sells": len(completed),
        "ending_positions": len(positions),
        "unresolved_exits": len(positions),
        "largest_positive_year_share": largest_year_share,
        "total_cost": costs,
        "maximum_cash_ledger_error": max_ledger_error,
        "buy_reject_reasons": dict(
            Counter(row["reason"] for row in attempted_buys if row.get("reason"))
        ),
        "sell_reject_reasons": dict(
            Counter(
                row["reason"]
                for row in orders
                if row["side"] == "SELL" and row.get("reason")
            )
        ),
        "return_observations": daily_frame.height,
    }
    records = pl.DataFrame(orders, infer_schema_length=None) if orders else pl.DataFrame()
    return summary, records


def market_benchmark(
    panel: pl.DataFrame,
    *,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    work = attach_trade_index(panel).sort(["symbol", "trade_index"]).with_columns(
        pl.col("trade_index").shift(1).over("symbol").alias("_prior_index"),
        pl.col("close").shift(1).over("symbol").alias("_prior_close"),
    )
    daily = (
        work.filter(
            pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
            & pl.col("trade_index").is_between(start_index, end_index, closed="both")
            & (pl.col("trade_index") == pl.col("_prior_index") + 1)
            & ~pl.col("excluded_name").fill_null(True)
            & (pl.col("amount").fill_null(0) >= 20_000_000.0)
            & (pl.col("_prior_close") > 0)
        )
        .with_columns((pl.col("close") / pl.col("_prior_close") - 1.0).alias("return"))
        .group_by("date")
        .agg(pl.col("return").mean().alias("return"))
        .sort("date")
    )
    compounded = float(np.prod(1.0 + np.asarray(daily["return"].to_list())) - 1.0)
    annualized = (1.0 + compounded) ** (252.0 / daily.height) - 1.0
    return {
        "trading_days": daily.height,
        "total_return": compounded,
        "annualized_return": annualized,
    }


def evaluate_gate(
    candidate: dict[str, Any],
    control: dict[str, Any],
    benchmark: dict[str, Any],
    model_audit: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "annualized_at_least_50pct": (candidate.get("annualized_return") or -math.inf) >= 0.50,
        "market_excess_at_least_20pp": (
            (candidate.get("annualized_return") or -math.inf)
            - (benchmark.get("annualized_return") or math.inf)
            >= 0.20
        ),
        "event_control_increment_at_least_15pp": (
            (candidate.get("annualized_return") or -math.inf)
            - (control.get("annualized_return") or math.inf)
            >= 0.15
        ),
        "max_drawdown_at_least_minus_25pct": candidate.get("max_drawdown", -math.inf) >= -0.25,
        "positive_years_at_least_4": candidate.get("positive_years", 0) >= 4,
        "completed_sells_at_least_500": candidate.get("completed_sells", 0) >= 500,
        "execution_rate_at_least_90pct": candidate.get("execution_rate", 0.0) >= 0.90,
        "no_unresolved_exits": candidate.get("unresolved_exits", 1) == 0,
        "block_ic_mean_at_least_003": (model_audit.get("block_ic_mean") or -math.inf) >= 0.03,
        "block_ic_ir_at_least_05": (model_audit.get("block_ic_ir") or -math.inf) >= 0.5,
        "feature_group_importance_at_most_50pct": (
            model_audit.get("maximum_group_feature_importance") or math.inf
        )
        <= 0.50,
        "largest_positive_year_share_at_most_40pct": (
            candidate.get("largest_positive_year_share") or math.inf
        )
        <= 0.40,
        "cash_ledger_error_at_most_one_cent": candidate.get(
            "maximum_cash_ledger_error", math.inf
        )
        <= 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
    }


def run(data_dir: Path, output: Path, artifact_dir: Path) -> dict[str, Any]:
    events = load_positive_events(data_dir)
    panel = load_market_panel(data_dir)
    indexed = attach_trade_index(panel)
    calendar = indexed.select("date", "trade_index").unique().sort("trade_index")
    active = build_active_pool(events, calendar)
    features = build_feature_panel(panel, active)
    predictions, model_audit = generate_predictions(features)
    if predictions.is_empty():
        raise RuntimeError("no walk-forward predictions were generated")
    prediction_indices = predictions.get_column("trade_index").unique().to_list()
    control_source = features.filter(
        pl.col("date").is_between(DEVELOPMENT_START, DEVELOPMENT_END, closed="both")
        & pl.col("trade_index").is_in(prediction_indices)
    ).select(
        "date",
        "trade_index",
        "symbol",
        "event_start_index",
        "families",
        pl.lit(0.0).alias("score"),
        "label_excess",
    )
    ranked_candidate = rank_candidates(predictions, control=False)
    ranked_control = rank_candidates(control_source, control=True)
    quote_lookup, calendar_dates = build_quote_lookup(
        panel, [ranked_candidate, ranked_control]
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = artifact_dir / "walk_forward_predictions.parquet"
    predictions.write_parquet(predictions_path)
    accounts: dict[str, Any] = {}
    artifact_hashes = {str(predictions_path): _sha256(predictions_path)}
    for capital in INITIAL_CAPITALS:
        candidate, candidate_records = simulate_account(
            ranked_candidate, quote_lookup, calendar_dates, capital
        )
        control, control_records = simulate_account(
            ranked_control, quote_lookup, calendar_dates, capital
        )
        candidate_path = artifact_dir / f"candidate_orders_{int(capital)}.parquet"
        control_path = artifact_dir / f"control_orders_{int(capital)}.parquet"
        candidate_records.write_parquet(candidate_path)
        control_records.write_parquet(control_path)
        artifact_hashes[str(candidate_path)] = _sha256(candidate_path)
        artifact_hashes[str(control_path)] = _sha256(control_path)
        accounts[str(int(capital))] = {
            "candidate": candidate,
            "event_only_control": control,
        }
    benchmark_start = int(ranked_candidate.get_column("entry_index").min())
    benchmark_end = min(
        len(calendar_dates) - 1,
        int(ranked_candidate.get_column("entry_index").max()) + MAX_EXIT_DELAY + 2,
    )
    benchmark = market_benchmark(
        panel,
        start_index=benchmark_start,
        end_index=benchmark_end,
    )
    decision = evaluate_gate(
        accounts["200000"]["candidate"],
        accounts["200000"]["event_only_control"],
        benchmark,
        model_audit,
    )
    capacity_checks = {
        f"capital_{capital}_annualized_at_least_50pct": (
            accounts[str(capital)]["candidate"].get("annualized_return") or -math.inf
        )
        >= 0.50
        for capital in (300_000, 500_000, 1_000_000)
    }
    capacity_checks.update(
        {
            f"capital_{capital}_capacity_rejection_at_most_5pct": accounts[
                str(capital)
            ]["candidate"].get("capacity_rejection_rate", math.inf)
            <= 0.05
            for capital in (300_000, 500_000, 1_000_000)
        }
    )
    if decision["passed"]:
        decision["checks"].update(capacity_checks)
        decision["failed_checks"] = [
            key for key, passed in decision["checks"].items() if not passed
        ]
        decision["passed"] = not decision["failed_checks"]
    decision["capacity_checks"] = capacity_checks
    decision.update(
        {
            "counts_toward_50pct_goal": False,
            "next_step": (
                "freeze_same_model_for_independent_validation"
                if decision["passed"]
                else "terminate_corporate_action_technical_ml_family"
            ),
        }
    )
    payload = {
        "schema_version": "p0-corporate-action-technical-forest-development-v1",
        "contract_frozen": "2026-08-31",
        "period": {
            "event_start": EVENT_START,
            "account_start": DEVELOPMENT_START,
            "development_end": DEVELOPMENT_END,
            "validation_read": False,
            "pressure_read": False,
        },
        "assumptions": {
            "active_days": ACTIVE_DAYS,
            "horizons": HORIZONS,
            "feature_columns": FEATURE_COLUMNS,
            "training_days": TRAINING_DAYS,
            "refit_days": REFIT_DAYS,
            "target_positions": TARGET_POSITIONS,
            "commission_pct": COMMISSION_PCT,
            "slippage_pct_per_side": SLIPPAGE_PCT,
            "daily_participation": DAILY_PARTICIPATION,
        },
        "data": {
            "event_family_rows": events.height,
            "unique_symbol_announcement_days": events.select(
                "symbol", "ann_date"
            ).unique().height,
            "event_symbols": events.get_column("symbol").n_unique(),
            "event_days": events.get_column("ann_date").n_unique(),
            "active_symbol_days": active.height,
            "feature_rows": features.height,
            "feature_dates": features.get_column("trade_index").n_unique(),
        },
        "model": model_audit,
        "benchmark": benchmark,
        "accounts": accounts,
        "artifacts": artifact_hashes,
        "decision": decision,
        "strict_qualified_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**payload, "output": str(output), "sha256": _sha256(output)},
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
            "/app/data/research/p0_corporate_action_technical_forest_development.json"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/app/data/research/corporate_action_technical_forest"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output, args.artifact_dir)


if __name__ == "__main__":
    main()
