"""Causal multi-horizon labels for Alpha discovery and risk modelling."""
# Requirements: AM-S4-001 through AM-S4-005.
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

SUPPORTED_HORIZONS = (1, 3, 5, 10, 20, 60)


def attach_alpha_labels(
    panel: pl.DataFrame,
    trading_dates: Sequence,
    *,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
    commission_pct: float = 0.0002,
    stamp_tax_pct: float = 0.0005,
    slippage_bps: float = 5.0,
) -> pl.DataFrame:
    """Attach returns, residuals, MFE/MAE, gaps and tradability on a global calendar."""
    selected = tuple(sorted(set(int(value) for value in horizons)))
    unknown = sorted(set(selected) - set(SUPPORTED_HORIZONS))
    if unknown:
        raise ValueError(f"unsupported Alpha label horizons: {unknown}")
    required = {"symbol", "date", "open", "high", "low", "close"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Alpha labels require columns: {missing}")
    calendar = pl.DataFrame(
        {"date": list(trading_dates), "_global_index": range(len(trading_dates))}
    ).with_columns(pl.col("date").cast(pl.Date))
    work = panel.with_columns(pl.col("date").cast(pl.Date)).join(calendar, on="date", how="inner")
    if min(commission_pct, stamp_tax_pct, slippage_bps) < 0:
        raise ValueError("Alpha label costs must not be negative")
    round_trip_cost = commission_pct * 2.0 + stamp_tax_pct + slippage_bps / 10_000.0 * 2.0
    optional = [name for name in ("volume", "signal_limit_up") if name in work.columns]
    key = work.select(
        "symbol", "date", "_global_index", "open", "high", "low", "close", *optional
    )
    for horizon in selected:
        future = key.select(
            "symbol",
            (pl.col("_global_index") - horizon).alias("_global_index"),
            pl.col("date").alias(f"_target_date_{horizon}d"),
            pl.col("close").alias(f"_future_close_{horizon}d"),
            pl.col("open").alias(f"_future_open_{horizon}d"),
            *(
                [pl.col("volume").alias(f"_future_volume_{horizon}d")]
                if "volume" in optional else []
            ),
            *(
                [pl.col("signal_limit_up").alias(f"_future_limit_up_{horizon}d")]
                if "signal_limit_up" in optional else []
            ),
        )
        missing_or_blocked = pl.col(f"_future_open_{horizon}d").is_null()
        if "volume" in optional:
            missing_or_blocked = missing_or_blocked | (
                pl.col(f"_future_volume_{horizon}d").fill_null(0) <= 0
            )
        if "signal_limit_up" in optional:
            missing_or_blocked = missing_or_blocked | pl.col(
                f"_future_limit_up_{horizon}d"
            ).fill_null(False)
        work = work.join(future, on=["symbol", "_global_index"], how="left").with_columns(
            (pl.col(f"_future_close_{horizon}d") / pl.col("close") - 1.0).alias(
                f"target_gross_return_{horizon}d"
            ),
            (pl.col(f"_future_open_{horizon}d") / pl.col("close") - 1.0).alias(
                f"target_gap_{horizon}d"
            ),
            missing_or_blocked.alias(f"target_untradable_{horizon}d"),
        ).with_columns(
            pl.when(~pl.col(f"target_untradable_{horizon}d"))
            .then(pl.col(f"target_gross_return_{horizon}d") - round_trip_cost)
            .otherwise(None)
            .alias(f"target_return_{horizon}d")
        ).with_columns(
            (
                pl.col(f"target_return_{horizon}d")
                - pl.col(f"target_return_{horizon}d").mean().over("date")
            ).alias(f"target_residual_return_{horizon}d")
        )
        path_rows = []
        for offset in range(1, horizon + 1):
            shifted = key.select(
                "symbol",
                (pl.col("_global_index") - offset).alias("_global_index"),
                pl.col("high").alias(f"_high_t{offset}"),
                pl.col("low").alias(f"_low_t{offset}"),
            )
            work = work.join(shifted, on=["symbol", "_global_index"], how="left")
            path_rows.append(offset)
        work = work.with_columns(
            (pl.max_horizontal(*(pl.col(f"_high_t{x}") for x in path_rows)) / pl.col("close") - 1.0).alias(
                f"target_mfe_{horizon}d"
            ),
            (pl.min_horizontal(*(pl.col(f"_low_t{x}") for x in path_rows)) / pl.col("close") - 1.0).alias(
                f"target_mae_{horizon}d"
            ),
        ).drop(
            f"_future_close_{horizon}d",
            f"_future_open_{horizon}d",
            *(
                [f"_future_volume_{horizon}d"] if "volume" in optional else []
            ),
            *(
                [f"_future_limit_up_{horizon}d"] if "signal_limit_up" in optional else []
            ),
            *(f"_high_t{x}" for x in path_rows),
            *(f"_low_t{x}" for x in path_rows),
        )
    return work.drop("_global_index")
