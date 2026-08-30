"""行情数据质量审计。

第一项检测是除权污染: 免费档没有除权因子权限时 ``close`` 恒等于 ``raw_close``,
除权日的价格断层会被当成真实跌幅, 制造假新低信号并误触发止损。

判据是交易所涨跌停规则: 一个交易日的跌幅不可能超过该标的所属板块的跌停限制,
超出即说明这一天的价格断层不是交易造成的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from app.price_limits import polars_price_limit_pct

# 交易所按分为单位撮合, 留出一档容差避免浮点与四舍五入造成的误报。
LIMIT_TOLERANCE = 0.005

# A股新股上市首日不设涨跌幅限制, 创业板与科创板前 5 个交易日同样放开;
# 取 10 个交易日覆盖各板块的放开窗口与紧随其后的高波动期。
NEW_LISTING_BARS = 10

_MAX_SAMPLES = 50


@dataclass
class AdjustmentAudit:
    """除权污染审计结果。"""

    total_rows: int = 0
    adjustment_applied: bool = False
    suspect_rows: int = 0
    suspect_symbols: int = 0
    new_listing_rows: int = 0
    suspension_rows: int = 0
    monthly_counts: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SignalContamination:
    """除权污染对策略信号的下游影响。"""

    suspect_rows: int = 0
    fake_new_lows: int = 0
    stop_loss_hits: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _suspect_returns(panel: pl.DataFrame) -> pl.DataFrame:
    """标出跌幅无法由交易解释的行, 并保留窗口计算所需的原始列。

    判定使用复权后的 ``close``: ``raw_close`` 按设计保留除权跳空(交易所涨跌停
    基准价需要原始价), 而策略信号读的是复权价。复权正确时 ``close`` 不应再有
    无法由交易解释的断层; 若仍有, 说明该标的的除权因子缺失或有误。
    """
    price = "close" if "close" in panel.columns else "raw_close"
    frame = (
        panel.with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("date").cast(pl.Date, strict=False),
            pl.col(price).cast(pl.Float64, strict=False).alias("_audit_price"),
        )
        .drop_nulls(["symbol", "date", "_audit_price"])
        .sort(["symbol", "date"])
    )
    if frame.is_empty():
        return frame
    return _with_market_calendar(frame).sort(["symbol", "date"]).with_columns(
        (
            pl.col("_audit_price") / pl.col("_audit_price").shift(1).over("symbol") - 1
        ).alias("return_pct"),
        polars_price_limit_pct(
            pl.col("symbol"),
            pl.col("date"),
            pl.lit(False),
        ).alias("_limit_pct"),
    ).with_columns(
        (
            pl.col("return_pct").is_not_null()
            & (pl.col("return_pct") < -(pl.col("_limit_pct") + LIMIT_TOLERANCE))
        ).alias("_is_suspect"),
        pl.int_range(pl.len()).over("symbol").alias("_bar_index"),
        # 全市场日历上的前一交易日; 与该标的自身的前一交易日不符即为停牌。
        pl.col("date").shift(1).over("symbol").alias("_prev_own_date"),
    ).with_columns(
        (
            pl.col("_prev_own_date").is_not_null()
            & (pl.col("_prev_own_date") != pl.col("_prev_market_date"))
        ).alias("_after_suspension")
    )


def _with_market_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    """给每一行标注全市场日历上的前一交易日。"""
    calendar = (
        frame.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").shift(1).alias("_prev_market_date"))
    )
    return frame.join(calendar, on="date", how="left")


def audit_signal_contamination(
    panel: pl.DataFrame,
    *,
    lookback: int = 60,
    min_samples: int = 20,
    stop_loss: float = -0.06,
    new_listing_bars: int = NEW_LISTING_BARS,
) -> SignalContamination:
    """量化除权假跌幅对策略信号的污染。

    ``fake_new_lows`` 是假跌幅同时把收盘价砸到 ``lookback`` 日区间低点的次数,
    这正是「新低反转」类策略的入场触发条件; ``stop_loss_hits`` 是跌幅穿透
    ``stop_loss`` 的次数, 对应持仓被除权断层误止损。
    """
    required = {"symbol", "date", "close", "raw_close"}
    if panel.is_empty() or not required <= set(panel.columns):
        return SignalContamination()

    frame = _suspect_returns(
        panel.select(
            "symbol",
            "date",
            pl.col("close").cast(pl.Float64, strict=False),
            "raw_close",
        )
    )
    if frame.is_empty():
        return SignalContamination()

    frame = frame.with_columns(
        pl.col("close")
        .rolling_min(lookback, min_samples=min_samples)
        .over("symbol")
        .alias("_window_low")
    )
    # 与 audit_price_adjustment 保持同一口径: 次新股与停牌复牌的断层是真实行情,
    # 由它们触发的止损不是误伤。
    suspects = frame.filter(
        pl.col("_is_suspect")
        & (pl.col("_bar_index") >= new_listing_bars)
        & ~pl.col("_after_suspension")
    )
    if suspects.is_empty():
        return SignalContamination()

    fake_lows = suspects.filter(
        pl.col("_window_low").is_not_null()
        & (pl.col("close") <= pl.col("_window_low"))
    )
    return SignalContamination(
        suspect_rows=suspects.height,
        fake_new_lows=fake_lows.height,
        stop_loss_hits=suspects.filter(pl.col("return_pct") < stop_loss).height,
        samples=(
            fake_lows.sort("return_pct")
            .head(_MAX_SAMPLES)
            .select("symbol", "date", "return_pct")
            .to_dicts()
        ),
    )


def audit_price_adjustment(
    panel: pl.DataFrame,
    *,
    new_listing_bars: int = NEW_LISTING_BARS,
) -> AdjustmentAudit:
    """检查面板中不可能由交易产生的跌幅, 并报告复权是否生效。

    ``panel`` 需要 symbol / date / raw_close 三列; 有 close 列时额外判断复权是否生效。
    ST 标的在 2026-07-06 前跌停是 5%, 这里统一按板块限制(更宽)判定,
    因此结果是保守下界: 只会漏报, 不会虚报。

    ``new_listing_bars`` 是次新股豁免窗口: A股新股上市初期不设涨跌幅限制,
    这些暴跌是合法交易; 被豁免的行数单独记入 ``new_listing_rows``。
    """
    required = {"symbol", "date", "raw_close"}
    if panel.is_empty() or not required <= set(panel.columns):
        return AdjustmentAudit()

    frame = panel.select(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("date").cast(pl.Date, strict=False),
        pl.col("raw_close").cast(pl.Float64, strict=False),
        *(
            [pl.col("close").cast(pl.Float64, strict=False)]
            if "close" in panel.columns
            else []
        ),
    ).drop_nulls(["symbol", "date", "raw_close"])
    if frame.is_empty():
        return AdjustmentAudit()

    adjustment_applied = False
    if "close" in frame.columns:
        both = frame.drop_nulls(["close", "raw_close"])
        if not both.is_empty():
            adjustment_applied = bool(
                ((both["close"] - both["raw_close"]).abs() > 1e-9).any()
            )

    marked = _suspect_returns(frame)
    if marked.is_empty():
        return AdjustmentAudit(
            total_rows=frame.height,
            adjustment_applied=adjustment_applied,
        )
    flagged = marked.filter(pl.col("_is_suspect"))
    new_listings = flagged.filter(pl.col("_bar_index") < new_listing_bars)
    seasoned = flagged.filter(pl.col("_bar_index") >= new_listing_bars)
    suspensions = seasoned.filter(pl.col("_after_suspension"))
    suspects = seasoned.filter(~pl.col("_after_suspension")).sort("return_pct")

    monthly = (
        suspects.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("_month"))
        .group_by("_month")
        .len()
        .sort("_month")
    )
    samples = (
        suspects.head(_MAX_SAMPLES)
        .select("symbol", "date", "return_pct", "raw_close")
        .to_dicts()
    )

    return AdjustmentAudit(
        total_rows=frame.height,
        adjustment_applied=adjustment_applied,
        suspect_rows=suspects.height,
        suspect_symbols=suspects["symbol"].n_unique(),
        new_listing_rows=new_listings.height,
        suspension_rows=suspensions.height,
        monthly_counts={row["_month"]: row["len"] for row in monthly.to_dicts()},
        samples=samples,
    )
