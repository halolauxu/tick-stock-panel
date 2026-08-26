"""Serenity-style supply-chain bottleneck research pre-screen.

Only the three scorecard dimensions supported by point-in-time local data are
scored here: demand inflection (15), valuation disconnect (11), and catalyst
timing (10).  The other 64 points require supply-chain and primary-evidence
data, so this strategy deliberately stops at a research candidate list.
"""
from __future__ import annotations

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
)

META = {
    "id": "serenity_bottleneck_research",
    "name": "Serenity 卡点研究初筛",
    "description": (
        "按公告后财务与量价计算 36 分自动初筛; 供应链位置、供应商集中度、"
        "扩产难度和证据质量等 64 分必须另行核验, 不代表已证明卡脖子"
    ),
    "tags": ["产业链", "卡点", "财务", "研究初筛"],
    "version": "1.0.0",
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3,
        "price_max": 300,
        "market_cap_min": 20e8,
        "amount_min": 1e8,
        "exclude_st": True,
        "exclude_new_days": 120,
    },
    "params": [
        {
            "id": "min_revenue_yoy",
            "label": "最低营收增速(%)",
            "type": "float",
            "default": 5.0,
            "min": -20.0,
            "max": 100.0,
            "step": 1.0,
        },
        {
            "id": "min_net_income_yoy",
            "label": "最低净利增速(%)",
            "type": "float",
            "default": 0.0,
            "min": -50.0,
            "max": 200.0,
            "step": 1.0,
        },
        {
            "id": "min_roe",
            "label": "最低 ROE(%)",
            "type": "float",
            "default": 8.0,
            "min": 0.0,
            "max": 40.0,
            "step": 1.0,
        },
        {
            "id": "min_gross_margin",
            "label": "最低毛利率(%)",
            "type": "float",
            "default": 15.0,
            "min": 0.0,
            "max": 80.0,
            "step": 1.0,
        },
        {
            "id": "max_debt_ratio",
            "label": "最高资产负债率(%)",
            "type": "float",
            "default": 75.0,
            "min": 10.0,
            "max": 100.0,
            "step": 1.0,
        },
        {
            "id": "max_pb",
            "label": "最高市净率",
            "type": "float",
            "default": 15.0,
            "min": 0.5,
            "max": 50.0,
            "step": 0.5,
        },
        {
            "id": "min_momentum_60d",
            "label": "最低 60 日动量",
            "type": "float",
            "default": -0.10,
            "min": -0.50,
            "max": 1.00,
            "step": 0.05,
        },
        {
            "id": "entry_auto_score",
            "label": "入选最低自动分(满分36)",
            "type": "float",
            "default": 12.0,
            "min": 0.0,
            "max": 36.0,
            "step": 1.0,
        },
        {
            "id": "exit_auto_score",
            "label": "退出最高自动分(满分36)",
            "type": "float",
            "default": 7.0,
            "min": 0.0,
            "max": 36.0,
            "step": 1.0,
        },
        {
            "id": "top_rank",
            "label": "每日最多入选",
            "type": "int",
            "default": 20,
            "min": 1,
            "max": 100,
            "step": 1,
        },
    ],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 20,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_serenity_auto_screen"]
EXIT_SIGNALS = ["signal_serenity_auto_downgrade"]
STOP_LOSS = -0.10
MAX_HOLD_DAYS = 60

_FINANCIAL_FIELDS = frozenset({
    "pb_latest",
    "roe_latest",
    "gross_margin_latest",
    "revenue_yoy_latest",
    "net_income_yoy_latest",
    "debt_ratio_latest",
})


class SerenityBottleneckResearchMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "amount", "turnover_rate"}) | _FINANCIAL_FIELDS

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 61

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        entry_score = _bounded_float(
            params.get("entry_auto_score", 12.0), "entry_auto_score", 0.0, 36.0
        )
        exit_score = _bounded_float(
            params.get("exit_auto_score", 7.0), "exit_auto_score", 0.0, 36.0
        )
        if exit_score > entry_score:
            raise ValueError("exit_auto_score must not exceed entry_auto_score")
        top_rank = int(params.get("top_rank", 20))
        if not 1 <= top_rank <= 100:
            raise ValueError("top_rank must be between 1 and 100")

        revenue_yoy = _optional_feature(market, "revenue_yoy_latest")
        net_income_yoy = _optional_feature(market, "net_income_yoy_latest")
        roe = _optional_feature(market, "roe_latest")
        gross_margin = _optional_feature(market, "gross_margin_latest")
        debt_ratio = _optional_feature(market, "debt_ratio_latest")
        pb = _optional_feature(market, "pb_latest")
        momentum = matrix_feature(market, "momentum_60d")
        amount_ratio = matrix_feature(market, "amount_ratio_5d")

        valid = np.isfinite(market.close)
        for values in (
            revenue_yoy,
            net_income_yoy,
            roe,
            gross_margin,
            debt_ratio,
            pb,
            momentum,
            amount_ratio,
        ):
            valid &= np.isfinite(values)

        demand_rating = (
            _linear_rating(revenue_yoy, 0.0, 50.0) * np.float32(0.6)
            + _linear_rating(net_income_yoy, 0.0, 50.0) * np.float32(0.4)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            roe_per_pb = roe / pb
        valuation_rating = _linear_rating(roe_per_pb, 1.0, 8.0)
        timing_rating = (
            _linear_rating(momentum, -0.10, 0.40) * np.float32(0.7)
            + _linear_rating(amount_ratio, -0.30, 1.20) * np.float32(0.3)
        )
        score = (
            demand_rating / np.float32(5.0) * np.float32(15.0)
            + valuation_rating / np.float32(5.0) * np.float32(11.0)
            + timing_rating / np.float32(5.0) * np.float32(10.0)
        ).astype(np.float32)
        score[~valid] = np.float32(0.0)

        financial_gate = (
            valid
            & (revenue_yoy >= float(params.get("min_revenue_yoy", 5.0)))
            & (net_income_yoy >= float(params.get("min_net_income_yoy", 0.0)))
            & (roe >= float(params.get("min_roe", 8.0)))
            & (gross_margin >= float(params.get("min_gross_margin", 15.0)))
            & (debt_ratio <= float(params.get("max_debt_ratio", 75.0)))
            & (pb > 0.0)
            & (pb <= float(params.get("max_pb", 15.0)))
            & (momentum >= float(params.get("min_momentum_60d", -0.10)))
        )
        entry = financial_gate & (score >= np.float32(entry_score))
        entry = _limit_top_rank(entry, score, top_rank)
        exit_ = np.isfinite(market.close) & (
            (~financial_gate) | (score <= np.float32(exit_score))
        )
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_serenity_auto_screen",),
            exit_signal_ids=("signal_serenity_auto_downgrade",),
        )


def _linear_rating(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    scaled = (values - np.float32(lower)) / np.float32(upper - lower) * np.float32(5.0)
    return np.clip(scaled, np.float32(0.0), np.float32(5.0)).astype(np.float32)


def _optional_feature(market: MarketDataMatrix, name: str) -> np.ndarray:
    """Keep absent financial evidence unknown instead of failing or filling zero."""
    if name in market.fields:
        return market.field(name)
    return np.full(market.shape, np.nan, dtype=np.float32)


def _bounded_float(value: object, name: str, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
    return number


def _limit_top_rank(
    eligible: np.ndarray,
    score: np.ndarray,
    top_rank: int,
) -> np.ndarray:
    result = np.zeros(eligible.shape, dtype=bool)
    for time_id in range(eligible.shape[0]):
        asset_ids = np.flatnonzero(eligible[time_id])
        if asset_ids.size <= top_rank:
            result[time_id, asset_ids] = True
            continue
        order = np.argsort(-score[time_id, asset_ids], kind="stable")[:top_rank]
        result[time_id, asset_ids[order]] = True
    return result


MATRIX_STRATEGY = SerenityBottleneckResearchMatrixStrategy()
