"""Independent causal A-share alpha families for controlled research.

Each family owns its entry and exit logic.  They do not inherit the reversal
strategy's holding horizon or exit semantics; those are frozen by the runner.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    build_basic_filter_mask,
    build_matrix_score,
    make_signal_matrix,
    matrix_feature,
    valid_rolling_max,
)
from app.backtest.matrix import valid_shift as shift

META = {
    "id": "independent_alpha_families",
    "name": "独立收益机制研究",
    "description": "趋势、行业轮动、情绪择时、财务质量和事件驱动的独立研究",
    "tags": ["研究", "趋势", "行业", "情绪", "财务", "事件"],
    "research_only": True,
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3,
        "price_max": 300,
        "market_cap_min": 10e8,
        "amount_min": 0.2e8,
        "exclude_st": False,
        "exclude_new_days": 30,
        "boards": ["沪主板", "深主板"],
    },
    "params": [
        {"id": "family", "label": "策略族", "type": "str", "default": "trend"},
        {"id": "eligibility_mode", "label": "历史股票池", "type": "str", "default": "pit"},
        {"id": "risk_on_only", "label": "仅风险偏好阶段", "type": "bool", "default": False},
        {"id": "quality_gate", "label": "财务质量门槛", "type": "bool", "default": False},
        {"id": "sentiment_confirm_days", "label": "情绪确认天数", "type": "int", "default": 1},
        {"id": "sentiment_max_momentum_60d", "label": "最大60日涨幅", "type": "float", "default": 99.0},
        {"id": "sentiment_max_distance_ma20", "label": "最大MA20乖离", "type": "float", "default": 99.0},
        {"id": "sentiment_rsi_max", "label": "最大RSI", "type": "float", "default": 100.0},
        {"id": "sentiment_stock_ma20_exit", "label": "个股跌破MA20退出", "type": "bool", "default": False},
    ],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_independent_alpha_entry"]
EXIT_SIGNALS = ["signal_independent_alpha_exit"]
STOP_LOSS = -0.08
MAX_HOLD_DAYS = 40


def _market_state(market: MarketDataMatrix) -> tuple[np.ndarray, np.ndarray]:
    change = matrix_feature(market, "change_pct")
    ma20 = matrix_feature(market, "ma20")
    symbols = np.asarray(market.symbols)
    main_board = np.array(
        [
            (symbol.endswith(".SH") and symbol.startswith("60"))
            or (
                symbol.endswith(".SZ")
                and symbol.startswith(("000", "001", "002", "003"))
            )
            for symbol in symbols
        ],
        dtype=bool,
    )
    eligible = matrix_feature(market, "pit_eligible") > np.float32(0.5)
    valid_change = np.isfinite(change) & eligible & main_board[None, :]
    count = valid_change.sum(axis=1)
    positive = ((change > 0) & valid_change).sum(axis=1)
    breadth = np.divide(
        positive,
        count,
        out=np.zeros_like(count, dtype=np.float32),
        where=count > 0,
    )
    valid_ma = (
        np.isfinite(ma20)
        & np.isfinite(market.close)
        & eligible
        & main_board[None, :]
    )
    ma_count = valid_ma.sum(axis=1)
    above = ((market.close > ma20) & valid_ma).sum(axis=1)
    above_ma20 = np.divide(
        above,
        ma_count,
        out=np.zeros_like(ma_count, dtype=np.float32),
        where=ma_count > 0,
    )
    return breadth, above_ma20


class IndependentAlphaFamiliesMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset(
            {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turnover_rate",
                "consecutive_limit_ups",
                "pit_eligible",
                "industry_momentum_20d",
                "industry_breadth_5d",
                "roe_latest",
                "net_margin_latest",
                "revenue_yoy_latest",
                "debt_ratio_latest",
            }
        )

    def required_warmup_bars(self, params: dict) -> int:
        family = str(params.get("family", "trend"))
        return 252 if family in {"monthly_momentum", "regime_reversal_quality"} else 60

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        family = str(params.get("family", "trend"))
        if str(params.get("eligibility_mode", "pit")) != "pit":
            raise ValueError("independent families require point-in-time eligibility")

        close = market.close
        ma10 = matrix_feature(market, "ma10")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        high_60d = matrix_feature(market, "high_60d")
        momentum_20d = matrix_feature(market, "momentum_20d")
        momentum_60d = matrix_feature(market, "momentum_60d")
        vol_ratio = matrix_feature(market, "vol_ratio_5d")
        change = matrix_feature(market, "change_pct")
        rsi = matrix_feature(market, "rsi_14")
        industry_momentum = matrix_feature(market, "industry_momentum_20d")
        industry_breadth = matrix_feature(market, "industry_breadth_5d")
        eligible = matrix_feature(market, "pit_eligible") > np.float32(0.5)
        bullish = close > market.open
        prior_close = shift(close, 1)
        prior_ma10 = shift(ma10, 1)
        prior_ma20 = shift(ma20, 1)

        score_override = None
        scoring: dict[str, float]
        directions: dict[str, str]
        if family == "trend":
            entry = (
                (close >= high_60d)
                & (prior_close < shift(high_60d, 1))
                & (ma20 > ma60)
                & (close > ma60)
                & (vol_ratio >= np.float32(1.2))
                & (change > 0)
                & (change <= np.float32(0.07))
            )
            exit_ = close < ma20
            scoring = {
                "momentum_60d": 0.35,
                "industry_momentum_20d": 0.20,
                "close_position": 0.15,
                "atr_pct": 0.15,
                "amihud_20d": 0.15,
            }
            directions = {"atr_pct": "low", "amihud_20d": "low"}
        elif family == "industry_rotation":
            entry = (
                (prior_close < prior_ma20)
                & (close >= ma20)
                & (ma20 > ma60)
                & (momentum_60d >= np.float32(0.10))
                & (industry_momentum > 0)
                & (industry_breadth >= np.float32(0.55))
                & bullish
            )
            exit_ = (close < ma20) | (industry_momentum < 0)
            scoring = {
                "industry_momentum_20d": 0.35,
                "industry_breadth_5d": 0.25,
                "momentum_60d": 0.25,
                "amihud_20d": 0.15,
            }
            directions = {"amihud_20d": "low"}
        elif family == "sentiment_timing":
            breadth, above_ma20 = _market_state(market)
            prior_above = np.empty_like(above_ma20)
            prior_above[0] = np.nan
            prior_above[1:] = above_ma20[:-1]
            confirm_days = int(params.get("sentiment_confirm_days", 1))
            if confirm_days not in {1, 2}:
                raise ValueError("sentiment_confirm_days must be 1 or 2")
            if confirm_days == 1:
                regime_on = (above_ma20 >= np.float32(0.50)) & (
                    prior_above < np.float32(0.50)
                )
            else:
                two_days_prior = np.empty_like(above_ma20)
                two_days_prior[:2] = np.nan
                two_days_prior[2:] = above_ma20[:-2]
                regime_on = (
                    (above_ma20 >= np.float32(0.50))
                    & (prior_above >= np.float32(0.50))
                    & (two_days_prior < np.float32(0.50))
                )
            distance_above_ma20 = close / ma20 - np.float32(1.0)
            risk_on_stock = (
                (close > ma20)
                & (ma20 > ma60)
                & (momentum_60d >= np.float32(0.10))
                & (
                    momentum_60d
                    <= np.float32(params.get("sentiment_max_momentum_60d", 99.0))
                )
                & (
                    distance_above_ma20
                    <= np.float32(params.get("sentiment_max_distance_ma20", 99.0))
                )
                & (rsi <= np.float32(params.get("sentiment_rsi_max", 100.0)))
            )
            entry = regime_on[:, None] & risk_on_stock
            exit_ = np.broadcast_to(
                ((above_ma20 < np.float32(0.40)) | (breadth < np.float32(0.30)))[
                    :, None
                ],
                market.shape,
            ).copy()
            if bool(params.get("sentiment_stock_ma20_exit", False)):
                exit_ |= close < ma20
            scoring = {
                "momentum_60d": 0.40,
                "industry_momentum_20d": 0.25,
                "industry_breadth_5d": 0.20,
                "amihud_20d": 0.15,
            }
            directions = {"amihud_20d": "low"}
        elif family in {
            "sentiment_secondary_ignition",
            "secondary_ignition",
            "accumulation_secondary_ignition",
            "accumulation_strong_market",
        }:
            breadth, above_ma20 = _market_state(market)
            prior_above = np.empty_like(above_ma20)
            prior_above[0] = np.nan
            prior_above[1:] = above_ma20[:-1]
            regime_cross = (above_ma20 >= np.float32(0.50)) & (
                prior_above < np.float32(0.50)
            )
            min_above_ma20 = np.float32(
                params.get("secondary_min_above_ma20", 0.50)
            )
            market_on = (above_ma20 >= min_above_ma20) & (
                breadth >= np.float32(0.40)
            )
            if family == "accumulation_strong_market":
                market_on = (above_ma20 >= np.float32(0.70)) & (
                    breadth >= np.float32(0.40)
                )
            limit_count = matrix_feature(market, "limit_up_count_20d")
            ret_skew = matrix_feature(market, "ret_skew_20d")
            annual_vol = matrix_feature(market, "annual_vol_20d")
            ma20_bias = matrix_feature(market, "ma20_bias")
            close_position = matrix_feature(market, "close_position")
            turnover = matrix_feature(market, "turnover_rate")
            secondary_pool = (
                (close > ma20)
                & (ma20 > ma60)
                & (momentum_60d >= np.float32(0.10))
                & (momentum_60d <= np.float32(0.50))
                & (ma20_bias <= np.float32(0.15))
                & (rsi >= np.float32(45.0))
                & (rsi <= np.float32(70.0))
                & (limit_count >= np.float32(1.0))
                & (limit_count <= np.float32(3.0))
                & (ret_skew > 0)
                & (annual_vol <= np.float32(0.90))
                & (turnover <= np.float32(10.0))
                & (vol_ratio >= np.float32(0.50))
                & (vol_ratio <= np.float32(1.50))
                & (close_position <= np.float32(0.80))
            )
            if family == "sentiment_secondary_ignition":
                entry = regime_cross[:, None] & secondary_pool
            else:
                entry = (
                    market_on[:, None]
                    & secondary_pool
                    & (prior_close < prior_ma10)
                    & (close >= ma10)
                    & bullish
                )
                if family in {
                    "accumulation_secondary_ignition",
                    "accumulation_strong_market",
                }:
                    turnover_z = matrix_feature(market, "turnover_z_60d")
                    prior_turnover_z = shift(turnover_z, 1)
                    prior_max_turnover_z = valid_rolling_max(
                        prior_turnover_z,
                        np.isfinite(prior_turnover_z),
                        5,
                    )
                    vol_price_corr = matrix_feature(market, "vol_price_corr_20d")
                    vwap_bias = matrix_feature(market, "vwap_bias")
                    boll_position = matrix_feature(market, "boll_position")
                    prior_industry_breadth = shift(industry_breadth, 5)
                    breadth_acceleration = industry_breadth - prior_industry_breadth
                    entry &= (
                        (prior_max_turnover_z >= np.float32(1.50))
                        & (turnover_z < prior_max_turnover_z)
                        & (vol_price_corr > 0)
                        & (vwap_bias >= np.float32(-0.03))
                        & (vwap_bias <= np.float32(0.02))
                        & (boll_position <= np.float32(0.85))
                        & (breadth_acceleration > 0)
                    )
            exit_ = (
                np.broadcast_to(
                    ((above_ma20 < np.float32(0.40)) | (breadth < np.float32(0.30)))[
                        :, None
                    ],
                    market.shape,
                ).copy()
                | (close < ma20)
            )
            scoring = {
                "ret_skew_20d": 0.20,
                "limit_up_count_20d": 0.20,
                "ma20_bias": 0.20,
                "close_position": 0.15,
                "industry_breadth_5d": 0.15,
                "momentum_60d": 0.10,
            }
            directions = {
                "ma20_bias": "low",
                "close_position": "low",
                "momentum_60d": "low",
            }
        elif family == "breadth_oversold_repair":
            _, above_ma20 = _market_state(market)
            prior_above = np.empty_like(above_ma20)
            prior_above[0] = np.nan
            prior_above[1:] = above_ma20[:-1]
            regime_cross = (above_ma20 >= np.float32(0.50)) & (
                prior_above < np.float32(0.50)
            )
            ma20_bias = matrix_feature(market, "ma20_bias")
            close_position = matrix_feature(market, "close_position")
            entry = (
                regime_cross[:, None]
                & (rsi <= np.float32(40.0))
                & (momentum_20d <= np.float32(-0.05))
                & (ma20_bias <= np.float32(-0.03))
                & (change > 0)
                & bullish
                & (close_position >= np.float32(0.60))
                & (vol_ratio >= np.float32(0.80))
                & (vol_ratio <= np.float32(2.00))
            )
            exit_ = (close >= ma20) | (rsi >= np.float32(55.0))
            scoring = {
                "rsi_14": 0.30,
                "momentum_20d": 0.25,
                "close_position": 0.25,
                "amihud_20d": 0.20,
            }
            directions = {
                "rsi_14": "low",
                "momentum_20d": "low",
                "amihud_20d": "low",
            }
        elif family == "quality_compounder":
            roe = matrix_feature(market, "roe_latest")
            margin = matrix_feature(market, "net_margin_latest")
            revenue_yoy = matrix_feature(market, "revenue_yoy_latest")
            debt_ratio = matrix_feature(market, "debt_ratio_latest")
            quality = (
                np.isfinite(roe)
                & np.isfinite(margin)
                & np.isfinite(revenue_yoy)
                & np.isfinite(debt_ratio)
                & (roe >= np.float32(8.0))
                & (margin >= np.float32(5.0))
                & (revenue_yoy >= 0)
                & (debt_ratio <= np.float32(65.0))
            )
            entry = (
                quality
                & (prior_close < prior_ma20)
                & (close >= ma20)
                & (ma20 > ma60)
                & (momentum_60d > 0)
            )
            exit_ = (close < ma20) | ~quality
            scoring = {
                "roe_latest": 0.25,
                "revenue_yoy_latest": 0.20,
                "net_margin_latest": 0.15,
                "debt_ratio_latest": 0.15,
                "momentum_60d": 0.15,
                "amihud_20d": 0.10,
            }
            directions = {"debt_ratio_latest": "low", "amihud_20d": "low"}
        elif family == "limit_event":
            limit_count = matrix_feature(market, "limit_up_count_20d")
            entry = (
                (limit_count >= np.float32(1.0))
                & (prior_close < prior_ma10)
                & (close >= ma10)
                & (close >= ma20)
                & (momentum_20d >= np.float32(-0.05))
                & (vol_ratio >= np.float32(0.5))
                & (vol_ratio <= np.float32(1.5))
                & (change < np.float32(0.05))
                & bullish
            )
            exit_ = (close < ma20) | (rsi >= np.float32(70.0))
            scoring = {
                "limit_up_count_20d": 0.30,
                "industry_momentum_20d": 0.20,
                "close_position": 0.20,
                "momentum_20d": 0.15,
                "amihud_20d": 0.15,
            }
            directions = {"amihud_20d": "low"}
        elif family == "monthly_momentum":
            momentum_12_1 = shift(close, 21) / shift(close, 252) - np.float32(1.0)
            labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
            rebalance = np.zeros(len(labels), dtype=bool)
            for index, value in enumerate(labels):
                rebalance[index] = index == 0 or (
                    value.year,
                    value.month,
                ) != (
                    labels[index - 1].year,
                    labels[index - 1].month,
                )
            entry = (
                rebalance[:, None]
                & np.isfinite(momentum_12_1)
                & (momentum_12_1 > 0)
                & (close > ma60)
            )
            if bool(params.get("quality_gate", False)):
                roe = matrix_feature(market, "roe_latest")
                margin = matrix_feature(market, "net_margin_latest")
                revenue_yoy = matrix_feature(market, "revenue_yoy_latest")
                debt_ratio = matrix_feature(market, "debt_ratio_latest")
                entry &= (
                    np.isfinite(roe)
                    & np.isfinite(margin)
                    & np.isfinite(revenue_yoy)
                    & np.isfinite(debt_ratio)
                    & (roe >= np.float32(8.0))
                    & (margin >= np.float32(5.0))
                    & (revenue_yoy >= 0)
                    & (debt_ratio <= np.float32(65.0))
                )
            exit_ = np.zeros(market.shape, dtype=bool)
            score_override = np.nan_to_num(
                momentum_12_1,
                nan=-np.inf,
                posinf=-np.inf,
                neginf=-np.inf,
            ).astype(np.float32)
            scoring = {}
            directions = {}
        elif family == "regime_reversal_quality":
            _, above_ma20 = _market_state(market)
            stress = above_ma20 < np.float32(0.35)
            prior_stress = np.empty_like(stress)
            prior_stress[0] = stress[0]
            prior_stress[1:] = stress[:-1]

            low_60d = matrix_feature(market, "low_60d")
            reversal_entry = (
                (~stress[:, None])
                & (close <= low_60d)
                & bullish
                & (vol_ratio >= np.float32(1.5))
            )
            recovery_score = build_matrix_score(
                market,
                reversal_entry,
                {
                    "close_position": 0.35,
                    "intraday_return": 0.25,
                    "rsi_14": 0.25,
                    "amihud_20d": 0.15,
                },
                "score",
                True,
                fallback=np.zeros(market.shape, dtype=np.float32),
                directions={"rsi_14": "low", "amihud_20d": "low"},
            )

            momentum_12_1 = shift(close, 21) / shift(close, 252) - np.float32(1.0)
            labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
            rebalance = np.zeros(len(labels), dtype=bool)
            for index, value in enumerate(labels):
                rebalance[index] = index == 0 or (
                    value.year,
                    value.month,
                ) != (
                    labels[index - 1].year,
                    labels[index - 1].month,
                )
            roe = matrix_feature(market, "roe_latest")
            margin = matrix_feature(market, "net_margin_latest")
            revenue_yoy = matrix_feature(market, "revenue_yoy_latest")
            debt_ratio = matrix_feature(market, "debt_ratio_latest")
            quality = (
                np.isfinite(roe)
                & np.isfinite(margin)
                & np.isfinite(revenue_yoy)
                & np.isfinite(debt_ratio)
                & (roe >= np.float32(8.0))
                & (margin >= np.float32(5.0))
                & (revenue_yoy >= 0)
                & (debt_ratio <= np.float32(65.0))
            )
            quality_entry = (
                stress[:, None]
                & rebalance[:, None]
                & quality
                & np.isfinite(momentum_12_1)
                & (momentum_12_1 > 0)
                & (close > ma60)
            )
            entry = reversal_entry | quality_entry
            state_changed = stress != prior_stress
            reversal_exit = (~stress[:, None]) & (close < ma20) & (
                prior_close >= prior_ma20
            )
            exit_ = reversal_exit | np.broadcast_to(
                state_changed[:, None], market.shape
            )
            momentum_score = np.nan_to_num(
                momentum_12_1,
                nan=-np.inf,
                posinf=-np.inf,
                neginf=-np.inf,
            ).astype(np.float32)
            score_override = np.where(
                stress[:, None], momentum_score, recovery_score
            ).astype(np.float32)
            scoring = {}
            directions = {}
        else:
            raise ValueError(f"unsupported family: {family}")

        if bool(params.get("risk_on_only", False)):
            _, above_ma20 = _market_state(market)
            risk_on = above_ma20 >= np.float32(0.55)
            entry &= risk_on[:, None]
            exit_ |= np.broadcast_to(
                (above_ma20 < np.float32(0.50))[:, None], market.shape
            )

        entry &= eligible & build_basic_filter_mask(market, META["basic_filter"])
        score = (
            score_override
            if score_override is not None
            else build_matrix_score(
                market,
                entry,
                scoring,
                "score",
                True,
                fallback=np.zeros(market.shape, dtype=np.float32),
                directions=directions,
            )
        )
        score[~entry] = np.float32(0.0)
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_independent_alpha_entry",),
            exit_signal_ids=("signal_independent_alpha_exit",),
        )


MATRIX_STRATEGY = IndependentAlphaFamiliesMatrixStrategy()
