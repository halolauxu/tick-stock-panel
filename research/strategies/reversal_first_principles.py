"""First-principles A-share reversal hypotheses for controlled research.

The strategy is deliberately research-only.  Its parameter sets are frozen by
the runner before any out-of-sample result is revealed.
"""
from __future__ import annotations

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    build_basic_filter_mask,
    build_matrix_score,
    make_signal_matrix,
    matrix_feature,
)
from app.backtest.matrix import (
    valid_shift as shift,
)

META = {
    "id": "reversal_first_principles",
    "name": "反转第一性原理研究",
    "description": "衰竭、下破收回、恐慌修复、强势股回踩与市场宽度的受控研究",
    "tags": ["研究", "反转", "市场宽度", "财务质量"],
    "research_only": True,
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "price_min": 3,
        "price_max": 300,
        "market_cap_min": 10e8,
        "amount_min": 0.2e8,
        "exclude_st": True,
        "exclude_new_days": 30,
        "boards": ["沪主板", "深主板"],
    },
    "params": [
        {"id": "family", "label": "事件族", "type": "str", "default": "new_low"},
        {"id": "rsi_max", "label": "RSI上限", "type": "float", "default": 100.0},
        {"id": "vol_min", "label": "最低量比", "type": "float", "default": 0.0},
        {"id": "vol_max", "label": "最高量比", "type": "float", "default": 99.0},
        {"id": "close_position_min", "label": "最低收盘位置", "type": "float", "default": 0.0},
        {"id": "market_mode", "label": "市场状态", "type": "str", "default": "none"},
        {"id": "industry_mode", "label": "行业状态", "type": "str", "default": "none"},
        {"id": "overlay_context", "label": "叠加事件环境", "type": "str", "default": "none"},
        {"id": "score_mode", "label": "信号优先级", "type": "str", "default": "recovery"},
        {"id": "quality_mode", "label": "财务质量", "type": "str", "default": "none"},
        {"id": "exit_mode", "label": "退出模式", "type": "str", "default": "ma20_cross"},
        {"id": "eligibility_mode", "label": "历史股票池", "type": "str", "default": "none"},
        {"id": "event_cooldown_days", "label": "事件冷却期", "type": "int", "default": 0},
    ],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_reversal_research_entry"]
EXIT_SIGNALS = ["signal_reversal_research_exit"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15

_FINANCIAL_FIELDS = frozenset(
    {
        "roe_latest",
        "net_margin_latest",
        "revenue_yoy_latest",
        "debt_ratio_latest",
    }
)


class ReversalFirstPrinciplesMatrixStrategy:
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
            }
        ) | _FINANCIAL_FIELDS

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 60

    def required_fields_for_params(self, params: dict) -> frozenset[str]:
        required = set()
        if str(params.get("eligibility_mode", "none")) == "pit":
            required.add("pit_eligible")
        industry_mode = str(params.get("industry_mode", "none"))
        overlay_context = str(params.get("overlay_context", "none"))
        score_mode = str(params.get("score_mode", "recovery"))
        if (
            industry_mode == "none"
            and "industry" not in overlay_context
            and score_mode
            not in {
                "industry_recovery",
                "hybrid",
                "continuous_stress_industry",
                "trend_breakout_quality",
                "trend_pullback_industry",
            }
        ):
            return frozenset(required)
        required.update({"industry_momentum_20d", "industry_breadth_5d"})
        return frozenset(required)

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        family = str(params.get("family", "new_low"))
        rsi = matrix_feature(market, "rsi_14")
        vol_ratio = matrix_feature(market, "vol_ratio_5d")
        close_position = matrix_feature(market, "close_position")
        change = matrix_feature(market, "change_pct")
        gap = matrix_feature(market, "gap_return")
        momentum_5d = matrix_feature(market, "momentum_5d")
        low_60d = matrix_feature(market, "low_60d")
        ma5 = matrix_feature(market, "ma5")
        ma10 = matrix_feature(market, "ma10")
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")

        valid = np.isfinite(market.close) & np.isfinite(rsi) & np.isfinite(vol_ratio)
        bullish = market.close > market.open

        priority_bonus = np.zeros(market.shape, dtype=np.float32)
        score_override = None
        apply_common_filters = True
        if family == "new_low":
            event = (market.close <= low_60d) & bullish
        elif family == "spring":
            prior_low = shift(low_60d, 1)
            event = (
                (market.low < prior_low * np.float32(0.995))
                & (market.close >= prior_low)
                & bullish
            )
        elif family == "capitulation_repair":
            distance_from_low = matrix_feature(market, "distance_from_low_60d")
            event = (
                (distance_from_low <= np.float32(0.02))
                & (gap <= np.float32(-0.005))
                & bullish
            )
        elif family == "limit_pullback":
            limit_gene = matrix_feature(market, "limit_up_count_20d")
            event = (
                (limit_gene >= np.float32(1.0))
                & (momentum_5d >= np.float32(-0.12))
                & (momentum_5d <= np.float32(0.02))
                & (market.low <= ma20 * np.float32(1.02))
                & (market.close >= ma20)
                & bullish
            )
        elif family == "breakout_retest":
            prior_distance = shift(matrix_feature(market, "distance_to_high_60d"), 1)
            event = (
                (prior_distance >= np.float32(-0.01))
                & (market.low <= ma5 * np.float32(1.005))
                & (market.close >= ma5)
                & bullish
                & (change < np.float32(0.05))
            )
        elif family in {"adaptive_union_cap", "adaptive_switch_cap"}:
            baseline_event = (
                (market.close <= low_60d)
                & bullish
                & (vol_ratio >= np.float32(1.5))
            )
            cap_event = (
                (matrix_feature(market, "distance_from_low_60d") <= np.float32(0.02))
                & (gap <= np.float32(-0.005))
                & bullish
                & (rsi <= np.float32(35.0))
                & (vol_ratio >= np.float32(1.0))
                & (vol_ratio <= np.float32(4.0))
                & (close_position >= np.float32(0.65))
            )
            context_gate = self._overlay_context_gate(market, params)
            cap_event &= context_gate
            if family == "adaptive_switch_cap":
                repair_day = self._market_gate(market, {"market_mode": "breadth_repair"})
                event = (repair_day & cap_event) | (~repair_day & baseline_event)
            else:
                event = baseline_event | cap_event
            priority_bonus[cap_event] = np.float32(25.0)
            apply_common_filters = False
        elif family == "adaptive_union_spring":
            baseline_event = (
                (market.close <= low_60d)
                & bullish
                & (vol_ratio >= np.float32(1.5))
            )
            prior_low = shift(low_60d, 1)
            spring_event = (
                (market.low < prior_low * np.float32(0.995))
                & (market.close >= prior_low)
                & bullish
                & (rsi <= np.float32(45.0))
                & (vol_ratio >= np.float32(0.8))
                & (vol_ratio <= np.float32(3.0))
                & (close_position >= np.float32(0.65))
                & self._overlay_context_gate(market, params)
            )
            event = baseline_event | spring_event
            priority_bonus[spring_event] = np.float32(20.0)
            apply_common_filters = False
        elif family == "baseline_ranked":
            event = (
                (market.close <= low_60d)
                & bullish
                & (vol_ratio >= np.float32(1.5))
            )
            apply_common_filters = False
        elif family == "sparse_breakout":
            high_60d = matrix_feature(market, "high_60d")
            prior_high = shift(high_60d, 1)
            prior_close = shift(market.close, 1)
            event = (
                (market.close >= high_60d)
                & (prior_close < prior_high)
                & (ma20 > ma60)
                & (market.close > ma60)
                & (vol_ratio >= np.float32(1.2))
                & (close_position >= np.float32(0.70))
                & (change > np.float32(0.0))
                & (change <= np.float32(0.07))
            )
            apply_common_filters = False
        elif family == "first_trend_pullback":
            momentum_60d = matrix_feature(market, "momentum_60d")
            prior_close = shift(market.close, 1)
            prior_ma10 = shift(ma10, 1)
            event = (
                (ma20 > ma60)
                & (momentum_60d >= np.float32(0.10))
                & (prior_close < prior_ma10)
                & (market.close >= ma10)
                & (market.low <= ma20 * np.float32(1.03))
                & (market.close >= ma20 * np.float32(0.98))
                & bullish
                & (vol_ratio >= np.float32(0.6))
                & (vol_ratio <= np.float32(1.5))
                & (change <= np.float32(0.05))
            )
            apply_common_filters = False
        elif family == "breakout_first_retest":
            high_60d = matrix_feature(market, "high_60d")
            prior_close = shift(market.close, 1)
            prior_high = shift(high_60d, 1)
            breakout = (market.close >= high_60d) & (prior_close < prior_high)
            recent_breakout = np.zeros(market.shape, dtype=bool)
            for lag in range(1, 21):
                recent_breakout[lag:] |= breakout[:-lag]
            prior_ma10 = shift(ma10, 1)
            event = (
                recent_breakout
                & (ma20 > ma60)
                & (prior_close < prior_ma10)
                & (market.close >= ma10)
                & (market.close >= ma20)
                & (market.close >= high_60d * np.float32(0.92))
                & bullish
                & (vol_ratio >= np.float32(0.6))
                & (vol_ratio <= np.float32(1.5))
                & (change <= np.float32(0.05))
            )
            apply_common_filters = False
        else:
            raise ValueError(f"unsupported family: {family}")

        cooldown_days = int(params.get("event_cooldown_days", 0))
        if cooldown_days < 0:
            raise ValueError("event_cooldown_days must be non-negative")
        if cooldown_days:
            recent_event = np.zeros(market.shape, dtype=bool)
            for lag in range(1, cooldown_days + 1):
                recent_event[lag:] |= event[:-lag]
            event &= ~recent_event

        entry = valid & event
        if apply_common_filters:
            entry &= (
                (rsi <= float(params.get("rsi_max", 100.0)))
                & (vol_ratio >= float(params.get("vol_min", 0.0)))
                & (vol_ratio <= float(params.get("vol_max", 99.0)))
                & (close_position >= float(params.get("close_position_min", 0.0)))
            )
        entry &= self._market_gate(market, params)
        entry &= self._industry_gate(market, params)
        entry &= self._quality_gate(market, params)
        eligibility_mode = str(params.get("eligibility_mode", "none"))
        if eligibility_mode == "pit":
            entry &= matrix_feature(market, "pit_eligible") > np.float32(0.5)
        elif eligibility_mode != "none":
            raise ValueError(f"unsupported eligibility_mode: {eligibility_mode}")

        if family in {
            "baseline_ranked",
            "sparse_breakout",
            "first_trend_pullback",
            "breakout_first_retest",
        }:
            score_override = self._ranking_score(market, entry, params)

        exit_mode = str(params.get("exit_mode", "ma20_cross"))
        if exit_mode == "ma20_cross":
            exit_ = (market.close < ma20) & (shift(market.close, 1) >= shift(ma20, 1))
        elif exit_mode == "ma10_recovery":
            exit_ = (market.close >= ma10) & (shift(market.close, 1) < shift(ma10, 1))
        elif exit_mode == "rsi_recovery":
            exit_ = (rsi >= np.float32(55.0)) & (shift(rsi, 1) < np.float32(55.0))
        elif exit_mode == "time_only":
            exit_ = np.zeros(market.shape, dtype=bool)
        else:
            raise ValueError(f"unsupported exit_mode: {exit_mode}")

        # Rank simultaneous signals by genuine intraday recovery, not by the
        # most negative prior trend.  Market gates are binary and add no score.
        score = (
            score_override
            if score_override is not None
            else (
                np.nan_to_num(close_position, nan=0.0) * np.float32(45.0)
                + np.clip(np.nan_to_num(change, nan=0.0), -0.10, 0.10)
                * np.float32(200.0)
                + np.clip(np.nan_to_num(vol_ratio, nan=0.0), 0.0, 4.0)
                * np.float32(8.0)
                + np.clip(
                    (50.0 - np.nan_to_num(rsi, nan=50.0)) / 50.0,
                    0.0,
                    1.0,
                )
                * np.float32(15.0)
                + priority_bonus
            ).astype(np.float32)
        )
        score[~entry] = np.float32(0.0)
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_reversal_research_entry",),
            exit_signal_ids=("signal_reversal_research_exit",),
        )

    @staticmethod
    def _market_gate(market: MarketDataMatrix, params: dict) -> np.ndarray:
        mode = str(params.get("market_mode", "none"))
        if mode == "none":
            return np.ones(market.shape, dtype=bool)

        breadth, above_ma20, prior_breadth = (
            ReversalFirstPrinciplesMatrixStrategy._market_state(market)
        )
        if mode == "not_crash":
            row_gate = breadth >= np.float32(0.30)
        elif mode == "breadth_repair":
            row_gate = (breadth >= np.float32(0.45)) & (
                breadth - prior_breadth >= np.float32(0.08)
            )
        elif mode == "trend_support":
            row_gate = above_ma20 >= np.float32(0.45)
        else:
            raise ValueError(f"unsupported market_mode: {mode}")
        return np.broadcast_to(row_gate[:, None], market.shape)

    @staticmethod
    def _market_state(market: MarketDataMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        change = matrix_feature(market, "change_pct")
        ma20 = matrix_feature(market, "ma20")
        symbols = np.asarray(market.symbols)
        main_board = np.array(
            [
                (symbol.endswith(".SH") and symbol.startswith("60"))
                or (symbol.endswith(".SZ") and symbol.startswith(("000", "001", "002", "003")))
                for symbol in symbols
            ],
            dtype=bool,
        )
        valid_change = np.isfinite(change[:, main_board])
        count = valid_change.sum(axis=1)
        positive = ((change[:, main_board] > 0) & valid_change).sum(axis=1)
        breadth = np.divide(positive, count, out=np.zeros_like(count, dtype=np.float32), where=count > 0)
        valid_ma = np.isfinite(ma20[:, main_board]) & np.isfinite(market.close[:, main_board])
        ma_count = valid_ma.sum(axis=1)
        above = ((market.close[:, main_board] > ma20[:, main_board]) & valid_ma).sum(axis=1)
        above_ma20 = np.divide(
            above,
            ma_count,
            out=np.zeros_like(ma_count, dtype=np.float32),
            where=ma_count > 0,
        )
        prior_breadth = np.empty_like(breadth)
        prior_breadth[0] = np.nan
        prior_breadth[1:] = breadth[:-1]
        return breadth, above_ma20, prior_breadth

    @staticmethod
    def _ranking_score(
        market: MarketDataMatrix,
        entry: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        mode = str(params.get("score_mode", "recovery"))
        definitions = {
            "baseline": (
                {"change_pct": 0.4, "vol_ratio_5d": 0.3, "momentum_5d": 0.3},
                {},
            ),
            "recovery": (
                {
                    "close_position": 0.35,
                    "intraday_return": 0.25,
                    "rsi_14": 0.25,
                    "amihud_20d": 0.15,
                },
                {"rsi_14": "low", "amihud_20d": "low"},
            ),
            "lottery_aware": (
                {
                    "close_position": 0.25,
                    "rsi_14": 0.20,
                    "max_ret_20d": 0.20,
                    "ret_skew_20d": 0.15,
                    "amihud_20d": 0.20,
                },
                {
                    "rsi_14": "low",
                    "max_ret_20d": "low",
                    "ret_skew_20d": "low",
                    "amihud_20d": "low",
                },
            ),
            "quality_recovery": (
                {
                    "close_position": 0.25,
                    "rsi_14": 0.15,
                    "roe_latest": 0.20,
                    "net_margin_latest": 0.15,
                    "revenue_yoy_latest": 0.15,
                    "debt_ratio_latest": 0.10,
                },
                {"rsi_14": "low", "debt_ratio_latest": "low"},
            ),
            "industry_recovery": (
                {
                    "close_position": 0.25,
                    "rsi_14": 0.20,
                    "industry_momentum_20d": 0.25,
                    "industry_breadth_5d": 0.15,
                    "amihud_20d": 0.15,
                },
                {"rsi_14": "low", "amihud_20d": "low"},
            ),
            "hybrid": (
                {
                    "close_position": 0.20,
                    "rsi_14": 0.15,
                    "max_ret_20d": 0.15,
                    "roe_latest": 0.15,
                    "debt_ratio_latest": 0.10,
                    "industry_momentum_20d": 0.15,
                    "industry_breadth_5d": 0.10,
                },
                {
                    "rsi_14": "low",
                    "max_ret_20d": "low",
                    "debt_ratio_latest": "low",
                },
            ),
            "trend_breakout": (
                {
                    "momentum_60d": 0.35,
                    "vol_ratio_5d": 0.20,
                    "close_position": 0.20,
                    "atr_pct": 0.15,
                    "amihud_20d": 0.10,
                },
                {"atr_pct": "low", "amihud_20d": "low"},
            ),
            "trend_breakout_quality": (
                {
                    "momentum_60d": 0.25,
                    "industry_momentum_20d": 0.20,
                    "roe_latest": 0.15,
                    "revenue_yoy_latest": 0.15,
                    "close_position": 0.15,
                    "atr_pct": 0.10,
                },
                {"atr_pct": "low"},
            ),
            "trend_pullback": (
                {
                    "momentum_60d": 0.35,
                    "rsi_14": 0.20,
                    "close_position": 0.15,
                    "atr_pct": 0.15,
                    "amihud_20d": 0.15,
                },
                {"rsi_14": "low", "atr_pct": "low", "amihud_20d": "low"},
            ),
            "trend_pullback_industry": (
                {
                    "momentum_60d": 0.25,
                    "industry_momentum_20d": 0.25,
                    "industry_breadth_5d": 0.15,
                    "rsi_14": 0.15,
                    "amihud_20d": 0.20,
                },
                {"rsi_14": "low", "amihud_20d": "low"},
            ),
        }
        adaptive_modes = {
            "adaptive_weak_breadth",
            "adaptive_trend_stress",
            "adaptive_breadth_repair",
        }
        continuous_stress_modes = {
            "continuous_stress_recovery": "recovery",
            "continuous_stress_lottery": "lottery_aware",
            "continuous_stress_industry": "industry_recovery",
            "continuous_stress_quality": "quality_recovery",
        }
        if mode in continuous_stress_modes:
            valid_entry = entry & build_basic_filter_mask(market, META["basic_filter"])
            baseline_scoring, baseline_directions = definitions["baseline"]
            defensive_mode = continuous_stress_modes[mode]
            defensive_scoring, defensive_directions = definitions[defensive_mode]
            baseline_score = build_matrix_score(
                market,
                valid_entry,
                baseline_scoring,
                "score",
                True,
                fallback=np.zeros(market.shape, dtype=np.float32),
                directions=baseline_directions,
            )
            defensive_score = build_matrix_score(
                market,
                valid_entry,
                defensive_scoring,
                "score",
                True,
                fallback=np.zeros(market.shape, dtype=np.float32),
                directions=defensive_directions,
            )
            _, above_ma20, _ = ReversalFirstPrinciplesMatrixStrategy._market_state(
                market
            )
            # A causal, parameter-free soft allocation: the fraction of the
            # main board below MA20 is the weight assigned to the defensive
            # ranking.  This avoids retuning Stage-8's failed 45% switch.
            stress = np.clip(
                np.float32(1.0) - above_ma20,
                np.float32(0.0),
                np.float32(1.0),
            )
            blended = (
                baseline_score * (np.float32(1.0) - stress[:, None])
                + defensive_score * stress[:, None]
            )
            return blended.astype(np.float32)
        if mode in adaptive_modes:
            valid_entry = entry & build_basic_filter_mask(market, META["basic_filter"])
            baseline_scoring, baseline_directions = definitions["baseline"]
            recovery_scoring, recovery_directions = definitions["recovery"]
            baseline_score = build_matrix_score(
                market,
                valid_entry,
                baseline_scoring,
                "score",
                True,
                fallback=np.zeros(market.shape, dtype=np.float32),
                directions=baseline_directions,
            )
            recovery_score = build_matrix_score(
                market,
                valid_entry,
                recovery_scoring,
                "score",
                True,
                fallback=np.zeros(market.shape, dtype=np.float32),
                directions=recovery_directions,
            )
            breadth, above_ma20, prior_breadth = (
                ReversalFirstPrinciplesMatrixStrategy._market_state(market)
            )
            if mode == "adaptive_weak_breadth":
                use_recovery = breadth < np.float32(0.50)
            elif mode == "adaptive_trend_stress":
                use_recovery = above_ma20 < np.float32(0.45)
            else:
                use_recovery = (breadth >= np.float32(0.45)) & (
                    breadth - prior_breadth >= np.float32(0.08)
                )
            return np.where(use_recovery[:, None], recovery_score, baseline_score)
        try:
            scoring, directions = definitions[mode]
        except KeyError as exc:
            raise ValueError(f"unsupported score_mode: {mode}") from exc
        return build_matrix_score(
            market,
            entry & build_basic_filter_mask(market, META["basic_filter"]),
            scoring,
            "score",
            True,
            fallback=np.zeros(market.shape, dtype=np.float32),
            directions=directions,
        )

    @staticmethod
    def _overlay_context_gate(market: MarketDataMatrix, params: dict) -> np.ndarray:
        mode = str(params.get("overlay_context", "none"))
        if mode == "none":
            return np.ones(market.shape, dtype=bool)
        gate = np.ones(market.shape, dtype=bool)
        if "breadth" in mode:
            gate &= ReversalFirstPrinciplesMatrixStrategy._market_gate(
                market, {"market_mode": "breadth_repair"}
            )
        if "industry" in mode:
            gate &= ReversalFirstPrinciplesMatrixStrategy._industry_gate(
                market, {"industry_mode": "stable"}
            )
        if mode not in {"breadth", "industry", "breadth_industry"}:
            raise ValueError(f"unsupported overlay_context: {mode}")
        return gate

    @staticmethod
    def _industry_gate(market: MarketDataMatrix, params: dict) -> np.ndarray:
        mode = str(params.get("industry_mode", "none"))
        if mode == "none":
            return np.ones(market.shape, dtype=bool)
        momentum = matrix_feature(market, "industry_momentum_20d")
        breadth = matrix_feature(market, "industry_breadth_5d")
        valid = np.isfinite(momentum) & np.isfinite(breadth)
        if mode == "stable":
            return valid & (momentum >= -0.05) & (breadth >= 0.45)
        if mode == "leader":
            return valid & (momentum >= 0.0) & (breadth >= 0.50)
        raise ValueError(f"unsupported industry_mode: {mode}")

    @staticmethod
    def _quality_gate(market: MarketDataMatrix, params: dict) -> np.ndarray:
        mode = str(params.get("quality_mode", "none"))
        if mode == "none":
            return np.ones(market.shape, dtype=bool)
        roe = matrix_feature(market, "roe_latest")
        net_margin = matrix_feature(market, "net_margin_latest")
        revenue_yoy = matrix_feature(market, "revenue_yoy_latest")
        debt_ratio = matrix_feature(market, "debt_ratio_latest")
        finite = (
            np.isfinite(roe)
            & np.isfinite(net_margin)
            & np.isfinite(revenue_yoy)
            & np.isfinite(debt_ratio)
        )
        if mode == "safety":
            return finite & (roe > 0) & (net_margin > 0) & (debt_ratio <= 75)
        if mode == "quality":
            return (
                finite
                & (roe >= 8)
                & (net_margin >= 5)
                & (revenue_yoy >= 0)
                & (debt_ratio <= 65)
            )
        raise ValueError(f"unsupported quality_mode: {mode}")


MATRIX_STRATEGY = ReversalFirstPrinciplesMatrixStrategy()
