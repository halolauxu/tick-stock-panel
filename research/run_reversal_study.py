"""Frozen discovery/OOS study for first-principles A-share reversal ideas."""
from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from app.backtest.engine import BacktestEngine
from app.backtest.strategy import (
    BacktestResultPolicy,
    StrategyBacktestConfig,
    StrategyBacktestService,
)
from app.strategy import config as strategy_config
from app.strategy.engine import StrategyEngine
from app.tickflow.repository import DataStore, KlineRepository
from app.share_capital import load_share_history

POLICY = BacktestResultPolicy(
    required_stats=frozenset(
        {"total_return", "sharpe", "max_drawdown", "n_trades", "win_rate", "profit_factor"}
    ),
    include_monte_carlo=False,
    include_curves=False,
    include_trades=False,
    include_per_symbol_stats=False,
    include_return_distribution=False,
    include_benchmark=False,
    include_strategy_info=False,
)

COMMON = {
    "rsi_max": 100.0,
    "vol_min": 0.0,
    "vol_max": 99.0,
    "close_position_min": 0.0,
    "market_mode": "none",
    "industry_mode": "none",
    "overlay_context": "none",
    "quality_mode": "none",
    "exit_mode": "ma20_cross",
    "eligibility_mode": "none",
    "score_mode": "recovery",
}

RESEARCH_BASIC_FILTER = {
    "enabled": True,
    "price_min": 3,
    "price_max": 300,
    "market_cap_min": 10e8,
    "amount_min": 0.2e8,
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板"],
}

# Frozen before any OOS run.  These are economic hypotheses, not a Cartesian grid.
STAGE1_CANDIDATES = {
    "new_low_exhaustion": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55},
    "new_low_deep_oversold": {"family": "new_low", "rsi_max": 30, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55},
    "new_low_strong_reclaim": {"family": "new_low", "rsi_max": 40, "vol_min": 1.2, "vol_max": 3.5, "close_position_min": 0.70},
    "new_low_not_crash": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55, "market_mode": "not_crash"},
    "new_low_breadth_repair": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55, "market_mode": "breadth_repair"},
    "new_low_trend_support": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55, "market_mode": "trend_support"},
    "new_low_industry_stable": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55, "industry_mode": "stable"},
    "new_low_financial_safety": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55, "quality_mode": "safety"},
    "new_low_quality_repair": {"family": "new_low", "rsi_max": 45, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.55, "market_mode": "not_crash", "quality_mode": "quality"},
    "new_low_fast_mean_revert": {"family": "new_low", "rsi_max": 35, "vol_min": 1.2, "vol_max": 3.0, "close_position_min": 0.60, "exit_mode": "ma10_recovery"},
    "spring_reclaim": {"family": "spring", "rsi_max": 45, "vol_min": 0.8, "vol_max": 3.0, "close_position_min": 0.65},
    "spring_deep": {"family": "spring", "rsi_max": 30, "vol_min": 0.8, "vol_max": 3.0, "close_position_min": 0.65},
    "spring_not_crash": {"family": "spring", "rsi_max": 45, "vol_min": 0.8, "vol_max": 3.0, "close_position_min": 0.65, "market_mode": "not_crash"},
    "spring_industry_stable": {"family": "spring", "rsi_max": 45, "vol_min": 0.8, "vol_max": 3.0, "close_position_min": 0.65, "industry_mode": "stable"},
    "spring_financial_safety": {"family": "spring", "rsi_max": 45, "vol_min": 0.8, "vol_max": 3.0, "close_position_min": 0.65, "quality_mode": "safety"},
    "spring_rsi_exit": {"family": "spring", "rsi_max": 40, "vol_min": 0.8, "vol_max": 3.0, "close_position_min": 0.65, "exit_mode": "rsi_recovery"},
    "capitulation_repair": {"family": "capitulation_repair", "rsi_max": 35, "vol_min": 1.0, "vol_max": 4.0, "close_position_min": 0.65},
    "capitulation_not_crash": {"family": "capitulation_repair", "rsi_max": 35, "vol_min": 1.0, "vol_max": 4.0, "close_position_min": 0.65, "market_mode": "not_crash"},
    "capitulation_breadth": {"family": "capitulation_repair", "rsi_max": 35, "vol_min": 1.0, "vol_max": 4.0, "close_position_min": 0.65, "market_mode": "breadth_repair"},
    "capitulation_industry": {"family": "capitulation_repair", "rsi_max": 35, "vol_min": 1.0, "vol_max": 4.0, "close_position_min": 0.65, "industry_mode": "stable"},
    "capitulation_safety": {"family": "capitulation_repair", "rsi_max": 35, "vol_min": 1.0, "vol_max": 4.0, "close_position_min": 0.65, "quality_mode": "safety"},
    "limit_pullback": {"family": "limit_pullback", "rsi_max": 65, "vol_min": 0.5, "vol_max": 1.5, "close_position_min": 0.60},
    "limit_pullback_trend": {"family": "limit_pullback", "rsi_max": 65, "vol_min": 0.5, "vol_max": 1.5, "close_position_min": 0.60, "market_mode": "trend_support"},
    "limit_pullback_industry": {"family": "limit_pullback", "rsi_max": 65, "vol_min": 0.5, "vol_max": 1.5, "close_position_min": 0.60, "industry_mode": "leader"},
    "limit_pullback_safety": {"family": "limit_pullback", "rsi_max": 65, "vol_min": 0.5, "vol_max": 1.5, "close_position_min": 0.60, "quality_mode": "safety"},
    "breakout_retest": {"family": "breakout_retest", "rsi_max": 75, "vol_min": 0.6, "vol_max": 1.8, "close_position_min": 0.55},
    "breakout_retest_trend": {"family": "breakout_retest", "rsi_max": 75, "vol_min": 0.6, "vol_max": 1.8, "close_position_min": 0.55, "market_mode": "trend_support"},
    "breakout_retest_safety": {"family": "breakout_retest", "rsi_max": 75, "vol_min": 0.6, "vol_max": 1.8, "close_position_min": 0.55, "quality_mode": "safety"},
}

# Stage 2 is derived only from stage-1 discovery behavior.  It preserves the
# baseline signal and adds one sparse, context-gated event instead of globally
# filtering every baseline trade.
STAGE2_CANDIDATES = {
    "adaptive_union_cap_breadth": {
        "family": "adaptive_union_cap",
        "overlay_context": "breadth",
    },
    "adaptive_switch_cap_breadth": {
        "family": "adaptive_switch_cap",
        "overlay_context": "breadth",
    },
    "adaptive_union_cap_industry": {
        "family": "adaptive_union_cap",
        "overlay_context": "industry",
    },
    "adaptive_union_cap_breadth_industry": {
        "family": "adaptive_union_cap",
        "overlay_context": "breadth_industry",
    },
    "adaptive_union_spring_breadth": {
        "family": "adaptive_union_spring",
        "overlay_context": "breadth",
    },
    "adaptive_union_spring_industry": {
        "family": "adaptive_union_spring",
        "overlay_context": "industry",
    },
    "adaptive_union_cap_breadth_rsi_exit": {
        "family": "adaptive_union_cap",
        "overlay_context": "breadth",
        "exit_mode": "rsi_recovery",
    },
}

STAGE3_CANDIDATES = {
    **{
        f"rank_baseline_p{positions}": {
            "family": "baseline_ranked",
            "score_mode": "baseline",
            "_max_positions": positions,
        }
        for positions in (5, 8, 10, 15)
    },
    **{
        f"rank_recovery_p{positions}": {
            "family": "baseline_ranked",
            "score_mode": "recovery",
            "_max_positions": positions,
        }
        for positions in (5, 8, 10, 15)
    },
    **{
        f"rank_lottery_aware_p{positions}": {
            "family": "baseline_ranked",
            "score_mode": "lottery_aware",
            "_max_positions": positions,
        }
        for positions in (5, 8, 10)
    },
    **{
        f"rank_quality_recovery_p{positions}": {
            "family": "baseline_ranked",
            "score_mode": "quality_recovery",
            "_max_positions": positions,
        }
        for positions in (5, 8, 10)
    },
    **{
        f"rank_industry_recovery_p{positions}": {
            "family": "baseline_ranked",
            "score_mode": "industry_recovery",
            "_max_positions": positions,
        }
        for positions in (5, 8, 10)
    },
    **{
        f"rank_hybrid_p{positions}": {
            "family": "baseline_ranked",
            "score_mode": "hybrid",
            "_max_positions": positions,
        }
        for positions in (5, 8, 10)
    },
}

# Stage 4 is derived from the only robust Stage-3 direction: keep the original
# new-low entry, rank simultaneous candidates by intraday recovery, and spread
# capital across 15 names. It tests a small set of economically distinct exit
# hypotheses instead of optimizing a dense stop/holding-day grid.
STAGE4_CANDIDATES = {
    "recovery_p15_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "recovery_p15_harvest_ma10": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "ma10_recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "recovery_p15_harvest_rsi55": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "recovery_p15_short_half_life": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "time_only",
        "_max_positions": 15,
        "_max_hold_days": 10,
        "_stop_loss": -0.06,
    },
    "recovery_p15_long_half_life": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 15,
        "_max_hold_days": 20,
        "_stop_loss": -0.06,
    },
    "recovery_p15_tight_invalidation": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.04,
    },
    "recovery_p15_wide_invalidation": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
    },
    "recovery_p15_ma10_wide_stop": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "ma10_recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
    },
}

# Stage 5 combines the two complementary Stage-4 mechanisms only: give a
# reversal enough room to develop, then realize gains once RSI confirms that
# the oversold state has repaired. The two component variants remain controls.
STAGE5_CANDIDATES = {
    "recovery_p15_wide_stop_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
    },
    "recovery_p15_rsi55_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "recovery_p15_rsi55_wide_stop": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
    },
}

# Stage 6 restores the baseline's 10-name concentration while retaining the
# Stage-5 exit mechanism. Stage 3 showed that recovery ranking at 10 names won
# 9/14 folds but lacked enough full-period return under the original exits.
STAGE6_CANDIDATES = {
    "recovery_p10_original_exit_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "recovery_p15_rsi55_wide_stop_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
    },
    "recovery_p10_rsi55_wide_stop": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
    },
}

# Stage 7 starts a separate market-ecology path. Entry, exit, stop and position
# count all match the baseline; only the same-day cross-sectional ranking
# switches between the baseline and recovery scores using causal market state.
STAGE7_CANDIDATES = {
    "market_score_baseline_control": {
        "family": "baseline_ranked",
        "score_mode": "baseline",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "market_score_recovery_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "market_score_adaptive_weak_breadth": {
        "family": "baseline_ranked",
        "score_mode": "adaptive_weak_breadth",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "market_score_adaptive_trend_stress": {
        "family": "baseline_ranked",
        "score_mode": "adaptive_trend_stress",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
    "market_score_adaptive_breadth_repair": {
        "family": "baseline_ranked",
        "score_mode": "adaptive_breadth_repair",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
    },
}

# Stage 8 is frozen from Stage 7 and only corrects the historical universe.
# Both variants use point-in-time listing/name/share-capital eligibility; the
# candidate changes only the previously selected market-conditioned ranking.
STAGE8_CANDIDATES = {
    "pit_baseline_control": {
        "family": "baseline_ranked",
        "score_mode": "baseline",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_adaptive_trend_stress": {
        "family": "baseline_ranked",
        "score_mode": "adaptive_trend_stress",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
}

# Stage 9 is frozen after correcting the historical universe and before any
# 2019 OOS observation.  It does not tune Stage 8's failed 45% threshold.
# Instead, the observable fraction of the main board below MA20 continuously
# allocates ranking weight from the original score to one economically distinct
# defensive score.  Entry, exit, risk and portfolio construction stay fixed.
STAGE9_CANDIDATES = {
    "pit_continuous_baseline_control": {
        "family": "baseline_ranked",
        "score_mode": "baseline",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_continuous_stress_recovery": {
        "family": "baseline_ranked",
        "score_mode": "continuous_stress_recovery",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_continuous_stress_lottery": {
        "family": "baseline_ranked",
        "score_mode": "continuous_stress_lottery",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_continuous_stress_industry": {
        "family": "baseline_ranked",
        "score_mode": "continuous_stress_industry",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_continuous_stress_quality": {
        "family": "baseline_ranked",
        "score_mode": "continuous_stress_quality",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
}

# Stage 10 leaves the reversal signal family.  These are sparse event strategies
# designed to fix the repeated-entry defect of state-based trend strategies:
# one fires only on the first 60-day closing breakout; the other only on the
# first MA10 reclaim after a pullback inside an established MA20/MA60 uptrend.
STAGE10_CANDIDATES = {
    "pit_trend_baseline_control": {
        "family": "baseline_ranked",
        "score_mode": "baseline",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_sparse_breakout": {
        "family": "sparse_breakout",
        "score_mode": "trend_breakout",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_sparse_breakout_quality": {
        "family": "sparse_breakout",
        "score_mode": "trend_breakout_quality",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_first_trend_pullback": {
        "family": "first_trend_pullback",
        "score_mode": "trend_pullback",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_first_trend_pullback_industry": {
        "family": "first_trend_pullback",
        "score_mode": "trend_pullback_industry",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
}

# Stage 11 addresses Stage 10's structural turnover failure without retuning
# price or volume thresholds.  A 60-session cooldown matches the signal's
# 60-day horizon, and the retest family waits for the first MA10 reclaim within
# 20 sessions after a genuine closing breakout instead of buying the breakout.
STAGE11_CANDIDATES = {
    "pit_sparse_baseline_control": {
        "family": "baseline_ranked",
        "score_mode": "baseline",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_sparse_breakout_cooldown60": {
        "family": "sparse_breakout",
        "score_mode": "trend_breakout",
        "eligibility_mode": "pit",
        "event_cooldown_days": 60,
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_breakout_first_retest": {
        "family": "breakout_first_retest",
        "score_mode": "trend_pullback",
        "eligibility_mode": "pit",
        "event_cooldown_days": 60,
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_breakout_first_retest_industry": {
        "family": "breakout_first_retest",
        "score_mode": "trend_pullback_industry",
        "eligibility_mode": "pit",
        "event_cooldown_days": 60,
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
}

# Stage 12 replays the strongest pre-existing reversal/risk hypotheses on the
# corrected PIT universe.  Values are copied unchanged from Stages 4-6 rather
# than re-estimated from the corrected discovery results.
STAGE12_CANDIDATES = {
    "pit_risk_baseline_control": {
        "family": "baseline_ranked",
        "score_mode": "baseline",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_recovery_p15_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "eligibility_mode": "pit",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_recovery_p15_rsi55_control": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "eligibility_mode": "pit",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.06,
        "_pit_universe": True,
    },
    "pit_recovery_p15_rsi55_wide_stop": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "eligibility_mode": "pit",
        "_max_positions": 15,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
        "_pit_universe": True,
    },
    "pit_recovery_p10_rsi55_wide_stop": {
        "family": "baseline_ranked",
        "score_mode": "recovery",
        "exit_mode": "rsi_recovery",
        "eligibility_mode": "pit",
        "_max_positions": 10,
        "_max_hold_days": 15,
        "_stop_loss": -0.08,
        "_pit_universe": True,
    },
}

DISCOVERY_FOLDS = (
    ("2020H1", date(2020, 1, 1), date(2020, 6, 30)),
    ("2020H2", date(2020, 7, 1), date(2020, 12, 31)),
    ("2021H1", date(2021, 1, 1), date(2021, 6, 30)),
    ("2021H2", date(2021, 7, 1), date(2021, 12, 31)),
    ("2022H1", date(2022, 1, 1), date(2022, 6, 30)),
    ("2022H2", date(2022, 7, 1), date(2022, 12, 31)),
    ("2023H1", date(2023, 1, 1), date(2023, 6, 30)),
    ("2023H2", date(2023, 7, 1), date(2023, 12, 31)),
)
OOS_FOLDS = (
    ("2024H1", date(2024, 1, 1), date(2024, 6, 30)),
    ("2024H2", date(2024, 7, 1), date(2024, 12, 31)),
    ("2025H1", date(2025, 1, 1), date(2025, 6, 30)),
    ("2025H2", date(2025, 7, 1), date(2025, 12, 31)),
    ("2026H1", date(2026, 1, 1), date(2026, 6, 30)),
    ("2026H2_partial", date(2026, 7, 1), date(2026, 8, 26)),
)

STAGE3_DISCOVERY_FOLDS = (
    *DISCOVERY_FOLDS,
    *OOS_FOLDS,
)
STAGE3_OOS_FOLDS = tuple(
    (
        f"{year}{half}",
        date(year, 1 if half == "H1" else 7, 1),
        date(year, 6, 30) if half == "H1" else date(year, 12, 31),
    )
    for year in range(2014, 2019)
    for half in ("H1", "H2")
)
STAGE8_OOS_FOLDS = (
    ("2019H1", date(2019, 1, 1), date(2019, 6, 30)),
    ("2019H2", date(2019, 7, 1), date(2019, 12, 31)),
)


def _config(
    strategy_id: str,
    start: date,
    end: date,
    params: dict | None = None,
    overrides: dict | None = None,
    max_positions: int = 10,
    max_hold_days: int | None = None,
    stop_loss: float | None = None,
    basic_filter_override: dict | None = None,
) -> StrategyBacktestConfig:
    effective_overrides = dict(overrides or {})
    if max_hold_days is not None:
        effective_overrides["max_hold_days"] = max_hold_days
    if stop_loss is not None:
        effective_overrides["stop_loss"] = stop_loss
    if basic_filter_override is not None:
        effective_overrides["basic_filter"] = basic_filter_override
    return StrategyBacktestConfig(
        strategy_id=strategy_id,
        symbols=None,
        start=start,
        end=end,
        params=params,
        overrides=effective_overrides,
        matching="open_t+1",
        entry_fill="open_t+1",
        exit_fill="open_t+1",
        fees_pct=0.0002,
        commission_pct=0.0002,
        stamp_tax_pct=0.001,
        slippage_bps=5.0,
        max_positions=max_positions,
        max_exposure_pct=1.0,
        initial_capital=200_000.0,
        position_sizing="equal",
        mode="position",
        asset_type="stock",
        holding_days=15,
        minute_fill=False,
    )


def _stats(result) -> dict:
    if result.error:
        raise RuntimeError(result.error)
    return {key: result.stats.get(key) for key in POLICY.required_stats}


def _run(service, config, prepared) -> dict:
    return _stats(service.run(config, prepared=prepared, result_policy=POLICY))


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _candidate_params(raw: dict) -> dict:
    return {**COMMON, **{key: value for key, value in raw.items() if not key.startswith("_")}}


def _candidate_max_positions(raw: dict) -> int:
    return int(raw.get("_max_positions", 10))


def _candidate_execution(raw: dict) -> dict:
    execution = {
        "max_positions": _candidate_max_positions(raw),
        "max_hold_days": int(raw.get("_max_hold_days", 15)),
        "stop_loss": float(raw.get("_stop_loss", -0.06)),
    }
    if raw.get("_pit_universe"):
        execution["basic_filter_override"] = {
            **RESEARCH_BASIC_FILTER,
            "exclude_st": False,
        }
    return execution


def _summary(candidate: list[dict], baseline: list[dict], full_candidate: dict, full_baseline: dict) -> dict:
    returns = [_finite(item["total_return"]) for item in candidate]
    base_returns = [_finite(item["total_return"]) for item in baseline]
    excess = [left - right for left, right in zip(returns, base_returns, strict=True)]
    sharpes = [_finite(item["sharpe"]) for item in candidate]
    base_sharpes = [_finite(item["sharpe"]) for item in baseline]
    positive = sum(value > 0 for value in returns)
    baseline_positive = sum(value > 0 for value in base_returns)
    beats = sum(value > base for value, base in zip(returns, base_returns, strict=True))
    trades = sum(int(_finite(item["n_trades"])) for item in candidate)
    full_excess = _finite(full_candidate["total_return"]) - _finite(full_baseline["total_return"])
    dd_gap = abs(_finite(full_candidate["max_drawdown"])) - abs(_finite(full_baseline["max_drawdown"]))
    score = (
        full_excess
        + statistics.median(excess)
        + 0.02 * statistics.median(
            [left - right for left, right in zip(sharpes, base_sharpes, strict=True)]
        )
        - 0.5 * max(0.0, dd_gap)
    )
    qualified = (
        positive >= baseline_positive
        and beats >= math.ceil(len(candidate) * 0.60)
        and trades >= 60
        and full_excess > 0
        and _finite(full_candidate["sharpe"]) > _finite(full_baseline["sharpe"])
        and dd_gap <= 0.05
    )
    return {
        "score": round(score, 6),
        "qualified": qualified,
        "positive_folds": positive,
        "baseline_positive_folds": baseline_positive,
        "beats_baseline_folds": beats,
        "folds": len(candidate),
        "trades": trades,
        "median_excess": round(statistics.median(excess), 6),
        "full": full_candidate,
        "baseline_full": full_baseline,
    }


def _engine(data_dir: Path, research_dir: Path):
    builtin = Path(__file__).resolve().parents[1] / "app" / "strategy" / "builtin"
    strategy_engine = StrategyEngine(
        [builtin, research_dir],
        override_loader=lambda strategy_id: strategy_config.load_override(data_dir, strategy_id),
    )
    if strategy_engine.load_errors():
        raise RuntimeError(f"strategy load errors: {strategy_engine.load_errors()}")
    repo = KlineRepository(DataStore(data_dir))
    return strategy_engine, StrategyBacktestService(BacktestEngine(repo), strategy_engine)


def _prepared(service, configs, base_market=None):
    return service.prepare_matrix_optimization(
        configs,
        matrix_cache_max_bytes=768 * 1024 * 1024,
        market_data_override=base_market,
    )


def _prepared_groups(service, names, configs, base_market):
    grouped: dict[tuple, list[tuple[str, StrategyBacktestConfig]]] = {}
    for name, config in zip(names, configs, strict=True):
        signature = service._matrix_prepare_signature(config)
        grouped.setdefault(signature, []).append((name, config))
    by_name = {}
    prepared_objects = []
    for items in grouped.values():
        prepared = _prepared(service, [config for _, config in items], base_market)
        prepared_objects.append(prepared)
        by_name.update({name: prepared for name, _ in items})
    return by_name, prepared_objects


def _attach_industry_context(market, data_dir: Path):
    research_dir = data_dir / "research"
    membership_path = research_dir / "sw_l1_membership.parquet"
    context_path = research_dir / "sw_l1_daily_context.parquet"
    if not membership_path.exists() or not context_path.exists():
        raise RuntimeError("point-in-time SW industry research data is missing")

    membership = pl.read_parquet(membership_path).sort(["symbol", "in_date"])
    context = pl.read_parquet(context_path).select(
        "l1_code", "date", "industry_momentum_20d", "industry_breadth_5d"
    )
    labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
    date_to_row = {value: index for index, value in enumerate(labels)}
    codes = sorted(context["l1_code"].unique().to_list())
    code_to_column = {code: index for index, code in enumerate(codes)}
    context_momentum = np.full((len(labels), len(codes)), np.nan, dtype=np.float32)
    context_breadth = np.full((len(labels), len(codes)), np.nan, dtype=np.float32)
    for row in context.iter_rows(named=True):
        time_id = date_to_row.get(row["date"])
        code_id = code_to_column.get(row["l1_code"])
        if time_id is None or code_id is None:
            continue
        context_momentum[time_id, code_id] = _finite(row["industry_momentum_20d"], np.nan)
        context_breadth[time_id, code_id] = _finite(row["industry_breadth_5d"], np.nan)

    momentum = np.full(market.shape, np.nan, dtype=np.float32)
    breadth = np.full(market.shape, np.nan, dtype=np.float32)
    grouped = membership.partition_by("symbol", as_dict=True, include_key=False)
    for asset_id, symbol in enumerate(market.symbols):
        rows = grouped.get((symbol,))
        if rows is None:
            continue
        for row in rows.iter_rows(named=True):
            code_id = code_to_column.get(row["l1_code"])
            if code_id is None:
                continue
            start = bisect.bisect_left(labels, row["in_date"])
            end_date = row["out_date"] or labels[-1]
            stop = bisect.bisect_right(labels, end_date)
            if start >= stop:
                continue
            momentum[start:stop, asset_id] = context_momentum[start:stop, code_id]
            breadth[start:stop, asset_id] = context_breadth[start:stop, code_id]
    momentum.flags.writeable = False
    breadth.flags.writeable = False
    return replace(
        market,
        fields={
            **market.fields,
            "industry_momentum_20d": momentum,
            "industry_breadth_5d": breadth,
        },
    )


def _value_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _attach_point_in_time_universe(market, data_dir: Path):
    research_dir = data_dir / "research"
    universe_path = research_dir / "historical_stock_universe.parquet"
    names_path = research_dir / "historical_stock_names.parquet"
    if not universe_path.exists() or not names_path.exists():
        raise RuntimeError("point-in-time historical stock universe is missing")
    shares = load_share_history(data_dir)
    if shares.is_empty():
        raise RuntimeError("point-in-time share history is missing")

    universe = pl.read_parquet(universe_path)
    names = pl.read_parquet(names_path).sort(["symbol", "start_date"])
    labels = [date.fromisoformat(value[:10]) for value in market.timestamp_labels]
    total_shares = np.full(market.shape, np.nan, dtype=np.float32)
    float_shares = np.full(market.shape, np.nan, dtype=np.float32)
    listed = np.zeros(market.shape, dtype=bool)
    name_known = np.zeros(market.shape, dtype=bool)
    eligible_name = np.zeros(market.shape, dtype=bool)

    universe_by_symbol = {
        row["symbol"]: row for row in universe.iter_rows(named=True)
    }
    names_by_symbol = names.partition_by("symbol", as_dict=True, include_key=False)
    shares_by_symbol = shares.sort(["symbol", "period_end"]).partition_by(
        "symbol", as_dict=True, include_key=False
    )

    for asset_id, symbol in enumerate(market.symbols):
        universe_row = universe_by_symbol.get(symbol)
        if universe_row is None:
            continue
        list_date = _value_date(universe_row.get("list_date"))
        delist_date = _value_date(universe_row.get("delist_date")) or labels[-1]
        if list_date is None:
            continue
        list_start = bisect.bisect_left(labels, list_date)
        list_stop = bisect.bisect_right(labels, delist_date)
        if list_start >= list_stop:
            continue
        listed[list_start:list_stop, asset_id] = True

        name_rows = names_by_symbol.get((symbol,))
        if name_rows is not None:
            for row in name_rows.iter_rows(named=True):
                start_date = _value_date(row.get("start_date"))
                end_date = _value_date(row.get("end_date")) or delist_date
                if start_date is None:
                    continue
                start_id = max(list_start, bisect.bisect_left(labels, start_date))
                stop_id = min(list_stop, bisect.bisect_right(labels, end_date))
                if start_id >= stop_id:
                    continue
                name_known[start_id:stop_id, asset_id] = True
                name = str(row.get("name") or "").upper()
                eligible_name[start_id:stop_id, asset_id] = not any(
                    token in name for token in ("ST", "*ST", "退")
                )

        share_rows = shares_by_symbol.get((symbol,))
        if share_rows is None:
            continue
        points = []
        for row in share_rows.iter_rows(named=True):
            available_date = _value_date(row.get("announce_date")) or _value_date(
                row.get("period_end")
            )
            total = _finite(row.get("total_shares"), np.nan)
            floating = _finite(row.get("float_shares"), np.nan)
            if (
                available_date is None
                or not math.isfinite(total)
                or total <= 0
                or not math.isfinite(floating)
                or floating <= 0
            ):
                continue
            points.append((available_date, total, floating))
        points.sort(key=lambda item: item[0])
        for point_id, (available_date, total, floating) in enumerate(points):
            start_id = max(list_start, bisect.bisect_left(labels, available_date))
            next_date = points[point_id + 1][0] if point_id + 1 < len(points) else None
            stop_id = (
                min(list_stop, bisect.bisect_left(labels, next_date))
                if next_date is not None
                else list_stop
            )
            if start_id >= stop_id:
                continue
            total_shares[start_id:stop_id, asset_id] = np.float32(total)
            float_shares[start_id:stop_id, asset_id] = np.float32(floating)

    share_known = (
        np.isfinite(total_shares)
        & (total_shares > 0)
        & np.isfinite(float_shares)
        & (float_shares > 0)
        & (float_shares <= total_shares)
    )
    eligible = listed & name_known & eligible_name & share_known
    eligible_field = eligible.astype(np.float32)
    listed_count = int(listed.sum())
    context_stats = {
        "symbols": len(market.symbols),
        "listed_symbol_days": listed_count,
        "name_coverage": round(float((listed & name_known).sum()) / listed_count, 6)
        if listed_count
        else 0.0,
        "share_coverage": round(float((listed & share_known).sum()) / listed_count, 6)
        if listed_count
        else 0.0,
        "eligible_symbol_days": int(eligible.sum()),
        "eligible_symbols": int((eligible.sum(axis=0) > 0).sum()),
    }
    for values in (total_shares, float_shares, eligible_field):
        values.flags.writeable = False
    return (
        replace(
            market,
            fields={
                **market.fields,
                "total_shares": total_shares,
                "float_shares": float_shares,
                "pit_eligible": eligible_field,
            },
        ),
        context_stats,
    )


def discover(
    data_dir: Path,
    research_dir: Path,
    output: Path,
    candidates: dict[str, dict],
    stage: int,
) -> None:
    _strategy_engine, service = _engine(data_dir, research_dir)
    if stage in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}:
        start, end = date(2020, 1, 1), date(2026, 8, 26)
        discovery_folds = STAGE3_DISCOVERY_FOLDS
    else:
        start, end = date(2020, 1, 1), date(2023, 12, 31)
        discovery_folds = DISCOVERY_FOLDS
    names = list(candidates)
    params = {name: _candidate_params(candidates[name]) for name in names}
    executions = {name: _candidate_execution(candidates[name]) for name in names}
    base_params = params[names[0]]
    if stage in {8, 9, 10, 11, 12}:
        base_params = {**base_params, "eligibility_mode": "none"}
    base_prepared = _prepared(
        service,
        [
            _config(
                "reversal_first_principles",
                start,
                end,
                base_params,
                **executions[names[0]],
            )
        ],
    )
    market = _attach_industry_context(base_prepared.market_data, data_dir)
    pit_context = None
    if stage in {8, 9, 10, 11, 12}:
        market, pit_context = _attach_point_in_time_universe(market, data_dir)
    try:
        baseline_override = strategy_config.load_override(data_dir, "n_day_low_reversal")
        baseline_strategy_id = "n_day_low_reversal"
        baseline_params = None
        baseline_execution = {}
        if stage in {8, 9, 10, 11, 12}:
            pit_baseline_name = {
                8: "pit_baseline_control",
                9: "pit_continuous_baseline_control",
                10: "pit_trend_baseline_control",
                11: "pit_sparse_baseline_control",
                12: "pit_risk_baseline_control",
            }[stage]
            baseline_strategy_id = "reversal_first_principles"
            baseline_params = params[pit_baseline_name]
            baseline_override = None
            baseline_execution = executions[pit_baseline_name]
        fold_candidate: dict[str, list[dict]] = {name: [] for name in names}
        fold_baseline: list[dict] = []
        detailed: dict[str, dict] = {}
        for label, fold_start, fold_end in discovery_folds:
            candidate_configs = [
                _config(
                    "reversal_first_principles",
                    fold_start,
                    fold_end,
                    params[name],
                    **executions[name],
                )
                for name in names
            ]
            candidate_prepared, candidate_prepared_objects = _prepared_groups(
                service, names, candidate_configs, market
            )
            baseline_config = _config(
                baseline_strategy_id,
                fold_start,
                fold_end,
                params=baseline_params,
                overrides=baseline_override,
                **baseline_execution,
            )
            baseline_prepared = _prepared(service, [baseline_config], market)
            try:
                baseline_stats = _run(service, baseline_config, baseline_prepared)
                fold_baseline.append(baseline_stats)
                detailed[label] = {"baseline": baseline_stats, "candidates": {}}
                for name, config in zip(names, candidate_configs, strict=True):
                    stats = _run(service, config, candidate_prepared[name])
                    fold_candidate[name].append(stats)
                    detailed[label]["candidates"][name] = stats
            finally:
                for prepared in candidate_prepared_objects:
                    prepared.compute_cache.close()
                baseline_prepared.compute_cache.close()

        full_candidate_configs = [
            _config(
                "reversal_first_principles",
                start,
                end,
                params[name],
                **executions[name],
            )
            for name in names
        ]
        full_prepared, full_prepared_objects = _prepared_groups(
            service, names, full_candidate_configs, market
        )
        baseline_config = _config(
            baseline_strategy_id,
            start,
            end,
            params=baseline_params,
            overrides=baseline_override,
            **baseline_execution,
        )
        baseline_prepared = _prepared(service, [baseline_config], market)
        try:
            full_baseline = _run(service, baseline_config, baseline_prepared)
            summaries = {}
            for name, config in zip(names, full_candidate_configs, strict=True):
                full_candidate = _run(service, config, full_prepared[name])
                summaries[name] = _summary(
                    fold_candidate[name], fold_baseline, full_candidate, full_baseline
                )
        finally:
            for prepared in full_prepared_objects:
                prepared.compute_cache.close()
            baseline_prepared.compute_cache.close()

        ranking = sorted(summaries, key=lambda name: summaries[name]["score"], reverse=True)
        qualified = [name for name in ranking if summaries[name]["qualified"]]
        winner = qualified[0] if qualified else None
        payload = {
            "phase": "discovery",
            "stage": stage,
            "method": (
                f"fixed hypotheses; {len(discovery_folds)} half-year folds; OOS not evaluated; "
                "positive-fold stability is relative to the baseline"
            ),
            "range": [start.isoformat(), end.isoformat()],
            "baseline": {
                "id": baseline_strategy_id,
                "params": baseline_params,
                "override": baseline_override,
                "execution": baseline_execution,
            },
            "point_in_time_context": pit_context,
            "ranking": ranking,
            "summaries": summaries,
            "fold_detail": detailed,
            "winner": winner,
            "winner_params": params.get(winner) if winner else None,
            "winner_execution": (
                executions[winner] if winner else None
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"winner": winner, "ranking": ranking[:10], "summaries": {name: summaries[name] for name in ranking[:10]}}, ensure_ascii=False, indent=2))
    finally:
        base_prepared.compute_cache.close()


def oos(data_dir: Path, research_dir: Path, discovery_file: Path, output: Path) -> None:
    discovery_payload = json.loads(discovery_file.read_text(encoding="utf-8"))
    winner = discovery_payload.get("winner")
    params = discovery_payload.get("winner_params")
    execution = discovery_payload.get("winner_execution") or {"max_positions": 10}
    if not winner or not isinstance(params, dict):
        raise RuntimeError("discovery produced no qualified frozen winner; OOS must not run")
    _, service = _engine(data_dir, research_dir)
    stage = int(discovery_payload.get("stage") or 1)
    baseline_payload = discovery_payload.get("baseline") or {}
    baseline_strategy_id = str(baseline_payload.get("id") or "n_day_low_reversal")
    baseline_params = baseline_payload.get("params")
    baseline_override = baseline_payload.get("override")
    baseline_execution = baseline_payload.get("execution") or {}
    if stage in {8, 9, 10, 11, 12}:
        start, end = date(2019, 1, 1), date(2019, 12, 31)
        oos_folds = STAGE8_OOS_FOLDS
    elif stage in {3, 4, 5, 6, 7}:
        start, end = date(2014, 1, 1), date(2018, 12, 31)
        oos_folds = STAGE3_OOS_FOLDS
    else:
        start, end = date(2024, 1, 1), date(2026, 8, 26)
        oos_folds = OOS_FOLDS
    candidate_execution = {
        "max_positions": int(execution.get("max_positions", 10)),
        "max_hold_days": int(execution.get("max_hold_days", 15)),
        "stop_loss": float(execution.get("stop_loss", -0.06)),
    }
    if execution.get("basic_filter_override") is not None:
        candidate_execution["basic_filter_override"] = execution[
            "basic_filter_override"
        ]
    base_params = params
    if stage in {8, 9, 10, 11, 12}:
        base_params = {**params, "eligibility_mode": "none"}
    base_prepared = _prepared(
        service,
        [
            _config(
                "reversal_first_principles",
                start,
                end,
                base_params,
                **candidate_execution,
            )
        ],
    )
    market = _attach_industry_context(base_prepared.market_data, data_dir)
    pit_context = None
    if stage in {8, 9, 10, 11, 12}:
        market, pit_context = _attach_point_in_time_universe(market, data_dir)
    try:
        fold_rows = []
        for label, fold_start, fold_end in oos_folds:
            candidate_config = _config(
                "reversal_first_principles",
                fold_start,
                fold_end,
                params,
                **candidate_execution,
            )
            baseline_config = _config(
                baseline_strategy_id,
                fold_start,
                fold_end,
                params=baseline_params,
                overrides=baseline_override,
                **baseline_execution,
            )
            candidate_prepared = _prepared(service, [candidate_config], market)
            baseline_prepared = _prepared(service, [baseline_config], market)
            try:
                fold_rows.append(
                    {
                        "label": label,
                        "candidate": _run(service, candidate_config, candidate_prepared),
                        "baseline": _run(service, baseline_config, baseline_prepared),
                    }
                )
            finally:
                candidate_prepared.compute_cache.close()
                baseline_prepared.compute_cache.close()

        candidate_config = _config(
            "reversal_first_principles",
            start,
            end,
            params,
            **candidate_execution,
        )
        baseline_config = _config(
            baseline_strategy_id,
            start,
            end,
            params=baseline_params,
            overrides=baseline_override,
            **baseline_execution,
        )
        candidate_prepared = _prepared(service, [candidate_config], market)
        baseline_prepared = _prepared(service, [baseline_config], market)
        try:
            full_candidate = _run(service, candidate_config, candidate_prepared)
            full_baseline = _run(service, baseline_config, baseline_prepared)
        finally:
            candidate_prepared.compute_cache.close()
            baseline_prepared.compute_cache.close()

        positive = sum(_finite(row["candidate"]["total_return"]) > 0 for row in fold_rows)
        baseline_positive = sum(
            _finite(row["baseline"]["total_return"]) > 0 for row in fold_rows
        )
        beats = sum(
            _finite(row["candidate"]["total_return"]) > _finite(row["baseline"]["total_return"])
            for row in fold_rows
        )
        pass_gate = (
            positive >= baseline_positive
            and beats >= math.ceil(len(fold_rows) * 0.60)
            and _finite(full_candidate["total_return"]) > _finite(full_baseline["total_return"])
            and _finite(full_candidate["sharpe"]) >= _finite(full_baseline["sharpe"])
            and _finite(full_candidate["max_drawdown"]) >= _finite(full_baseline["max_drawdown"]) - 0.05
            and _finite(full_candidate["n_trades"]) >= 60
        )
        payload = {
            "phase": "oos",
            "frozen_from": str(discovery_file),
            "candidate": winner,
            "params": params,
            "execution": execution,
            "point_in_time_context": pit_context,
            "range": [start.isoformat(), end.isoformat()],
            "full_candidate": full_candidate,
            "full_baseline": full_baseline,
            "positive_folds": positive,
            "baseline_positive_folds": baseline_positive,
            "beats_baseline_folds": beats,
            "folds": fold_rows,
            "pass_gate": pass_gate,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        base_prepared.compute_cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("discover", "oos"))
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--discovery-file", type=Path, default=Path("/tmp/reversal_discovery.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 13), default=1)
    args = parser.parse_args()
    if args.phase == "discover":
        candidates = {
            1: STAGE1_CANDIDATES,
            2: STAGE2_CANDIDATES,
            3: STAGE3_CANDIDATES,
            4: STAGE4_CANDIDATES,
            5: STAGE5_CANDIDATES,
            6: STAGE6_CANDIDATES,
            7: STAGE7_CANDIDATES,
            8: STAGE8_CANDIDATES,
            9: STAGE9_CANDIDATES,
            10: STAGE10_CANDIDATES,
            11: STAGE11_CANDIDATES,
            12: STAGE12_CANDIDATES,
        }[args.stage]
        discover(args.data_dir, args.research_dir, args.output, candidates, args.stage)
    else:
        oos(args.data_dir, args.research_dir, args.discovery_file, args.output)


if __name__ == "__main__":
    main()
