from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.backtest.matrix import (
    MatrixPipelineConfig,
    MatrixStrategyPipeline,
    build_market_data_matrix,
)
from app.strategy.builtin.n_day_low_reversal import NDayLowReversalMatrixStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "research" / "strategies" / "reversal_first_principles.py"
INDEPENDENT_STRATEGY_PATH = (
    REPO_ROOT / "research" / "strategies" / "independent_alpha_families.py"
)


def _strategy_module():
    spec = importlib.util.spec_from_file_location("research_reversal_strategy", STRATEGY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _independent_strategy_module():
    spec = importlib.util.spec_from_file_location(
        "independent_alpha_families", INDEPENDENT_STRATEGY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_module():
    path = REPO_ROOT / "research" / "run_reversal_study.py"
    spec = importlib.util.spec_from_file_location("research_reversal_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel(days: int = 140, assets: int = 12) -> pl.DataFrame:
    rows = []
    for day_id in range(days):
        for asset_id in range(assets):
            symbol = f"600{asset_id:03d}.SH"
            trend = 18.0 + asset_id * 0.2 - day_id * (0.025 + asset_id * 0.0003)
            close = trend + np.sin(day_id / 7.0 + asset_id) * 0.3
            open_ = close - (0.10 if day_id % 4 == 0 else -0.03)
            volume = 100_000.0 + (day_id % 7) * 5_000.0
            if day_id in {80, 110}:
                close -= 1.5
                open_ = close - 0.15
                volume *= 2.5
            rows.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "date": date(2020, 1, 1) + timedelta(days=day_id),
                    "open": open_,
                    "high": max(open_, close) + 0.20,
                    "low": min(open_, close) - 0.20,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                    "turnover_rate": 2.0 + asset_id * 0.05,
                    "total_shares": 1_000_000_000.0,
                    "float_shares": 800_000_000.0,
                    "consecutive_limit_ups": 0,
                    "roe_latest": 10.0 + asset_id * 0.2,
                    "net_margin_latest": 6.0 + asset_id * 0.1,
                    "revenue_yoy_latest": 5.0 + asset_id * 0.2,
                    "debt_ratio_latest": 55.0 - asset_id * 0.3,
                    "industry_momentum_20d": 0.02 + asset_id * 0.001,
                    "industry_breadth_5d": 0.52 + asset_id * 0.002,
                }
            )
    return pl.DataFrame(rows)


def _market(panel: pl.DataFrame):
    core = {"symbol", "name", "date", "open", "high", "low", "close", "volume"}
    return build_market_data_matrix(panel, field_columns=set(panel.columns) - core)


def _rank_params(score_mode: str = "hybrid") -> dict:
    return {
        "family": "baseline_ranked",
        "score_mode": score_mode,
        "rsi_max": 100.0,
        "vol_min": 0.0,
        "vol_max": 99.0,
        "close_position_min": 0.0,
        "market_mode": "none",
        "industry_mode": "none",
        "overlay_context": "none",
        "quality_mode": "none",
        "exit_mode": "ma20_cross",
    }


def test_ranked_research_preserves_baseline_entry_definition():
    market = _market(_panel())
    research = _strategy_module().MATRIX_STRATEGY.compute_signals(
        market, _rank_params("recovery")
    )
    baseline = NDayLowReversalMatrixStrategy().compute_signals(
        market,
        {
            "require_n_day_low": True,
            "require_bullish_candle": True,
            "use_volume_filter": True,
            "vol_ratio_min": 1.5,
        },
    )

    np.testing.assert_array_equal(research.entry, baseline.entry)


def test_baseline_rank_mode_matches_framework_baseline_score():
    market = _market(_panel())
    module = _strategy_module()
    basic_filter = module.META["basic_filter"]
    pipeline = MatrixStrategyPipeline()
    research = pipeline.run(
        module.MATRIX_STRATEGY,
        market,
        _rank_params("baseline"),
        MatrixPipelineConfig(
            basic_filter=basic_filter,
            scoring={},
            order_by="score",
            descending=True,
        ),
    )
    baseline = pipeline.run(
        NDayLowReversalMatrixStrategy(),
        market,
        {
            "require_n_day_low": True,
            "require_bullish_candle": True,
            "use_volume_filter": True,
            "vol_ratio_min": 1.5,
        },
        MatrixPipelineConfig(
            basic_filter=basic_filter,
            scoring={"change_pct": 0.4, "vol_ratio_5d": 0.3, "momentum_5d": 0.3},
            order_by="score",
            descending=True,
        ),
    )

    np.testing.assert_array_equal(research.entry, baseline.entry)
    np.testing.assert_allclose(research.score, baseline.score)


def test_ranked_research_does_not_change_past_signals_when_future_changes():
    original = _panel()
    changed = original.with_columns(
        pl.when(pl.col("date") > date(2020, 1, 1) + timedelta(days=99))
        .then(pl.col("close") * 1.8)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("date") > date(2020, 1, 1) + timedelta(days=99))
        .then(pl.col("volume") * 3.0)
        .otherwise(pl.col("volume"))
        .alias("volume"),
        pl.when(pl.col("date") > date(2020, 1, 1) + timedelta(days=99))
        .then(pl.col("industry_momentum_20d") + 0.5)
        .otherwise(pl.col("industry_momentum_20d"))
        .alias("industry_momentum_20d"),
    )
    strategy = _strategy_module().MATRIX_STRATEGY
    for mode in (
        "hybrid",
        "adaptive_weak_breadth",
        "adaptive_trend_stress",
        "adaptive_breadth_repair",
        "continuous_stress_recovery",
        "continuous_stress_lottery",
        "continuous_stress_industry",
        "continuous_stress_quality",
        "trend_breakout",
        "trend_breakout_quality",
        "trend_pullback",
        "trend_pullback_industry",
    ):
        first = strategy.compute_signals(_market(original), _rank_params(mode))
        second = strategy.compute_signals(_market(changed), _rank_params(mode))

        np.testing.assert_array_equal(first.entry[:100], second.entry[:100])
        np.testing.assert_array_equal(first.exit[:100], second.exit[:100])
        np.testing.assert_allclose(first.score[:100], second.score[:100])


def test_stage4_keeps_entry_family_and_only_varies_exit_execution():
    runner = _runner_module()
    for raw in runner.STAGE4_CANDIDATES.values():
        params = runner._candidate_params(raw)
        execution = runner._candidate_execution(raw)
        assert params["family"] == "baseline_ranked"
        assert params["score_mode"] == "recovery"
        assert params["market_mode"] == "none"
        assert params["industry_mode"] == "none"
        assert execution["max_positions"] == 15
        assert execution["max_hold_days"] in {10, 15, 20}
        assert execution["stop_loss"] in {-0.04, -0.06, -0.08}


def test_research_config_applies_frozen_risk_overrides():
    config = _runner_module()._config(
        "reversal_first_principles",
        date(2020, 1, 1),
        date(2020, 12, 31),
        max_positions=15,
        max_hold_days=10,
        stop_loss=-0.08,
    )

    assert config.max_positions == 15
    assert config.overrides == {"max_hold_days": 10, "stop_loss": -0.08}


def test_stage5_has_one_combined_mechanism_and_two_controls():
    runner = _runner_module()
    candidates = runner.STAGE5_CANDIDATES

    assert set(candidates) == {
        "recovery_p15_wide_stop_control",
        "recovery_p15_rsi55_control",
        "recovery_p15_rsi55_wide_stop",
    }
    combined = candidates["recovery_p15_rsi55_wide_stop"]
    assert combined["exit_mode"] == "rsi_recovery"
    assert combined["_stop_loss"] == -0.08


def test_stage6_changes_concentration_without_retuning_exit_thresholds():
    runner = _runner_module()
    candidate = runner.STAGE6_CANDIDATES["recovery_p10_rsi55_wide_stop"]

    assert candidate["score_mode"] == "recovery"
    assert candidate["exit_mode"] == "rsi_recovery"
    assert candidate["_max_positions"] == 10
    assert candidate["_max_hold_days"] == 15
    assert candidate["_stop_loss"] == -0.08


def test_stage7_changes_only_market_conditioned_ranking():
    runner = _runner_module()

    assert len(runner.STAGE7_CANDIDATES) == 5
    for candidate in runner.STAGE7_CANDIDATES.values():
        assert candidate["family"] == "baseline_ranked"
        assert candidate["_max_positions"] == 10
        assert candidate["_max_hold_days"] == 15
        assert candidate["_stop_loss"] == -0.06
        assert candidate.get("exit_mode", "ma20_cross") == "ma20_cross"


def test_point_in_time_universe_attaches_dynamic_shares_and_name_eligibility(tmp_path):
    runner = _runner_module()
    market = _market(_panel(days=10, assets=1))
    research_root = tmp_path / "research"
    research_root.mkdir()
    pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "name": ["测试股份"],
            "market": ["主板"],
            "exchange": ["SSE"],
            "list_status": ["L"],
            "list_date": [date(2020, 1, 1)],
            "delist_date": [None],
        }
    ).write_parquet(research_root / "historical_stock_universe.parquet")
    pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "name": ["测试股份", "*ST测试"],
            "start_date": [date(2020, 1, 1), date(2020, 1, 6)],
            "end_date": [date(2020, 1, 5), None],
        }
    ).write_parquet(research_root / "historical_stock_names.parquet")
    share_root = tmp_path / "financials" / "shares"
    share_root.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "period_end": ["2020-01-01", "2020-01-08"],
            "announce_date": ["2020-01-01", "2020-01-08"],
            "total_shares": [1_000_000_000.0, 1_100_000_000.0],
            "float_shares": [800_000_000.0, 850_000_000.0],
        }
    ).write_parquet(share_root / "part.parquet")

    result, stats = runner._attach_point_in_time_universe(market, tmp_path)

    assert result.fields["pit_eligible"][:5, 0].tolist() == [1.0] * 5
    assert result.fields["pit_eligible"][5:, 0].tolist() == [0.0] * 5
    assert np.isclose(result.fields["total_shares"][6, 0], 1_000_000_000.0)
    assert np.isclose(result.fields["total_shares"][7, 0], 1_100_000_000.0)
    assert stats["name_coverage"] == 1.0
    assert stats["share_coverage"] == 1.0


def test_stage8_freezes_prior_candidate_and_changes_only_ranking():
    runner = _runner_module()
    baseline = runner.STAGE8_CANDIDATES["pit_baseline_control"]
    candidate = runner.STAGE8_CANDIDATES["pit_adaptive_trend_stress"]

    differing = {key for key in baseline | candidate if baseline.get(key) != candidate.get(key)}
    assert differing == {"score_mode"}
    assert baseline["eligibility_mode"] == candidate["eligibility_mode"] == "pit"


def test_stage9_candidates_change_only_continuous_defensive_ranking():
    runner = _runner_module()
    baseline = runner.STAGE9_CANDIDATES["pit_continuous_baseline_control"]

    assert len(runner.STAGE9_CANDIDATES) == 5
    for name, candidate in runner.STAGE9_CANDIDATES.items():
        differing = {
            key for key in baseline | candidate if baseline.get(key) != candidate.get(key)
        }
        if name == "pit_continuous_baseline_control":
            assert not differing
        else:
            assert differing == {"score_mode"}
            assert candidate["score_mode"].startswith("continuous_stress_")
        assert candidate["eligibility_mode"] == "pit"
        assert candidate["_max_positions"] == 10
        assert candidate["_max_hold_days"] == 15
        assert candidate["_stop_loss"] == -0.06


def test_stage10_uses_sparse_trend_events_with_frozen_execution():
    runner = _runner_module()
    candidates = runner.STAGE10_CANDIDATES

    assert set(candidates) == {
        "pit_trend_baseline_control",
        "pit_sparse_breakout",
        "pit_sparse_breakout_quality",
        "pit_first_trend_pullback",
        "pit_first_trend_pullback_industry",
    }
    assert candidates["pit_sparse_breakout"]["family"] == "sparse_breakout"
    assert (
        candidates["pit_first_trend_pullback"]["family"]
        == "first_trend_pullback"
    )
    for candidate in candidates.values():
        assert candidate["eligibility_mode"] == "pit"
        assert candidate["_max_positions"] == 10
        assert candidate["_max_hold_days"] == 15
        assert candidate["_stop_loss"] == -0.06


def test_stage11_adds_only_horizon_cooldown_and_first_retest():
    runner = _runner_module()
    candidates = runner.STAGE11_CANDIDATES

    assert set(candidates) == {
        "pit_sparse_baseline_control",
        "pit_sparse_breakout_cooldown60",
        "pit_breakout_first_retest",
        "pit_breakout_first_retest_industry",
    }
    for name, candidate in candidates.items():
        if name != "pit_sparse_baseline_control":
            assert candidate["event_cooldown_days"] == 60
        assert candidate["eligibility_mode"] == "pit"
        assert candidate["_max_positions"] == 10
        assert candidate["_max_hold_days"] == 15
        assert candidate["_stop_loss"] == -0.06


def test_sparse_trend_cooldown_does_not_use_future_rows():
    original = _panel()
    cutoff = date(2020, 1, 1) + timedelta(days=99)
    changed = original.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("close") * 1.8)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("volume") * 3.0)
        .otherwise(pl.col("volume"))
        .alias("volume"),
    )
    strategy = _strategy_module().MATRIX_STRATEGY
    for family, score_mode in (
        ("sparse_breakout", "trend_breakout"),
        ("breakout_first_retest", "trend_pullback"),
    ):
        params = {
            **_rank_params(score_mode),
            "family": family,
            "event_cooldown_days": 60,
        }
        first = strategy.compute_signals(_market(original), params)
        second = strategy.compute_signals(_market(changed), params)

        np.testing.assert_array_equal(first.entry[:100], second.entry[:100])
        np.testing.assert_array_equal(first.exit[:100], second.exit[:100])
        np.testing.assert_allclose(first.score[:100], second.score[:100])


def test_stage12_replays_prior_risk_hypotheses_without_retuning():
    runner = _runner_module()
    candidates = runner.STAGE12_CANDIDATES

    assert set(candidates) == {
        "pit_risk_baseline_control",
        "pit_recovery_p15_control",
        "pit_recovery_p15_rsi55_control",
        "pit_recovery_p15_rsi55_wide_stop",
        "pit_recovery_p10_rsi55_wide_stop",
    }
    combined = candidates["pit_recovery_p15_rsi55_wide_stop"]
    assert combined["score_mode"] == "recovery"
    assert combined["exit_mode"] == "rsi_recovery"
    assert combined["_max_positions"] == 15
    assert combined["_max_hold_days"] == 15
    assert combined["_stop_loss"] == -0.08
    for candidate in candidates.values():
        assert candidate["eligibility_mode"] == "pit"


def test_independent_alpha_families_are_causal_and_executable():
    original = _panel().with_columns(pl.lit(1.0).alias("pit_eligible"))
    cutoff = date(2020, 1, 1) + timedelta(days=99)
    changed = original.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("close") * 1.8)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("volume") * 3.0)
        .otherwise(pl.col("volume"))
        .alias("volume"),
    )
    strategy = _independent_strategy_module().MATRIX_STRATEGY
    for family in (
        "trend",
        "industry_rotation",
        "sentiment_timing",
        "sentiment_secondary_ignition",
        "secondary_ignition",
        "accumulation_secondary_ignition",
        "accumulation_strong_market",
        "breadth_oversold_repair",
        "quality_compounder",
        "limit_event",
        "monthly_momentum",
        "regime_reversal_quality",
    ):
        params = {"family": family, "eligibility_mode": "pit"}
        first = strategy.compute_signals(_market(original), params)
        second = strategy.compute_signals(_market(changed), params)

        np.testing.assert_array_equal(first.entry[:100], second.entry[:100])
        np.testing.assert_array_equal(first.exit[:100], second.exit[:100])
        np.testing.assert_allclose(first.score[:100], second.score[:100])

    risk_params = {
        "family": "trend",
        "eligibility_mode": "pit",
        "risk_on_only": True,
    }
    first = strategy.compute_signals(_market(original), risk_params)
    second = strategy.compute_signals(_market(changed), risk_params)
    np.testing.assert_array_equal(first.entry[:100], second.entry[:100])
    np.testing.assert_array_equal(first.exit[:100], second.exit[:100])
    np.testing.assert_allclose(first.score[:100], second.score[:100])

    repaired_sentiment_params = {
        "family": "sentiment_timing",
        "eligibility_mode": "pit",
        "sentiment_confirm_days": 2,
        "sentiment_max_momentum_60d": 0.5,
        "sentiment_max_distance_ma20": 0.15,
        "sentiment_rsi_max": 70.0,
        "sentiment_stock_ma20_exit": True,
    }
    first = strategy.compute_signals(_market(original), repaired_sentiment_params)
    second = strategy.compute_signals(_market(changed), repaired_sentiment_params)
    np.testing.assert_array_equal(first.entry[:100], second.entry[:100])
    np.testing.assert_array_equal(first.exit[:100], second.exit[:100])
    np.testing.assert_allclose(first.score[:100], second.score[:100])


def test_monthly_momentum_only_enters_on_first_available_day_of_month():
    panel = _panel(days=320).with_columns(pl.lit(1.0).alias("pit_eligible"))
    market = _market(panel)
    signals = _independent_strategy_module().MATRIX_STRATEGY.compute_signals(
        market,
        {
            "family": "monthly_momentum",
            "eligibility_mode": "pit",
            "quality_gate": False,
        },
    )

    entry_rows = np.where(signals.entry.any(axis=1))[0]
    for row_id in entry_rows:
        current = date.fromisoformat(market.timestamp_labels[row_id][:10])
        previous = date.fromisoformat(market.timestamp_labels[row_id - 1][:10])
        assert (current.year, current.month) != (previous.year, previous.month)


def test_point_in_time_eligibility_fails_closed_before_entry():
    panel = _panel().with_columns(pl.lit(0.0).alias("pit_eligible"))
    params = {**_rank_params("baseline"), "eligibility_mode": "pit"}

    signals = _strategy_module().MATRIX_STRATEGY.compute_signals(_market(panel), params)

    assert signals.entry.sum() == 0
