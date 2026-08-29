# ruff: noqa: RUF001
from __future__ import annotations

from pathlib import Path

import polars as pl

from app.alpha_mining.contracts import TrainOnlyContext, TrialBudget
from app.alpha_mining.engines.cross_sectional import CrossSectionalEngine
from app.alpha_mining.hypotheses import AlphaHypothesisStore, system_hypotheses


def test_system_supplies_distinct_falsifiable_ashare_hypotheses() -> None:
    rows = system_hypotheses()

    assert len(rows) >= 3
    assert {row["source_kind"] for row in rows} == {"prior"}
    assert len({row["mechanism"] for row in rows}) == len(rows)
    assert all(row["thesis"] and row["falsification"] for row in rows)
    assert all(row["test_spec"]["factor_names"] for row in rows)
    assert all(
        set(row["test_spec"]["factor_names"])
        == set(row["test_spec"]["expected_directions"])
        for row in rows
    )
    assert all("n_day_low_reversal" not in str(row) for row in rows)


def test_manual_hypothesis_is_persisted_and_immutable(tmp_path: Path) -> None:
    store = AlphaHypothesisStore(tmp_path)
    created = store.create({
        "source_kind": "manual",
        "title": "缩量止跌后的修复",
        "thesis": "短期下跌后量能收缩且收盘位置改善，卖压可能已经衰竭。",
        "mechanism": "A股个人投资者止损集中释放后，边际卖盘下降。",
        "prediction_object": "forward_net_return",
        "asset_type": "stock",
        "forward_horizon": 5,
        "information_domains": ["price_volume", "liquidity"],
        "test_spec": {
            "engine_ids": ["cross_sectional_rank"],
            "factor_names": ["momentum_5d", "vol_ratio_5d", "close_position"],
            "expected_directions": {
                "momentum_5d": -1,
                "vol_ratio_5d": -1,
                "close_position": 1,
            },
            "weights": {
                "momentum_5d": 0.4,
                "vol_ratio_5d": 0.3,
                "close_position": 0.3,
            },
        },
        "falsification": ["训练窗组合IC未达到预注册下限", "独立样本外净收益不为正"],
        "data_requirements": ["daily_enriched", "historical_universe"],
    })

    assert created["hypothesis_id"].startswith("ah-")
    assert store.get(created["hypothesis_id"]) == created
    assert store.list_saved()[0]["title"] == "缩量止跌后的修复"

    try:
        store.create({**created, "title": "覆盖旧假设"})
    except ValueError as exc:
        assert "已存在" in str(exc)
    else:
        raise AssertionError("immutable hypothesis was overwritten")


def test_preregistered_direction_is_tested_instead_of_inferred_afterwards() -> None:
    frame = pl.DataFrame({
        "date": ["2026-01-01"] * 20 + ["2026-01-02"] * 20,
        "symbol": [f"S{index:02d}" for index in range(20)] * 2,
        "lottery": list(range(20)) * 2,
        "target": [float(19 - index) for index in range(20)] * 2,
    })
    contract = {
        "title": "彩票偏好反转",
        "thesis": "高彩票特征预期未来收益更低",
        "test_spec": {
            "factor_names": ["lottery"],
            "expected_directions": {"lottery": -1},
            "weights": {"lottery": 1.0},
            "parameters": {"entry_score": 75, "exit_score": 40, "top_rank": 20},
        },
    }
    context = TrainOnlyContext(
        frame=frame,
        date_labels=("2026-01-01", "2026-01-02"),
        feature_names=("lottery",),
        target_column="target",
        asset_type="stock",
        metadata={"hypothesis_contract": contract},
    )

    candidates = CrossSectionalEngine().discover(
        context,
        TrialBudget(max_candidates=2, max_trials=8, min_cross_section=20, min_dates=2),
    )

    assert len(candidates) == 1
    assert candidates[0].directions == (-1,)
    assert candidates[0].features == ("lottery",)
    assert candidates[0].thesis == contract["thesis"]
