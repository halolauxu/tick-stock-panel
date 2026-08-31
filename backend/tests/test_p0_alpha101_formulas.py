from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "alpha101_formulas.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("alpha101_formulas", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


alpha = _load_module()


def _inputs(rows: int = 320, assets: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260831)
    time = np.arange(rows, dtype=np.float32)[:, None]
    asset = np.arange(assets, dtype=np.float32)[None, :]
    close = (
        10.0
        + time * 0.01
        + asset * 0.2
        + np.sin(time * (0.07 + asset * 0.004) + asset) * 0.1
        + rng.normal(0, 0.08, size=(rows, assets))
    )
    open_ = close * (
        1.0 + np.cos(time * (0.09 + asset * 0.003) + asset) * 0.002
        + rng.normal(0, 0.001, size=(rows, assets))
    )
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = (
        100_000.0
        + time * 100
        + asset * 10_000
        + np.sin(time * (0.05 + asset * 0.007)) * 5_000
        + rng.normal(0, 8_000, size=(rows, assets))
    )
    amount = volume * 100.0 * ((open_ + high + low + close) / 4.0)
    return {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
        "amount": amount.astype(np.float32),
    }


def test_frozen_formula_set_is_complete_and_each_formula_computes() -> None:
    inputs = _inputs()
    context = alpha.Alpha101Context.from_arrays(**inputs)

    assert alpha.ALPHA_IDS == (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 22, 23, 25, 33, 34, 41, 52, 53, 54, 57, 101,
    )
    for alpha_id in alpha.ALPHA_IDS:
        values = alpha.compute_alpha101(context, alpha_id)
        assert values.shape == inputs["close"].shape
        assert values.dtype == np.float32
        assert np.isfinite(values).any(), alpha_id


def test_cross_sectional_rank_uses_average_percentile_and_preserves_nan() -> None:
    values = np.array([[3.0, 1.0, 1.0, np.nan]], dtype=np.float32)

    ranked = alpha.cross_sectional_rank(values)

    np.testing.assert_allclose(ranked[0, :3], [1.0, 0.25, 0.25])
    assert np.isnan(ranked[0, 3])


def test_contiguous_windows_do_not_jump_across_missing_rows() -> None:
    values = np.arange(1, 13, dtype=np.float32)[:, None]
    values[6, 0] = np.nan

    rolled = alpha.rolling_sum(values, 3)

    assert rolled[5, 0] == 15.0
    assert np.isnan(rolled[6:9, 0]).all()
    assert rolled[9, 0] == 27.0


def test_alpha101_matches_published_intraday_position_formula() -> None:
    inputs = _inputs(rows=3, assets=2)
    context = alpha.Alpha101Context.from_arrays(**inputs)

    actual = alpha.compute_alpha101(context, 101)
    expected = (inputs["close"] - inputs["open"]) / (
        inputs["high"] - inputs["low"] + 0.001
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_conditional_formulas_remain_null_until_their_windows_are_complete() -> None:
    context = alpha.Alpha101Context.from_arrays(**_inputs(rows=40, assets=3))

    assert np.isnan(alpha.compute_alpha101(context, 9)[:5]).all()
    assert np.isnan(alpha.compute_alpha101(context, 10)[:4]).all()
    assert np.isnan(alpha.compute_alpha101(context, 23)[:19]).all()
