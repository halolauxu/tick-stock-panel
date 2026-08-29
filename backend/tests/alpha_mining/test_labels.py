from __future__ import annotations

# Requirements: AM-S4-001 through AM-S4-004.
from datetime import date

import polars as pl
import pytest

from app.alpha_mining.labels import SUPPORTED_HORIZONS, attach_alpha_labels


def test_alpha_labels_cover_all_horizons_costs_paths_and_untradable_risk() -> None:
    days = [date(2026, 1, value) for value in range(1, 8)]
    panel = pl.DataFrame({
        "symbol": ["000001.SZ"] * len(days),
        "date": days,
        "open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
        "high": [10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
        "low": [9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4],
        "close": [10.0, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
        "volume": [100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 100.0],
        "signal_limit_up": [False, False, True, False, False, False, False],
    })
    result = attach_alpha_labels(panel, days, horizons=(1, 3), commission_pct=0.001, stamp_tax_pct=0.001, slippage_bps=10)
    first = result.row(0, named=True)
    assert first["target_gross_return_1d"] == pytest.approx(0.02)
    assert first["target_return_1d"] == pytest.approx(0.015)
    assert first["target_mfe_3d"] == pytest.approx(0.05)
    assert first["target_mae_3d"] == pytest.approx(-0.01)
    second = result.row(1, named=True)
    assert second["target_untradable_1d"] is True
    assert second["target_return_1d"] is None
    assert SUPPORTED_HORIZONS == (1, 3, 5, 10, 20, 60)


def test_missing_future_symbol_row_is_delist_or_suspension_risk() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    panel = pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "000001.SZ"],
        "date": [days[0], days[0], days[1]],
        "open": [10.0, 20.0, 11.0],
        "high": [10.0, 20.0, 11.0],
        "low": [10.0, 20.0, 11.0],
        "close": [10.0, 20.0, 11.0],
    })
    result = attach_alpha_labels(panel, days, horizons=(1,))
    missing = result.filter((pl.col("symbol") == "000002.SZ") & (pl.col("date") == days[0])).row(0, named=True)
    assert missing["target_untradable_1d"] is True
    assert missing["target_return_1d"] is None
