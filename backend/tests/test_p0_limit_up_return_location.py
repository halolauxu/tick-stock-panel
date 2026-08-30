from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_limit_up_return_location.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_limit_up_location", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p0a2 = _load_module()


def _outcome() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "name": ["浦发银行"],
            "signal_date": [date(2026, 6, 1)],
            "entry_date": [date(2026, 6, 2)],
            "exit_date_h1": [date(2026, 6, 3)],
            "event_type": ["first_board"],
            "state": ["ferment"],
            "close": [10.0],
            "entry_price": [11.0],
            "entry_raw_price": [11.0],
            "entry_raw_close": [10.8],
            "entry_limit_up_price": [11.0],
            "signal_day_amount": [20_000_000.0],
            "entry_day_amount": [20_000_000.0],
            "exit_price_h1": [11.5],
            "net_return_h1": [0.04],
        }
    )


def _minutes(*, last_price: float = 10.9) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "date": [date(2026, 6, 2)],
            "minute_open": [11.0],
            "minute_last_5m": [last_price],
            "amount_5m": [3_000_000.0],
            "volume_5m": [2_750.0],
            "minute_bars_5m": [5],
            "vwap_5m": [10.8],
        }
    )


def test_delayed_entry_can_trade_after_open_limit_unseals() -> None:
    frame = p0a2.attach_return_components(_outcome(), _minutes())

    assert frame["delayed_entry_status"].item() == "ok"
    assert frame["delayed_tradable"].item() is True
    assert frame["signal_close_to_open"].item() == pytest.approx(0.10)
    assert frame["open_to_5m_vwap"].item() == pytest.approx(10.8 / 11.0 - 1)
    assert frame["vwap_5m_to_t1_sell_open_net"].item() is not None


def test_delayed_entry_does_not_manufacture_fill_while_still_limit_up() -> None:
    frame = p0a2.attach_return_components(
        _outcome(),
        _minutes(last_price=11.0),
    )

    assert frame["delayed_entry_status"].item() == "still_limit_up_at_09_34"
    assert frame["delayed_tradable"].item() is False
    assert frame["vwap_5m_to_t1_sell_open_net"].item() is None


def test_candidate_requires_both_time_slices() -> None:
    summary = pl.DataFrame(
        {
            "location_period": ["location", "confirmation"],
            "event_type": ["first_board", "first_board"],
            "event_count": [100, 100],
            "tradable_count": [90, 90],
            "tradable_rate": [0.9, 0.9],
            "cluster_day_count": [100, 40],
            "cluster_mean_net": [0.01, -0.01],
            "cluster_std": [0.01, 0.01],
            "cluster_win_rate": [0.6, 0.4],
            "cluster_t": [2.0, -2.0],
            "positive_month_count": [5, 2],
            "top_month_positive_share": [0.3, 0.4],
        }
    )

    candidate = p0a2.evaluate_candidates(summary)[0]

    assert candidate["passed"] is False
