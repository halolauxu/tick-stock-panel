from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_cb_stock_lead_lag_discovery.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_cb_stock_lead_lag", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_universe_uses_prior_day_and_preserves_stock_mapping() -> None:
    days = [date(2026, 8, 3), date(2026, 8, 4)]
    basic = pl.DataFrame(
        {
            "symbol": ["110001.SH", "132001.SH"],
            "stock_symbol": ["600001.SH", "600002.SH"],
            "cb_type": ["CB", "EB"],
            "list_date": [date(2025, 1, 1), date(2025, 1, 1)],
        }
    )
    daily = pl.DataFrame(
        {
            "symbol": ["110001.SH", "132001.SH", "110001.SH", "132001.SH"],
            "date": [days[0], days[0], days[1], days[1]],
            "close": [100.0, 100.0, 139.0, 139.0],
            "amount_cny": [150_000_000.0] * 4,
            "cb_over_rate": [20.0] * 4,
        }
    )

    result = study.build_causal_universe(basic, daily)

    assert result.height == 1
    assert result["symbol"][0] == "110001.SH"
    assert result["stock_symbol"][0] == "600001.SH"
    assert result["previous_close"][0] == 100.0


def _minute_rows(symbol: str, *, stock: bool) -> list[dict]:
    rows = []
    start = datetime(2026, 8, 4, 9, 30)
    for offset in range(32):
        stamp = start + timedelta(minutes=offset)
        close = (10.0 + offset * 0.03) if stock else (100.0 + offset * 0.01)
        row = {
            "symbol": symbol,
            "datetime": stamp,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
        if stock:
            row.update({"volume": 10_000.0, "amount": 10_000_000.0})
        else:
            row.update({"volume_hands": 100.0, "amount_cny": 3_000_000.0})
        rows.append(row)
    return rows


def test_observation_requires_stock_lead_and_uses_future_cb_only_for_return() -> None:
    day = date(2026, 8, 4)
    universe = pl.DataFrame(
        {
            "symbol": ["110001.SH"],
            "stock_symbol": ["600001.SH"],
            "date": [day],
            "previous_date": [date(2026, 8, 3)],
            "previous_close": [100.0],
            "previous_amount_cny": [150_000_000.0],
            "previous_cb_over_rate": [20.0],
        }
    )
    cb = study.prepare_cb_minutes(
        pl.DataFrame(_minute_rows("110001.SH", stock=False)), universe
    )
    stock = study.prepare_stock_minutes(
        pl.DataFrame(_minute_rows("600001.SH", stock=True))
    )

    result = study.build_observations(cb, stock)

    assert result.height == 1
    assert result["datetime"][0] == datetime(2026, 8, 4, 9, 45)
    assert result["stock_past_5m"][0] >= study.STOCK_MOVE_FLOOR
    assert result["cb_past_5m"][0] <= study.CB_MOVE_CEILING
    assert result["capacity_cny"][0] == 30_000.0


def test_evaluation_requires_both_time_halves_and_capacity() -> None:
    market_dates = [date(2026, 8, 1) + timedelta(days=i) for i in range(16)]
    rows = []
    for current in market_dates:
        for event in range(13):
            rows.append(
                {
                    "date": current,
                    "datetime": datetime.combine(current, datetime.min.time())
                    + timedelta(minutes=event),
                    "symbol": f"110{event:03d}.SH",
                    "stock_symbol": f"600{event:03d}.SH",
                    "stock_past_5m": 0.02,
                    "cb_past_5m": 0.0,
                    "expected_pass_through": 0.018,
                    "lag": 0.018,
                    "gross_return": 0.005,
                    "net_return": 0.0036,
                    "benchmark_gross_return": 0.001,
                    "excess_return": 0.004,
                    "capacity_cny": 120_000.0,
                    "slot": event + 1,
                }
            )
    observations = pl.DataFrame(rows)

    result = study.evaluate(observations, market_dates)

    assert result["promotion_passed"] is True
    assert all(result["checks"]["discovery"].values())
    assert all(result["checks"]["confirmation"].values())
