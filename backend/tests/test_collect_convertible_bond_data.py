from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_convertible_bond_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_cb_data", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def test_daily_amount_is_converted_from_ten_thousand_cny() -> None:
    frame = collector.normalize_daily(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260828",
                "pre_close": 100.0,
                "open": 101.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.5,
                "pct_chg": 1.5,
                "vol": 123.0,
                "amount": 456.7,
                "bond_value": 95.0,
                "bond_over_rate": 6.8,
                "cb_value": 90.0,
                "cb_over_rate": 12.8,
            }
        ],
        date(2026, 8, 28),
    )

    assert frame["amount_cny"][0] == 4_567_000.0
    assert frame["volume_hands"][0] == 123.0
    assert frame["date"][0] == date(2026, 8, 28)


def test_minute_volume_is_already_hands_and_amount_is_cny() -> None:
    frame = collector.normalize_minute(
        [
            {
                "ts_code": "110001.SH",
                "trade_time": "2026-08-28 09:30:00",
                "open": 101.0,
                "high": 101.0,
                "low": 100.9,
                "close": 100.9,
                "vol": 500.0,
                "amount": 5_045_000.0,
            }
        ],
        date(2026, 8, 1),
        date(2026, 8, 28),
    )

    assert frame["volume_hands"][0] == 500.0
    assert frame["amount_cny"][0] == 5_045_000.0


def test_shenzhen_minute_volume_is_bonds_and_converts_to_hands() -> None:
    frame = collector.normalize_minute(
        [
            {
                "ts_code": "123001.SZ",
                "trade_time": "2026-08-28 09:30:00",
                "open": 101.0,
                "high": 101.0,
                "low": 100.9,
                "close": 100.9,
                "vol": 500.0,
                "amount": 504_500.0,
            }
        ],
        date(2026, 8, 1),
        date(2026, 8, 28),
    )

    assert frame["volume_hands"][0] == 50.0
    assert frame["amount_cny"][0] == 504_500.0


def test_date_partitions_follow_local_trading_calendar(tmp_path: Path) -> None:
    root = tmp_path / "kline_daily_enriched"
    for value in ("2026-08-03", "2026-08-04", "invalid"):
        path = root / f"date={value}" / "part.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame({"x": [1]}).write_parquet(path)

    dates = collector._date_partitions(
        tmp_path, date(2026, 8, 1), date(2026, 8, 31)
    )

    assert dates == [date(2026, 8, 3), date(2026, 8, 4)]
