from __future__ import annotations

import importlib.util
from datetime import date, datetime
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "audit_cb_limit_spillover_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_cb_limit_spillover_data", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_stock_limit_reference_excludes_point_in_time_st_name() -> None:
    daily = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "date": [
                date(2026, 8, 3),
                date(2026, 8, 4),
                date(2026, 8, 3),
                date(2026, 8, 4),
            ],
            "close": [10.0, 11.0, 10.0, 10.5],
        }
    )
    names = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "name": ["正常股份", "*ST风险"],
            "start_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "end_date": [None, None],
        },
        schema_overrides={"end_date": pl.Date},
    )

    result = study.build_stock_limit_reference(daily, names)

    assert result.get_column("stock_symbol").to_list() == ["000001.SZ"]
    assert result.get_column("limit_up_price").to_list() == [11.0]


def test_trigger_keeps_first_limit_close_per_bond_day() -> None:
    universe = pl.DataFrame(
        {
            "symbol": ["123001.SZ"],
            "stock_symbol": ["000001.SZ"],
            "date": [date(2026, 8, 4)],
        }
    )
    reference = pl.DataFrame(
        {
            "stock_symbol": ["000001.SZ"],
            "date": [date(2026, 8, 4)],
            "name": ["正常股份"],
            "previous_close": [10.0],
            "limit_up_price": [11.0],
        }
    )
    minutes = pl.DataFrame(
        {
            "symbol": ["000001.SZ"] * 3,
            "datetime": [
                datetime(2026, 8, 4, 10, 0),
                datetime(2026, 8, 4, 10, 1),
                datetime(2026, 8, 4, 10, 2),
            ],
            "close": [10.99, 11.0, 11.0],
            "amount": [1_000_000.0] * 3,
        }
    )

    result = study.build_trigger_events(universe, reference, minutes)

    assert result.height == 1
    assert result.item(0, "datetime") == datetime(2026, 8, 4, 10, 1)


def test_execution_bar_audit_requires_exact_minutes_and_positive_amount() -> None:
    event_time = datetime(2026, 8, 4, 10, 0)
    events = pl.DataFrame(
        {
            "date": [date(2026, 8, 4)],
            "datetime": [event_time],
            "symbol": ["123001.SZ"],
            "stock_symbol": ["000001.SZ"],
            "previous_close": [10.0],
            "limit_up_price": [11.0],
            "trigger_clock": [event_time.time()],
        }
    )
    minutes = pl.DataFrame(
        {
            "symbol": ["123001.SZ"] * 16,
            "datetime": [
                datetime(2026, 8, 4, 10, minute) for minute in range(16)
            ],
            "open": [120.0] * 16,
            "high": [120.0] * 16,
            "low": [120.0] * 16,
            "close": [120.0] * 16,
            "volume_hands": [100.0] * 16,
            "amount_cny": [1_000_000.0] * 16,
        }
    )

    result = study.attach_execution_bar_audit(events, minutes)

    assert result.item(0, "execution_bars_usable") is True


def test_compatible_partition_scan_ignores_new_runtime_columns(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pl.DataFrame(
        {"symbol": ["000001.SZ"], "close": [10.0]}
    ).write_parquet(first)
    pl.DataFrame(
        {"symbol": ["000001.SZ"], "close": [10.1], "quote_ts": [123]}
    ).write_parquet(second)

    result = study.scan_compatible_partitions(
        [first, second], ["symbol", "close"]
    ).collect()

    assert result.columns == ["symbol", "close"]
    assert result.height == 2
