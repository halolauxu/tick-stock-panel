from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "research"
        / "run_p0_index_inclusion_microcap_overlay.py"
    )
    spec = importlib.util.spec_from_file_location("p0_index_inclusion_overlay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _additions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cycle_month": [date(2022, 6, 1)] * 3,
            "announcement_date": [date(2022, 5, 27)] * 3,
            "effective_date": [date(2022, 6, 10)] * 3,
            "symbol": ["600001.SH", "300001.SZ", "688001.SH"],
        }
    )


def _quotes(*, entry_limit_up: bool = False, first_exit_limit_down: bool = False) -> pl.DataFrame:
    rows = []
    for day, raw_open, limit_up, limit_down in (
        (date(2022, 5, 30), 10.0, 10.0 if entry_limit_up else 11.0, 9.0),
        (date(2022, 6, 13), 10.5, 11.5, 10.5 if first_exit_limit_down else 9.5),
        (date(2022, 6, 14), 10.6, 11.6, 9.6),
    ):
        rows.append(
            {
                "symbol": "600001.SH",
                "date": day,
                "open": raw_open,
                "raw_open": raw_open,
                "close": raw_open,
                "raw_close": raw_open,
                "volume": 1_000_000.0,
                "amount": 30_000_000.0,
                "limit_up_price": limit_up,
                "limit_down_price": limit_down,
                "is_excluded_name": False,
            }
        )
    return pl.DataFrame(rows)


def test_build_cycles_excludes_chinext_and_star_and_uses_next_session() -> None:
    days = [date(2022, 5, 27), date(2022, 5, 30), date(2022, 6, 10), date(2022, 6, 13)]

    result = study.build_cycles(_additions(), days, start=date(2022, 1, 1), end=date(2022, 12, 31))

    assert len(result) == 1
    assert result[0]["symbols"] == ["600001.SH"]
    assert result[0]["entry_date"] == date(2022, 5, 30)
    assert result[0]["planned_exit_date"] == date(2022, 6, 13)


def test_simulation_rejects_limit_up_without_manufacturing_fill() -> None:
    days = [date(2022, 5, 30), date(2022, 6, 13), date(2022, 6, 14)]
    cycles = study.build_cycles(_additions(), days, start=date(2022, 1, 1), end=date(2022, 12, 31))

    result = study.simulate(cycles, _quotes(entry_limit_up=True), days, initial_cash=200_000.0)

    assert result["ending_positions"] == {}
    assert result["ending_cash"] == 200_000.0
    assert result["orders"][0]["reason"] == "limit_up"


def test_simulation_retries_limit_down_exit_and_reconciles_cash() -> None:
    days = [date(2022, 5, 30), date(2022, 6, 13), date(2022, 6, 14)]
    cycles = study.build_cycles(_additions(), days, start=date(2022, 1, 1), end=date(2022, 12, 31))

    result = study.simulate(
        cycles,
        _quotes(first_exit_limit_down=True),
        days,
        initial_cash=200_000.0,
    )
    sells = [row for row in result["orders"] if row["side"] == "SELL"]

    assert [row["status"] for row in sells] == ["REJECTED", "FILLED"]
    assert sells[0]["reason"] == "limit_down"
    assert result["ending_positions"] == {}
    assert result["ending_cash"] > 200_000.0
    assert result["max_cash_reconciliation_error"] == 0.0
    assert study.execution_summary(result)["sell"]["execution_rate"] == 1.0
