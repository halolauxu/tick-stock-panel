from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "run_p0_main_board_microcap_unlock_supply_risk",
    ROOT / "research" / "run_p0_main_board_microcap_unlock_supply_risk.py",
)
assert SPEC and SPEC.loader
unlock = module_from_spec(SPEC)
SPEC.loader.exec_module(unlock)


def _trading_dates(count: int = 500) -> list[date]:
    dates = []
    current = date(2014, 1, 1)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def test_only_details_announced_by_signal_date_count_toward_exclusion() -> None:
    dates = _trading_dates(40)
    weekly = pl.DataFrame({"date": [dates[5], dates[7]], "entry_date": [dates[6], dates[8]]})
    details = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH"],
            "ann_date": [dates[4], dates[6]],
            "float_date": [dates[10], dates[10]],
            "float_shares": [100.0, 100.0],
            "float_ratio": [3.0, 3.0],
        }
    )

    exclusions, clock = unlock.build_weekly_unlock_risk(weekly, details, dates)

    assert exclusions.to_dicts() == [{"symbol": "600001.SH", "date": dates[7]}]
    assert clock["upcoming_material_unlock_symbols_20d"].to_list() == [0, 1]


def test_systemic_gate_uses_only_previous_52_weeks() -> None:
    dates = _trading_dates()
    weekly_dates = dates[5::5][:54]
    weekly = pl.DataFrame(
        {"date": weekly_dates, "entry_date": [dates[dates.index(day) + 1] for day in weekly_dates]}
    )
    rows = []
    for week_index, signal_date in enumerate(weekly_dates[:53]):
        count = 12 if week_index == 52 else 1
        signal_index = dates.index(signal_date)
        for number in range(count):
            rows.append(
                {
                    "symbol": f"{600000 + number:06d}.SH",
                    "ann_date": dates[signal_index - 1],
                    "float_date": dates[signal_index + 2],
                    "float_shares": 100.0,
                    "float_ratio": 6.0,
                }
            )
    details = pl.DataFrame(rows)

    _, clock = unlock.build_weekly_unlock_risk(weekly, details, dates)

    assert clock["risk_off"].head(52).sum() == 0
    assert clock.row(52, named=True)["risk_off"] is True
    assert clock.row(52, named=True)["upcoming_material_unlock_symbols_20d"] == 12
