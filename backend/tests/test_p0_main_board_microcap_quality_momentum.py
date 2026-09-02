from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "run_p0_main_board_microcap_quality_momentum",
    ROOT / "research" / "run_p0_main_board_microcap_quality_momentum.py",
)
assert SPEC and SPEC.loader
quality = module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def _panel(symbol: str, growth: float) -> list[dict[str, object]]:
    start = date(2019, 1, 1)
    return [
        {
            "symbol": symbol,
            "date": start + timedelta(days=index),
            "close": 10.0 * (1.0 + growth) ** index,
        }
        for index in range(61)
    ]


def test_composite_ranks_stronger_quality_and_momentum_first() -> None:
    signal_date = date(2019, 3, 2)
    candidates = pl.DataFrame(
        {
            "date": [signal_date, signal_date],
            "entry_date": [date(2019, 3, 4), date(2019, 3, 4)],
            "symbol": ["600001.SH", "600002.SH"],
            "market_cap": [1.0, 2.0],
            "signal_amount": [1e8, 1e8],
            "cap_rank": [1, 2],
        }
    )
    panel = pl.DataFrame(_panel("600001.SH", 0.001) + _panel("600002.SH", 0.01))
    snapshots = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "financial_available_date": [date(2019, 2, 1), date(2019, 2, 1)],
            "total_assets": [100.0, 100.0],
            "total_equity": [50.0, 60.0],
            "net_income_attributable": [2.0, 10.0],
            "net_operating_cash_flow": [3.0, 12.0],
            "debt_ratio": [0.7, 0.2],
        }
    )

    ranked = quality.attach_quality_momentum(candidates, panel, snapshots)

    assert ranked.sort("cap_rank")["symbol"].to_list() == ["600002.SH", "600001.SH"]


def test_future_financial_snapshot_is_not_joined() -> None:
    signal_date = date(2019, 3, 2)
    candidates = pl.DataFrame(
        {
            "date": [signal_date],
            "entry_date": [date(2019, 3, 4)],
            "symbol": ["600001.SH"],
            "market_cap": [1.0],
            "signal_amount": [1e8],
            "cap_rank": [1],
        }
    )
    panel = pl.DataFrame(_panel("600001.SH", 0.001))
    snapshots = pl.DataFrame(
        {
            "symbol": ["600001.SH"],
            "financial_available_date": [date(2019, 3, 3)],
            "total_assets": [100.0],
            "total_equity": [50.0],
            "net_income_attributable": [2.0],
            "net_operating_cash_flow": [3.0],
            "debt_ratio": [0.5],
        }
    )

    ranked = quality.attach_quality_momentum(candidates, panel, snapshots)

    assert ranked.is_empty()


def test_stage_trading_dates_excludes_out_of_stage_history() -> None:
    dates = [date(2013, 12, 31), date(2014, 1, 2), date(2020, 12, 31), date(2021, 1, 4)]

    assert quality.stage_trading_dates(dates, "development") == [
        date(2014, 1, 2),
        date(2020, 12, 31),
    ]
