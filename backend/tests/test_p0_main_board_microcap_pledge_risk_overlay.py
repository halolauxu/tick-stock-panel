from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "run_p0_main_board_microcap_pledge_risk_overlay",
    ROOT / "research" / "run_p0_main_board_microcap_pledge_risk_overlay.py",
)
assert SPEC and SPEC.loader
overlay = module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


def _trading_dates(count: int = 500) -> list[date]:
    dates = []
    current = date(2020, 1, 1)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def test_event_is_only_available_after_announcement_and_excludes_126_days() -> None:
    dates = _trading_dates()
    announcement = dates[10]
    events = pl.DataFrame(
        {"symbol": ["600001.SH"], "ann_date": [announcement], "pledge_ratio": [8.0]}
    )

    available = overlay.attach_available_dates(events, dates)
    exclusions = overlay.exclusion_calendar(available, dates)

    assert available["available_date"].to_list() == [dates[11]]
    assert exclusions.height == 126
    assert exclusions["date"].min() == dates[11]
    assert exclusions["date"].max() == dates[136]


def test_systemic_gate_uses_only_prior_52_week_counts() -> None:
    dates = _trading_dates()
    weekly = pl.DataFrame(
        {
            "date": dates[20::5][:54],
            "entry_date": dates[21::5][:54],
        }
    )
    event_rows = []
    for week_date in weekly["date"].to_list()[:53]:
        count = 1 if week_date != weekly["date"].to_list()[52] else 12
        index = dates.index(week_date)
        for number in range(count):
            event_rows.append(
                {
                    "symbol": f"{600000 + number:06d}.SH",
                    "ann_date": dates[index - 1],
                    "pledge_ratio": 6.0,
                    "available_date": week_date,
                }
            )
    clock = overlay.systemic_risk_clock(
        weekly, pl.DataFrame(event_rows), dates
    ).sort("date")

    assert clock["risk_off"].head(52).sum() == 0
    assert clock.row(52, named=True)["risk_off"] is True
    assert clock.row(52, named=True)["material_pledge_symbols_20d"] == 12


def test_development_gate_requires_every_year_positive() -> None:
    yearly = [
        {"year": year, "account_return": 0.1 if year != 2018 else -0.01}
        for year in range(2014, 2021)
    ]
    account = {
        "metrics": {
            "account_annualized": 0.40,
            "account_max_drawdown": -0.20,
            "yearly": yearly,
        },
        "execution": {
            "buy": {"execution_rate": 0.9},
            "sell": {"execution_rate": 0.9},
        },
        "integrity": {
            "ending_unresolved_positions": 0,
            "max_cash_reconciliation_error": 0.0,
        },
    }
    result = {
        "accounts": {str(int(overlay.PRIMARY_CAPITAL)): {"combined": account}}
    }

    decision = overlay.evaluate("development", result)

    assert decision["passed"] is False
    assert decision["checks"]["all_7_years_positive"] is False
