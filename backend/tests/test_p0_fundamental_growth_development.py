from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_fundamental_growth_development.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_fundamental_growth", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_metrics_are_not_available_on_announcement_day() -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "date": [date(2020, 4, 30), date(2020, 5, 6)],
        }
    )
    metrics = pl.DataFrame(
        {
            "symbol": ["A.SZ"],
            "report_period_end": [date(2020, 3, 31)],
            "report_announce_date": [date(2020, 4, 30)],
            "roe": [10.0],
        }
    )

    result = study.attach_latest_metrics(panel, metrics)

    assert result["report_age_days"].to_list() == [None, 6]
    assert result["roe"].to_list() == [10.0, 10.0]


def test_load_metrics_keeps_latest_period_when_same_announcement(tmp_path: Path) -> None:
    target = tmp_path / "financials" / "metrics"
    target.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["A.SZ", "A.SZ"],
            "period_end": ["2019-12-31", "2020-03-31"],
            "announce_date": ["2020-04-30", "2020-04-30"],
            "roe": [8.0, 10.0],
            "net_margin": [6.0, 7.0],
            "debt_to_asset_ratio": [50.0, 51.0],
            "revenue_yoy": [20.0, 30.0],
            "net_income_yoy": [40.0, 50.0],
            "operating_cash_to_revenue": [8.0, 9.0],
        }
    ).write_parquet(target / "part.parquet")

    result = study.load_metrics(tmp_path)

    assert result.height == 1
    assert result["report_period_end"][0] == date(2020, 3, 31)
    assert result["roe"][0] == 10.0
