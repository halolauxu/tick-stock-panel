from __future__ import annotations

import importlib.util
import stat
from datetime import date
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_fund_ownership_breadth.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "collect_fund_ownership_breadth", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def _holding(code: str = "000001", coverage: int = 20) -> dict:
    return {
        "SECCODE": code,
        "SECNAME": "平安银行",
        "ENDDATE": "2020-06-30",
        "F001N": coverage,
        "F002N": 1_000_000,
        "F003N": 4_000,
    }


def _market_statistics() -> list[dict]:
    return [{"ENDDATE": "2020-06-30", "F001N": 2_000}]


def test_availability_uses_conservative_disclosure_deadline() -> None:
    assert collector.available_after(2020, 1) == date(2020, 4, 30)
    assert collector.available_after(2020, 2) == date(2020, 8, 31)
    assert collector.available_after(2020, 3) == date(2020, 10, 31)
    assert collector.available_after(2020, 4) == date(2021, 3, 31)


def test_normalize_converts_units_and_market_share() -> None:
    frame = collector.normalize([_holding()], _market_statistics(), 2020, 2)

    assert frame.height == 1
    assert frame["symbol"][0] == "000001.SZ"
    assert frame["coverage_share"][0] == pytest.approx(0.01)
    assert frame["total_market_value_cny"][0] == pytest.approx(40_000_000)
    assert frame["average_market_value_per_fund_cny"][0] == pytest.approx(
        2_000_000
    )


def test_collect_quarter_persists_metadata_atomically(tmp_path) -> None:
    result = collector.collect_quarter(
        lambda _period: [_holding()],
        _market_statistics,
        tmp_path,
        2020,
        2,
    )
    path = Path(result["path"])

    assert result["events"] == result["symbols"] == 1
    assert collector.pl.read_parquet(path).schema == collector.EVENT_SCHEMA
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_period_is_bounded_to_observed_history() -> None:
    collector.validate_period(2017, 1)
    collector.validate_period(2026, 2)
    with pytest.raises(ValueError, match=r"2017Q1\.\.2026Q2"):
        collector.validate_period(2016, 4)
