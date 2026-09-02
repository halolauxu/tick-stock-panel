from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "audit_p0_pledge_risk_data",
    ROOT / "research" / "audit_p0_pledge_risk_data.py",
)
assert SPEC and SPEC.loader
audit_module = module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def test_audit_qualifies_complete_point_in_time_fixture(tmp_path: Path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    pl.DataFrame(
        {
            "symbol": ["600001.SH"],
            "name": ["样本"],
            "market": ["主板"],
            "exchange": ["SSE"],
            "list_status": ["L"],
            "list_date": [date(1990, 1, 1)],
            "delist_date": [None],
        }
    ).write_parquet(research / "historical_stock_universe.parquet")

    for year in range(2014, 2027):
        partition = tmp_path / "event_data" / "pledge_detail" / f"year={year}"
        partition.mkdir(parents=True)
        pl.DataFrame(
            {
                "symbol": ["600001.SH"] * 50,
                "ann_date": [date(year, 1, day % 28 + 1) for day in range(50)],
                "is_release": ["0"] * 50,
                "pledge_ratio": [5.0] * 50,
            }
        ).write_parquet(partition / "part.parquet")

    result = audit_module.audit(tmp_path, research / "audit.json")

    assert result["status"] == "DATA_QUALIFIED"
    assert result["period"]["future_returns_read"] is False
    assert result["counts"]["material_main_board_events"] == 650
