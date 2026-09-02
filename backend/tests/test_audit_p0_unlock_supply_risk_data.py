from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "audit_p0_unlock_supply_risk_data",
    ROOT / "research" / "audit_p0_unlock_supply_risk_data.py",
)
assert SPEC and SPEC.loader
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_material_unlocks_are_aggregated_and_late_announcements_are_excluded() -> None:
    details = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH", "600002.SH"],
            "ann_date": [date(2019, 1, 1), date(2019, 1, 2), date(2019, 2, 2)],
            "float_date": [date(2019, 2, 1), date(2019, 2, 1), date(2019, 2, 1)],
            "float_shares": [100.0, 200.0, 500.0],
            "float_ratio": [2.0, 4.0, 10.0],
        }
    )
    universe = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "market": ["主板", "主板"],
            "list_date": [date(2010, 1, 1), date(2010, 1, 1)],
            "delist_date": [None, None],
        }
    ).with_columns(pl.col("delist_date").cast(pl.Date))

    result = audit.aggregate_material_events(details, universe)

    assert result.height == 1
    assert result.row(0, named=True)["symbol"] == "600001.SH"
    assert result.row(0, named=True)["float_ratio_pct"] == 6.0
    assert result.row(0, named=True)["last_ann_date"] == date(2019, 1, 2)
