from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "audit_p0_margin_deleveraging_risk_data",
    ROOT / "research" / "audit_p0_margin_deleveraging_risk_data.py",
)
assert SPEC and SPEC.loader
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_comparable_margin_requires_adjacent_market_dates() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH", "600001.SH", "600002.SH"],
            "trade_date": [
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 1, 7),
                date(2020, 1, 6),
            ],
            "rzye": [100.0, 90.0, 80.0, 50.0],
        }
    )

    result = audit.comparable_margin(frame)

    assert result.select("symbol", "trade_date").to_dicts() == [
        {"symbol": "600001.SH", "trade_date": date(2020, 1, 3)}
    ]
    assert result["balance_change"].to_list() == pytest.approx([-0.1])
