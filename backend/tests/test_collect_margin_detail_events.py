from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_margin_detail_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_margin", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def test_normalize_preserves_cny_amount_units():
    row = {field: "1000.5" for field in collector.NUMERIC_FIELDS}
    row.update({"ts_code": "000001.SZ", "trade_date": "20200102"})

    frame = collector.normalize([row], date(2020, 1, 2))

    result = frame.row(0, named=True)
    assert result["symbol"] == "000001.SZ"
    assert result["rzye"] == 1000.5
    assert result["rzmre"] == 1000.5


def test_normalize_deduplicates_symbol_date():
    base = {field: "1" for field in collector.NUMERIC_FIELDS}
    rows = [
        {**base, "ts_code": "000001.SZ", "trade_date": "20200102"},
        {**base, "ts_code": "000001.SZ", "trade_date": "20200102"},
    ]

    assert collector.normalize(rows, date(2020, 1, 2)).height == 1
