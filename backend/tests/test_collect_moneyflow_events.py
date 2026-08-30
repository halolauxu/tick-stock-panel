from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "collect_moneyflow_events.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_moneyflow", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module()


def test_normalize_converts_ten_thousand_cny_units():
    trade_date = date(2020, 1, 2)
    row = {field: "1.5" for field in collector.AMOUNT_FIELDS}
    row.update({"ts_code": "000001.SZ", "trade_date": "20200102"})

    frame = collector.normalize([row], trade_date)

    result = frame.row(0, named=True)
    assert result["symbol"] == "000001.SZ"
    assert result["trade_date"] == trade_date
    assert result["buy_lg_cny"] == 15_000.0
    assert result["net_mf_cny"] == 15_000.0


def test_normalize_drops_wrong_date_and_duplicate_key():
    base = {field: "1" for field in collector.AMOUNT_FIELDS}
    rows = [
        {**base, "ts_code": "000001.SZ", "trade_date": "20200102"},
        {**base, "ts_code": "000001.SZ", "trade_date": "20200102"},
        {**base, "ts_code": "000002.SZ", "trade_date": "20200103"},
    ]

    frame = collector.normalize(rows, date(2020, 1, 2))

    assert frame.height == 1
