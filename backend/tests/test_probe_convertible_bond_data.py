from __future__ import annotations

import importlib.util
from pathlib import Path

from app.plugins.tushare.client import TushareError

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "probe_convertible_bond_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_cb_data", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_module()


def test_summarize_keyed_rows_reports_duplicates_and_nulls() -> None:
    summary = probe.summarize_keyed_rows(
        [
            {"code": "A", "date": "20260101", "close": 1.0},
            {"code": "A", "date": "20260101", "close": None},
            {"code": "B", "date": "20260101", "close": 2.0},
        ],
        key=("code", "date"),
        critical_fields=("close",),
    )

    assert summary["rows"] == 3
    assert summary["unique_keys"] == 2
    assert summary["duplicate_keys"] == 1
    assert summary["missing_rates"]["close"] == 1 / 3


def test_safe_probe_keeps_permission_failure_as_data() -> None:
    def fail() -> list[dict]:
        raise TushareError("没有接口权限")

    result = probe.safe_probe(fail)

    assert result["status"] == "UNAVAILABLE"
    assert result["rows"] == []
    assert "没有接口权限" in result["error"]
