from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "probe_market_microstructure_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_microstructure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_module()


class FakeClient:
    def query(self, api_name, params, fields):
        date_field = params["trade_date"]
        row = {field: 1.0 for field in fields}
        row["ts_code"] = "000001.SZ"
        row["trade_date"] = date_field
        return [row]


def test_run_marks_all_three_sources_ready(tmp_path):
    output = tmp_path / "audit.json"

    result = probe.run(output, client=FakeClient())

    assert result["decision"]["ready_for_bounded_collection"] == [
        "moneyflow",
        "margin_detail",
        "stk_auction_o",
    ]
    assert output.is_file()


def test_specs_use_distinct_auction_dates():
    auction = next(spec for spec in probe.SPECS if spec.api_name == "stk_auction_o")

    assert auction.requests == probe.AUCTION_DATES
    assert auction.requests != probe.FLOW_DATES
