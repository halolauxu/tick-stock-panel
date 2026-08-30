from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "run_p0_major_contract_development.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p0_major_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _contract(
    event_id: str,
    ann_date: date,
    ratio: float,
    contract_type: str = "销售合同",
    name: str = "正式销售合同",
) -> dict:
    return {
        "event_id": event_id,
        "ann_date": ann_date,
        "symbol": "600001.SH",
        "company_name": "测试公司",
        "contract_name": name,
        "contract_type_code": "001001",
        "contract_type_name": contract_type,
        "signatory": "测试公司",
        "signatory_relation": "公司本身",
        "counterparty": "客户",
        "counterparty_relation": "无关联关系",
        "sign_date": ann_date,
        "contract_amount_cny": 100.0,
        "previous_revenue_cny": 100.0,
        "revenue_ratio_pct": ratio,
        "ratio_source": "test",
        "is_abolished": "",
        "contents": "正式生效",
        "stated_effect": "预计产生积极影响",
    }


def test_frozen_taxonomy_separates_magnitude_and_negative_control() -> None:
    frame = pl.DataFrame(
        [
            _contract("E1", date(2019, 1, 1), 60.0),
            _contract("E2", date(2019, 2, 1), 25.0),
            _contract("E3", date(2019, 3, 1), 7.0),
            _contract("E4", date(2019, 4, 1), 80.0, name="框架销售合同"),
            _contract("E5", date(2019, 5, 1), 80.0, contract_type="采购合同"),
        ]
    )

    result = study.categorize_events(frame)

    assert result["category"].to_list() == [
        "transformational_contract",
        "material_contract",
        "low_impact_control",
    ]
    assert result["holding_trading_days"].to_list() == [20, 10, 10]


def test_same_category_cooldown_keeps_first_after_90_calendar_days() -> None:
    frame = pl.DataFrame(
        [
            _contract("E1", date(2019, 1, 1), 60.0),
            _contract("E2", date(2019, 3, 1), 70.0),
            _contract("E3", date(2019, 4, 1), 80.0),
        ]
    )

    result = study.categorize_events(frame)

    assert result["ann_date"].to_list() == [date(2019, 1, 1), date(2019, 4, 1)]
