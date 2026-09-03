import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.collect_p0_hfc_quota_evidence import (  # noqa: E402
    build_issuer_mapping,
    qualify_forecast_reason,
    validate_quota_rows,
)


def test_forecast_reason_requires_all_three_bound_claims() -> None:
    complete = (
        "第三代氟制冷剂继续实行生产配额管理,行业供给端约束强化,"
        "产品价格持续上行,毛利率稳步提升。"
    )
    result = qualify_forecast_reason(complete)
    assert result == {
        "quota_constraint": True,
        "price_increase": True,
        "earnings_improvement": True,
        "qualified": True,
    }

    missing_price = qualify_forecast_reason(
        "第三代氟制冷剂继续实行生产配额管理,利润增长。"
    )
    assert missing_price["qualified"] is False
    assert missing_price["price_increase"] is False


def test_quota_audit_uses_document_totals_not_cross_product_additivity() -> None:
    rows = [
        {
            "sequence": "1",
            "quota_entity": "企业甲",
            "product": "HFC-32",
            "production_quota_tonnes": 791_881,
        },
        {
            "sequence": "34",
            "quota_entity": "企业乙",
            "product": "HFC-125",
            "production_quota_tonnes": 1,
        },
    ]
    audit = validate_quota_rows(2025, rows)
    assert audit["production_total_tonnes"] == 791_882
    assert audit["checks"]["production_total_matches"] is True
    assert audit["checks"]["producer_count_matches"] is False
    assert audit["passed"] is False


def test_every_frozen_issuer_must_match_a_quota_entity() -> None:
    rows = [
        {
            "quota_year": 2025,
            "quota_entity": "浙江三美化工股份有限公司",
            "product": "HFC-125",
        }
    ]
    mappings = build_issuer_mapping(rows, 2025)
    passed = {row["symbol"]: row["passed"] for row in mappings}
    assert passed["603379.SH"] is True
    assert passed["600160.SH"] is False
