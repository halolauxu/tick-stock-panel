# ruff: noqa: RUF001

from app.services.serenity_thesis_iteration import (
    conservative_dimension_consensus,
    is_full_annual_report,
    thesis_dimension_gate,
)


def test_full_annual_report_excludes_half_year_summary_and_english() -> None:
    assert is_full_annual_report("某公司2025年年度报告") is True
    assert is_full_annual_report("某公司2024年年度报告（更正后）") is True
    assert is_full_annual_report("某公司2025年半年度报告") is False
    assert is_full_annual_report("某公司2025年年度报告摘要") is False
    assert is_full_annual_report("某公司2025年年度报告（英文版）") is False


def test_thesis_gate_requires_complete_score_and_all_five_ratings() -> None:
    dimensions = [
        {"dimension_id": dimension_id, "status": "EVIDENCED", "rating": 3}
        for dimension_id in (
            "architecture_coupling",
            "chokepoint_severity",
            "supplier_concentration",
            "expansion_difficulty",
            "evidence_quality",
        )
    ]
    assert thesis_dimension_gate(dimensions, 38.4) is True
    assert thesis_dimension_gate(dimensions, None) is False
    assert thesis_dimension_gate(dimensions, 38.3) is False
    dimensions[2]["rating"] = 2
    assert thesis_dimension_gate(dimensions, 50.0) is False


def test_consensus_keeps_evidence_over_unknown_and_takes_lower_rating() -> None:
    first = [
        {
            "dimension_id": "architecture_coupling",
            "status": "EVIDENCED",
            "rating": 4,
            "evidence": [{"document_id": "DOC-1", "page": 1, "quote": "证据一"}],
        },
        {
            "dimension_id": "chokepoint_severity",
            "status": "EVIDENCED",
            "rating": 4,
            "evidence": [],
        },
    ]
    second = [
        {
            "dimension_id": "architecture_coupling",
            "status": "UNKNOWN",
            "rating": None,
            "evidence": [],
        },
        {
            "dimension_id": "chokepoint_severity",
            "status": "EVIDENCED",
            "rating": 3,
            "evidence": [],
        },
    ]
    merged = {item["dimension_id"]: item for item in conservative_dimension_consensus(first, second)}
    assert merged["architecture_coupling"]["rating"] == 4
    assert merged["chokepoint_severity"]["rating"] == 3
    assert merged["supplier_concentration"]["status"] == "UNKNOWN"
