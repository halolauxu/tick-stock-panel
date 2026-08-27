# ruff: noqa: RUF001

from app.services.serenity_thesis_iteration import (
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
