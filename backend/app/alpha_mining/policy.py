"""Research charter, coverage map, and centrally enforced promotion gates."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ALPHA_ALGORITHM_VERSION = "alpha-mining-v1"

HARD_GATES: tuple[dict[str, Any], ...] = (
    {"id": "return_vs_champion", "label": "13年拼接样本外净收益高于冠军", "required": True},
    {"id": "sharpe", "label": "夏普≥max(0.8, 冠军+0.2)", "required": True},
    {"id": "drawdown", "label": "最大回撤≤25%且不差于冠军", "required": True},
    {"id": "positive_half_years", "label": "正收益半年窗口≥65%", "required": True},
    {"id": "beat_champion_windows", "label": "跑赢冠军半年窗口≥60%", "required": True},
    {"id": "recent_year", "label": "最近一年跑赢冠军", "required": True},
    {"id": "recent_quarter", "label": "最近三个月绝对收益为正", "required": True},
    {"id": "double_cost", "label": "双倍成本后仍为正", "required": True},
    {"id": "delay", "label": "延迟成交不崩塌", "required": True},
    {"id": "parameter_perturbation", "label": "参数扰动不崩塌", "required": True},
    {"id": "capacity", "label": "扩大持仓容量后仍为正", "required": True},
    {"id": "concentration", "label": "行业/个股/交易贡献集中度受控", "required": True},
    {"id": "forward_shadow", "label": "前瞻模拟达到预注册门槛", "required": True},
)

# This is a coverage roadmap, not a closed taxonomy. Ready engines come from the registry.
COVERAGE_ROADMAP: tuple[dict[str, Any], ...] = (
    {"id": "cross_sectional", "name": "截面排序", "status": "ready", "slot": "engine"},
    {"id": "outcome_attribution", "name": "赢家/输家归因", "status": "ready", "slot": "engine"},
    {"id": "interaction", "name": "非线性交互", "status": "ready", "slot": "engine"},
    {"id": "market_sector_timing", "name": "市场/行业时序", "status": "ready", "slot": "engine"},
    {"id": "event_sequence", "name": "事件序列", "status": "ready", "slot": "engine"},
    {"id": "network_diffusion", "name": "行业/概念扩散", "status": "ready", "slot": "engine"},
    {"id": "financial_revision", "name": "财务变化与预期差", "status": "ready", "slot": "engine"},
    {"id": "portfolio_residual", "name": "组合残差与互补收益", "status": "ready", "slot": "engine"},
    {"id": "auction_microstructure", "name": "竞价与微观结构", "status": "data_gap", "slot": "engine"},
    {"id": "holder_supply", "name": "股东/筹码供需", "status": "data_gap", "slot": "engine"},
    {"id": "event_text", "name": "公告/政策文本事件", "status": "data_gap", "slot": "engine"},
    {"id": "cross_asset", "name": "跨资产传导", "status": "data_gap", "slot": "engine"},
    {"id": "relative_value", "name": "相对价值与配对", "status": "planned", "slot": "engine"},
    {"id": "future_engine", "name": "后续未知发现路径", "status": "extension_slot", "slot": "engine"},
)


def charter() -> dict[str, Any]:
    return {
        "algorithm_version": ALPHA_ALGORITHM_VERSION,
        "objective": "稳定找到净收益、风险收益比和跨窗口表现均高于当前冠军的可交易 Alpha",
        "benchmark_policy": "发现不依赖冠军; 所有候选最终必须以相同资金、成本和成交口径对比当前冠军",
        "historical_policy": "13年数据用于滚动训练与拼接样本外; 所有发现引擎只能读取外层训练窗",
        "completion_levels": [
            {"id": "engineering", "label": "工程完成", "meaning": "链路、隔离、重跑和审计通过"},
            {"id": "research", "label": "研究完成", "meaning": "候选完成冻结与样本外证伪"},
            {"id": "goal", "label": "目标完成", "meaning": "候选通过全部历史硬门槛和前瞻模拟"},
        ],
        "hard_gates": list(HARD_GATES),
        "coverage_roadmap": list(COVERAGE_ROADMAP),
        "extension_contract": {
            "registry_frozen_after_startup": True,
            "engine_failure_isolated": True,
            "orchestrator_edit_required_for_new_engine": False,
            "api_version": "1.0",
        },
    }


def evaluate_historical_gates(
    metrics: Mapping[str, Any],
    champion: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return explicit pass/fail/pending evidence; never infer missing tests as passes."""
    output: list[dict[str, Any]] = []

    def add(gate_id: str, passed: bool | None, actual: Any, required: Any) -> None:
        output.append({
            "id": gate_id,
            "status": "pending" if passed is None else ("passed" if passed else "failed"),
            "actual": actual,
            "required": required,
        })

    candidate_return = _number(metrics.get("stitched_oos_return"))
    champion_return = _number(champion.get("stitched_oos_return"))
    add(
        "return_vs_champion",
        None if candidate_return is None or champion_return is None else candidate_return > champion_return,
        candidate_return,
        champion_return,
    )
    candidate_sharpe = _number(metrics.get("stitched_oos_sharpe"))
    champion_sharpe = _number(champion.get("stitched_oos_sharpe"))
    sharpe_floor = None if champion_sharpe is None else max(0.8, champion_sharpe + 0.2)
    add(
        "sharpe",
        None if candidate_sharpe is None or sharpe_floor is None else candidate_sharpe >= sharpe_floor,
        candidate_sharpe,
        sharpe_floor,
    )
    candidate_dd = _number(metrics.get("max_drawdown"))
    champion_dd = _number(champion.get("max_drawdown"))
    dd_floor = None if champion_dd is None else max(-0.25, champion_dd)
    add(
        "drawdown",
        None if candidate_dd is None or dd_floor is None else candidate_dd >= dd_floor,
        candidate_dd,
        dd_floor,
    )
    for gate_id, metric_id, threshold in (
        ("positive_half_years", "positive_half_year_ratio", 0.65),
        ("beat_champion_windows", "beat_champion_half_year_ratio", 0.60),
    ):
        actual = _number(metrics.get(metric_id))
        add(gate_id, None if actual is None else actual >= threshold, actual, threshold)
    recent_year = _number(metrics.get("recent_1y_return"))
    champion_year = _number(champion.get("recent_1y_return"))
    add(
        "recent_year",
        None if recent_year is None or champion_year is None else recent_year > champion_year,
        recent_year,
        champion_year,
    )
    recent_quarter = _number(metrics.get("recent_3m_return"))
    add("recent_quarter", None if recent_quarter is None else recent_quarter > 0, recent_quarter, "> 0")
    for gate_id, metric_id in (
        ("double_cost", "double_cost_return"),
        ("delay", "delayed_entry_return"),
        ("parameter_perturbation", "worst_parameter_return"),
        ("capacity", "capacity_passed"),
        ("concentration", "concentration_passed"),
    ):
        actual = metrics.get(metric_id)
        if gate_id in {"capacity", "concentration"}:
            passed = actual if isinstance(actual, bool) else None
            required: Any = True
        else:
            value = _number(actual)
            passed = None if value is None else value > 0
            actual = value
            required = "> 0"
        add(gate_id, passed, actual, required)
    add("forward_shadow", None, None, "前瞻模拟完成后判定")
    return output


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result
