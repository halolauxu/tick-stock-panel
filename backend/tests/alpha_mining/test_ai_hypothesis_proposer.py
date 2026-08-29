# ruff: noqa: RUF001
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.alpha_mining import ai_hypothesis_proposer as proposer_module
from app.alpha_mining.ai_hypothesis_proposer import AlphaAIHypothesisProposer


def _write(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _research_fixture(tmp_path: Path) -> tuple[date, date]:
    first = date(2026, 1, 5)
    last = date(2026, 1, 6)
    _write(tmp_path / "research" / "historical_stock_universe.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "list_date": [date(2020, 1, 1), date(2020, 1, 1)],
        "delist_date": [None, None],
    }))
    _write(tmp_path / "research" / "historical_stock_names.parquet", pl.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "name": ["甲公司", "乙公司"],
        "start_date": [date(2020, 1, 1), date(2020, 1, 1)],
        "end_date": [None, None],
    }))
    for day, closes in ((first, [10.0, 20.0]), (last, [10.5, 19.0])):
        _write(
            tmp_path / "kline_daily_enriched" / f"date={day.isoformat()}" / "part.parquet",
            pl.DataFrame({
                "symbol": ["000001.SZ", "000002.SZ"],
                "date": [day, day],
                "close": closes,
                "amount": [1.2e8, 8.0e7],
                "consecutive_limit_ups": [0, 0],
                "consecutive_limit_downs": [0, 0],
            }),
        )
    return first, last


@pytest.mark.asyncio
async def test_deepseek_proposals_are_validated_frozen_and_auditable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, last = _research_fixture(tmp_path)
    captured: dict[str, object] = {}

    async def fake_generator(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return json.dumps({
            "hypotheses": [{
                "title": "尾盘承接与缩量卖压衰竭",
                "thesis": "短期下跌且缩量、同时收盘位置改善的股票，未来5日净收益更高。",
                "mechanism": "A股T+1约束使止损卖盘集中释放，缩量与尾盘承接共同出现时可能代表边际卖压衰竭。",
                "forward_horizon": 5,
                "information_domains": ["price_volume", "liquidity"],
                "factor_names": ["momentum_5d", "vol_ratio_5d", "close_position"],
                "expected_directions": {
                    "momentum_5d": -1,
                    "vol_ratio_5d": -1,
                    "close_position": 1,
                },
                "weights": {
                    "momentum_5d": 0.4,
                    "vol_ratio_5d": 0.25,
                    "close_position": 0.35,
                },
                "falsification": ["独立样本外净收益不为正", "双倍成本后净收益不为正"],
            }],
        }, ensure_ascii=False)

    monkeypatch.setattr(proposer_module, "current_ai_provider", lambda: "openai_compat")
    monkeypatch.setattr(proposer_module, "current_ai_model", lambda: "deepseek-v4-pro")
    monkeypatch.setattr(proposer_module, "ai_configured", lambda provider=None: True)

    result = await AlphaAIHypothesisProposer(tmp_path, generator=fake_generator).propose(
        asset_type="stock",
        start=first,
        end=last,
        count=1,
    )

    assert result["model"] == "deepseek-v4-pro"
    assert result["outcome_data_exposed"] is False
    assert len(result["items"]) == 1
    hypothesis = result["items"][0]
    assert hypothesis["source_kind"] == "ai"
    assert hypothesis["test_spec"]["engine_ids"] == ["cross_sectional_rank"]
    assert hypothesis["test_spec"]["expected_directions"]["momentum_5d"] == -1
    assert sum(hypothesis["test_spec"]["weights"].values()) == pytest.approx(1.0)
    assert hypothesis["provenance"]["outcome_data_exposed"] is False
    prompt = str(captured["messages"])
    assert "available_factors" in prompt
    assert "stitched_oos_return" not in prompt
    assert "样本外收益\": -" not in prompt
    receipt = tmp_path / "alpha_mining" / "hypothesis_proposals" / f"{result['batch_id']}.json"
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["raw_response"]


@pytest.mark.asyncio
async def test_deepseek_unknown_factor_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, last = _research_fixture(tmp_path)

    async def fake_generator(messages, **kwargs):
        return json.dumps({"hypotheses": [{
            "title": "不存在因子假设",
            "thesis": "该假设故意使用目录外因子以验证确定性门禁。",
            "mechanism": "该机制说明长度满足要求但字段必须被系统拒绝。",
            "forward_horizon": 5,
            "information_domains": ["price_volume"],
            "factor_names": ["unknown_alpha", "momentum_5d"],
            "expected_directions": {"unknown_alpha": 1, "momentum_5d": -1},
            "weights": {"unknown_alpha": 0.5, "momentum_5d": 0.5},
            "falsification": ["外测收益不为正", "成本压力后失效"],
        }]}, ensure_ascii=False)

    monkeypatch.setattr(proposer_module, "current_ai_provider", lambda: "openai_compat")
    monkeypatch.setattr(proposer_module, "current_ai_model", lambda: "deepseek-v4-pro")
    monkeypatch.setattr(proposer_module, "ai_configured", lambda provider=None: True)

    with pytest.raises(ValueError, match="当前不可用的因子"):
        await AlphaAIHypothesisProposer(tmp_path, generator=fake_generator).propose(
            asset_type="stock",
            start=first,
            end=last,
            count=1,
        )


@pytest.mark.asyncio
async def test_non_deepseek_model_cannot_impersonate_ai_hypothesis_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, last = _research_fixture(tmp_path)
    monkeypatch.setattr(proposer_module, "current_ai_provider", lambda: "openai_compat")
    monkeypatch.setattr(proposer_module, "current_ai_model", lambda: "another-model")
    monkeypatch.setattr(proposer_module, "ai_configured", lambda provider=None: True)

    with pytest.raises(ValueError, match="切换到DeepSeek"):
        await AlphaAIHypothesisProposer(tmp_path).propose(
            asset_type="stock",
            start=first,
            end=last,
        )
