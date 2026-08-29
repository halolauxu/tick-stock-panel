"""Explicit create-only publication for evidence-qualified Alpha challengers."""
# Requirements: AM-S9-001 through AM-S9-012.
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

from app.alpha_mining.evidence import AlphaEvidenceStore
from app.strategy.ai_generator import AIStrategyGenerator

_ID = re.compile(r"^alpha_factor_[a-f0-9]{16}$")


class AlphaPublicationService:
    def __init__(self, data_dir: Path | str, strategy_engine) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.strategy_engine = strategy_engine
        self.evidence = AlphaEvidenceStore(self.data_dir)

    def publish(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.evidence.get_candidate(candidate_id)
        if candidate["state"]["state"] != "challenger":
            raise ValueError("只有全部历史、压力和前向门槛通过的挑战者可以发布")
        frozen = dict(candidate["candidate"])
        definition = dict(frozen.get("definition") or {})
        if definition.get("kind") != "factor_rank":
            raise ValueError("当前发布器只支持公共factor_rank执行合同")
        strategy_id = f"alpha_factor_{candidate_id.removeprefix('ac-')[:16]}"
        if not _ID.fullmatch(strategy_id):
            raise ValueError("Alpha发布策略ID不安全")
        source = self._render(strategy_id, candidate, definition)
        validation = AIStrategyGenerator().validate_code(source)
        if not validation.get("valid"):
            raise ValueError(f"Alpha策略渲染验证失败: {validation.get('error')}")
        root = self.data_dir / "strategies" / "custom"
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError("自定义策略目录不能是符号链接")
        root = root.resolve()
        if not root.is_relative_to(self.data_dir):
            raise ValueError("自定义策略目录越界")
        target = (root / f"{strategy_id}.py").resolve(strict=False)
        if target.parent != root or target.is_symlink():
            raise ValueError("Alpha发布目标越界")
        created = False
        if target.exists():
            if not target.is_file() or target.read_text(encoding="utf-8") != source:
                raise ValueError("Alpha策略ID冲突")
        else:
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(source)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, target)
                created = True
            finally:
                temporary.unlink(missing_ok=True)
        try:
            self.strategy_engine.reload()
            strategy = self.strategy_engine.get(strategy_id)
            if (
                strategy.meta.get("research_only")
                or strategy.meta.get("alpha_candidate_id") != candidate_id
                or strategy.source != "custom"
            ):
                raise ValueError("已发布Alpha策略的来源或权限校验失败")
        except Exception:
            if created and target.is_file() and target.read_text(encoding="utf-8") == source:
                target.unlink()
                self.strategy_engine.reload()
            raise
        receipt = self.evidence.write_publication_receipt(candidate_id, {
            "strategy_id": strategy_id,
            "candidate_sha256": candidate["content_sha256"],
            "path_name": target.name,
            "create_only": True,
        })
        return {"ok": True, "strategy_id": strategy_id, "receipt": receipt}

    @staticmethod
    def _render(strategy_id: str, candidate: dict[str, Any], definition: dict[str, Any]) -> str:
        scoring = {str(key): float(value) for key, value in dict(definition["scoring"]).items()}
        directions = {str(key): str(value) for key, value in dict(definition["directions"]).items()}
        parameters = dict(definition.get("parameters") or {})
        meta = {
            "id": strategy_id,
            "name": f"Alpha冠军候选 {candidate['engine_id']}",
            "description": "Published from immutable Alpha challenger evidence",
            "tags": ["alpha-mining", "factor-rank", "challenger"],
            "asset_types": ["stock"],
            "timeframes": ["1d"],
            "research_only": False,
            "alpha_candidate_id": candidate["candidate_id"],
            "alpha_candidate_sha256": candidate["content_sha256"],
            "params": [
                {"id": "entry_score", "label": "入场最低分", "type": "float", "default": float(parameters.get("entry_score", 70.0)), "min": 0.0, "max": 100.0, "step": 5.0},
                {"id": "exit_score", "label": "离场最高分", "type": "float", "default": float(parameters.get("exit_score", 40.0)), "min": 0.0, "max": 100.0, "step": 5.0},
                {"id": "top_rank", "label": "每日最多入选", "type": "int", "default": int(parameters.get("top_rank", 20)), "min": 1, "max": 100, "step": 1},
            ],
            "scoring": {},
            "order_by": "score",
            "descending": True,
            "limit": 100,
        }
        return (
            '\"\"\"Evidence-qualified strategy published by Alpha Mining.\"\"\"\n'
            "from app.strategy.builtin.factor_rank_research import FactorRankResearchMatrixStrategy\n\n"
            f"META = {meta!r}\n\n"
            'EXECUTION_BACKEND = "matrix_native"\n'
            'ENTRY_SIGNALS = ["signal_factor_rank_entry"]\n'
            'EXIT_SIGNALS = ["signal_factor_rank_exit"]\n'
            "STOP_LOSS = -0.08\n"
            "MAX_HOLD_DAYS = 30\n\n"
            f"SCORING = {scoring!r}\n"
            f"DIRECTIONS = {directions!r}\n"
            "MATRIX_STRATEGY = FactorRankResearchMatrixStrategy(SCORING, DIRECTIONS)\n"
        )
