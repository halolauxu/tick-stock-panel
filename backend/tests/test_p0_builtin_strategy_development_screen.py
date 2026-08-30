from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_builtin_strategy_development_screen.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_builtin_strategy_screen", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_eligible_strategy_ids_exclude_research_and_non_stock() -> None:
    metadata = [
        {
            "id": "stock_builtin",
            "source": "builtin",
            "asset_types": ["stock"],
            "execution_backend": "matrix_native",
        },
        {
            "id": "research",
            "source": "builtin",
            "asset_types": ["stock"],
            "execution_backend": "matrix_native",
            "research_only": True,
        },
        {
            "id": "etf",
            "source": "builtin",
            "asset_types": ["etf"],
            "execution_backend": "matrix_native",
        },
    ]

    assert study.eligible_strategy_ids(metadata) == ["stock_builtin"]


def test_screen_gate_requires_return_risk_quality_and_sample_size() -> None:
    passed = study.evaluate_result(
        {
            "annual_return": 0.60,
            "max_drawdown": -0.20,
            "sharpe": 1.5,
            "n_trades": 120,
        }
    )
    failed = study.evaluate_result(
        {
            "annual_return": 0.80,
            "max_drawdown": -0.40,
            "sharpe": 1.5,
            "n_trades": 120,
        }
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_json_writer_serializes_point_in_time_dates(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    study._write_json(output, {"last_date": date(2020, 12, 31)})

    assert output.read_text(encoding="utf-8").strip().endswith(
        '"2020-12-31"\n}'
    )
