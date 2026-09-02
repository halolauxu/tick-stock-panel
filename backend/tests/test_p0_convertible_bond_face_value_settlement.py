from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "run_p0_convertible_bond_face_value_settlement.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "p0_cb_face_value_settlement", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_module()


def test_face_value_is_fixed_and_below_sample_official_prices() -> None:
    assert study.FACE_VALUE == 100.0
    assert study.FACE_VALUE < 100.29
    assert study.FACE_VALUE < 100.31
