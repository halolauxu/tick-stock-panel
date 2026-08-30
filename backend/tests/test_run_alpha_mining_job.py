from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "run_alpha_mining_job.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_alpha_mining_job", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_module()


def test_prepare_run_replaces_only_research_path_and_refreshes_digest() -> None:
    source = {
        "request": {
            "engine_ids": ["nonlinear_interaction"],
            "factor_names": ["momentum_20d"],
            "budget_profile": "strict",
            "forward_horizon": 20,
        },
        "data_fingerprint": {
            "asset_type": "stock",
            "generation": "g1",
            "digest": "old",
            "engines": [
                {"engine_id": "nonlinear_interaction", "version": "1.0.0"}
            ],
        },
    }

    request, fingerprint = launcher.prepare_run(
        source,
        ["matched_outcomes", "financial_revision"],
        {"matched_outcomes": "1.0.0", "financial_revision": "1.1.0"},
        max_candidates_per_engine=8,
        max_trials_per_engine=128,
    )

    assert request["engine_ids"] == ["matched_outcomes", "financial_revision"]
    assert request["factor_names"] == ["momentum_20d"]
    assert request["budget_profile"] == "strict"
    assert request["max_candidates_per_engine"] == 8
    assert request["max_trials_per_engine"] == 128
    assert fingerprint["generation"] == "g1"
    assert fingerprint["digest"] != "old"
    assert fingerprint["engines"] == [
        {"engine_id": "matched_outcomes", "version": "1.0.0"},
        {"engine_id": "financial_revision", "version": "1.1.0"},
    ]
