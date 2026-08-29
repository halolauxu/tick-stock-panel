from __future__ import annotations

# Requirements: AM-S2-006 through AM-S2-016 and AM-S6-002.
import importlib
import sys
from dataclasses import replace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    CandidateSpec,
    DataQualification,
    FrozenSignalSpec,
    TrainOnlyContext,
    TrialBudget,
)
from app.alpha_mining.registry import (
    AlphaEngineRegistry,
    AlphaEngineRegistryError,
)


class DummyEngine:
    manifest = AlphaEngineManifest(
        engine_id="dummy",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="Dummy",
        family="test",
        information_domains=("price_volume",),
        mechanism_classes=("risk_compensation",),
        economic_mechanism="test only",
        discovery_classes=("cross_sectional_rank",),
        discovery_method="deterministic",
        prediction_objects=("forward_net_return",),
        forecast_targets=("5d",),
        required_features=("factor",),
    )

    def preflight(self, context):
        return DataQualification(ready=True, reasons=(), observations={"rows": 2})

    def discover(self, context: TrainOnlyContext, budget: TrialBudget):
        del context, budget
        return [
            CandidateSpec(
                recipe_id="dummy.rank",
                engine_id=self.manifest.engine_id,
                engine_version=self.manifest.version,
                name="Dummy candidate",
                thesis="test",
                signal_kind="factor_rank",
                features=("factor",),
                directions=(1,),
                weights=(1.0,),
                parameters={"top_rank": 20},
                train_evidence={"score": 1.0},
            )
        ]

    def materialize(self, candidate, context):
        del context
        return FrozenSignalSpec.from_candidate(candidate)


def test_registry_freezes_and_rejects_duplicate_ids() -> None:
    registry = AlphaEngineRegistry()
    registry.register(DummyEngine())

    with pytest.raises(AlphaEngineRegistryError, match="duplicate engine id"):
        registry.register(DummyEngine())

    registry.freeze()
    with pytest.raises(AlphaEngineRegistryError, match="frozen"):
        registry.register(DummyEngine())


def test_registry_rejects_incompatible_contract_version() -> None:
    engine = DummyEngine()
    engine.manifest = replace(engine.manifest, api_version="99.0")

    with pytest.raises(AlphaEngineRegistryError, match="API version"):
        AlphaEngineRegistry().register(engine)


def test_registry_runs_engine_without_orchestrator_branching() -> None:
    registry = AlphaEngineRegistry()
    registry.register(DummyEngine())
    registry.freeze()
    context = TrainOnlyContext(
        frame=pl.DataFrame({"date": ["2026-01-01"], "factor": [1.0], "_target_5d": [0.1]}),
        date_labels=("2026-01-01",),
        feature_names=("factor",),
        target_column="_target_5d",
        asset_type="stock",
        metadata={},
    )

    candidates, failures = registry.discover(
        ["dummy"],
        context,
        TrialBudget(max_candidates=2, max_trials=4),
    )

    assert failures == []
    assert [candidate.recipe_id for candidate in candidates] == ["dummy.rank"]


def test_registry_isolates_one_engine_failure() -> None:
    class BrokenEngine(DummyEngine):
        manifest = replace(DummyEngine.manifest, engine_id="broken", name="Broken")

        def discover(self, context, budget):
            del context, budget
            raise RuntimeError("boom")

    registry = AlphaEngineRegistry()
    registry.register(BrokenEngine())
    registry.register(DummyEngine())
    registry.freeze()
    context = TrainOnlyContext(
        frame=pl.DataFrame({"date": ["2026-01-01"], "factor": [1.0], "_target_5d": [0.1]}),
        date_labels=("2026-01-01",),
        feature_names=("factor",),
        target_column="_target_5d",
        asset_type="stock",
        metadata={},
    )

    candidates, failures = registry.discover(
        ["broken", "dummy"],
        context,
        TrialBudget(max_candidates=2, max_trials=4),
    )

    assert [candidate.recipe_id for candidate in candidates] == ["dummy.rank"]
    assert failures == [{"engine_id": "broken", "stage": "discover", "error": "boom"}]


def test_malicious_engine_cannot_read_outer_test_rows() -> None:
    class MaliciousEngine(DummyEngine):
        manifest = replace(DummyEngine.manifest, engine_id="malicious", name="Malicious")

        def discover(self, context, budget):
            del budget
            return context.outer_test_rows

    registry = AlphaEngineRegistry()
    registry.register(MaliciousEngine())
    registry.freeze()
    context = TrainOnlyContext(
        frame=pl.DataFrame({"date": ["2026-01-01"], "factor": [1.0], "_target_5d": [0.1]}),
        date_labels=("2026-01-01",),
        feature_names=("factor",),
        target_column="_target_5d",
        asset_type="stock",
    )
    candidates, failures = registry.discover(["malicious"], context, TrialBudget())
    assert candidates == []
    assert "outer_test_rows" in failures[0]["error"]
    assert not hasattr(context, "__dict__")


def test_auto_discovery_adds_and_removes_module_without_orchestrator_change(tmp_path, monkeypatch) -> None:
    package = tmp_path / "dynamic_alpha_engines"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "engine_one.py"
    module.write_text(
        "from app.alpha_mining.contracts import *\n"
        "class E:\n"
        " manifest=AlphaEngineManifest(engine_id='plug',version='1',api_version=ENGINE_API_VERSION,name='Plug',family='test',information_domains=('price_volume',),mechanism_classes=('risk_compensation',),economic_mechanism='x',discovery_classes=('cross_sectional_rank',),discovery_method='x',prediction_objects=('forward_net_return',),forecast_targets=('5d',))\n"
        " def preflight(self,c): return DataQualification(True,())\n"
        " def discover(self,c,b): return []\n"
        " def materialize(self,candidate,context): return FrozenSignalSpec.from_candidate(candidate)\n"
        "ENGINE=E()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    from app.alpha_mining.registry import load_builtin_registry

    registry, failures = load_builtin_registry("dynamic_alpha_engines")
    assert failures == []
    assert [engine.manifest.engine_id for engine in registry.list()] == ["plug"]
    from app.api import alpha_mining as alpha_api

    app = FastAPI()
    app.include_router(alpha_api.router)
    monkeypatch.setattr(alpha_api, "_REGISTRY", registry)
    assert [
        item["engine_id"]
        for item in TestClient(app).get("/api/alpha-mining/v1/engines").json()["items"]
    ] == ["plug"]
    module.unlink()
    sys.modules.pop("dynamic_alpha_engines.engine_one", None)
    importlib.invalidate_caches()
    registry, failures = load_builtin_registry("dynamic_alpha_engines")
    assert failures == []
    assert registry.list() == ()
    monkeypatch.setattr(alpha_api, "_REGISTRY", registry)
    assert TestClient(app).get("/api/alpha-mining/v1/engines").json()["items"] == []


def test_auto_discovery_rejects_direct_supplier_import(tmp_path, monkeypatch) -> None:
    package = tmp_path / "bad_alpha_engines"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "bad.py").write_text("import requests\nENGINE = object()\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    from app.alpha_mining.registry import load_builtin_registry

    registry, failures = load_builtin_registry("bad_alpha_engines")
    assert registry.list() == ()
    assert "ResearchProvider boundary" in failures[0]["error"]
