"""Announcement-time financial revision and expectation-gap discovery."""
from __future__ import annotations

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    DataCatalogContext,
    DataQualification,
    DatasetRequirement,
    qualify_manifest_datasets,
)
from app.alpha_mining.engines._shared import materialize, rank_univariate_candidates

FINANCIAL_FEATURES = (
    "pb_latest",
    "roe_latest",
    "gross_margin_latest",
    "net_margin_latest",
    "revenue_yoy_latest",
    "net_income_yoy_latest",
    "debt_ratio_latest",
    "financial_revision_revenue",
    "financial_revision_profit",
)


class FinancialRevisionEngine:
    manifest = AlphaEngineManifest(
        engine_id="financial_revision",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="财务变化与预期差",
        family="financial_revision",
        information_domains=("fundamentals", "corporate_event"),
        mechanism_classes=("expectation_revision", "behavioral_underreaction"),
        economic_mechanism="公告后的盈利质量变化和相对历史预期差可能被市场分期消化",
        discovery_classes=("revision_surprise", "event_study"),
        discovery_method="按公告时间构造同比变化、环比修正及同行业标准化意外",
        prediction_objects=("forward_net_return", "market_residual_return", "gap_risk"),
        forecast_targets=("3d", "5d", "10d", "20d", "60d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
            DatasetRequirement("financial_pit", 0.90, True, "announce_date"),
        ),
        forecast_horizons=(3, 5, 10, 20, 60),
        output_candidate_types=("factor_rank",),
        description="财务字段在公告前必须为空; 报告期不能替代公告时间。",
    )

    def preflight(self, context: DataCatalogContext):
        data = qualify_manifest_datasets(self.manifest, context.datasets)
        available = tuple(name for name in FINANCIAL_FEATURES if name in context.available_features)
        reasons = list(data.reasons)
        if not available:
            reasons.append("缺少公告日门控的财务特征")
        return DataQualification(not reasons, tuple(reasons), {**data.observations, "financial_features": available})

    def discover(self, context, budget):
        features = tuple(name for name in FINANCIAL_FEATURES if name in context.feature_names)
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="financial_revision",
            name_prefix="财务预期差",
            thesis="公告后可见的财务变化特征在训练窗中对未来净收益具有稳定排序。",
            features=features,
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = FinancialRevisionEngine()
