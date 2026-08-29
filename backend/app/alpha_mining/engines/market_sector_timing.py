"""Market and sector conditional-time-series discovery."""
from __future__ import annotations

import polars as pl

from app.alpha_mining.contracts import (
    ENGINE_API_VERSION,
    AlphaEngineManifest,
    DataCatalogContext,
    DatasetRequirement,
)
from app.alpha_mining.engines._shared import materialize, rank_univariate_candidates


class MarketSectorTimingEngine:
    manifest = AlphaEngineManifest(
        engine_id="market_sector_timing",
        version="1.0.0",
        api_version=ENGINE_API_VERSION,
        name="市场与板块时序",
        family="market_sector_timing",
        information_domains=("market_regime", "industry", "price_volume"),
        mechanism_classes=("structural_flow", "information_diffusion"),
        economic_mechanism="市场广度与行业状态改变个股因子的收益兑现条件",
        discovery_classes=("conditional_time_series",),
        discovery_method="按训练窗市场方向和点时行业状态分层检验截面因子",
        prediction_objects=("forward_net_return", "market_residual_return"),
        forecast_targets=("1d", "3d", "5d", "10d", "20d"),
        required_datasets=(
            DatasetRequirement("daily_enriched", 0.95, True, "date"),
            DatasetRequirement("historical_universe", 0.95, True, "available_date"),
            DatasetRequirement("industry_pit", 0.90, True, "in_date"),
        ),
        forecast_horizons=(1, 3, 5, 10, 20),
        output_candidate_types=("factor_rank",),
        description="从市场和板块状态寻找可重复条件, 不引用任何现有策略信号。",
    )

    def preflight(self, context: DataCatalogContext):
        from app.alpha_mining.contracts import qualify_manifest_datasets

        return qualify_manifest_datasets(self.manifest, context.datasets)

    def discover(self, context, budget):
        frame = context.frame
        if "target_residual_return" in frame.columns:
            frame = frame.with_columns(
                pl.col("target_residual_return").alias(context.target_column)
            )
            context = type(context)(
                frame=frame,
                date_labels=context.date_labels,
                feature_names=context.feature_names,
                target_column=context.target_column,
                asset_type=context.asset_type,
                metadata=context.metadata,
            )
        return rank_univariate_candidates(
            context=context,
            budget=budget,
            engine_id=self.manifest.engine_id,
            engine_version=self.manifest.version,
            recipe_prefix="market_sector_timing",
            name_prefix="状态条件因子",
            thesis="训练窗中该因子对市场残差收益的方向在多个市场/行业状态下保持一致。",
        )

    def materialize(self, candidate, context):
        del context
        return materialize(candidate)


ENGINE = MarketSectorTimingEngine()
