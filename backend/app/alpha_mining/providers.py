"""Point-in-time provider boundaries used by Alpha research engines."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import polars as pl

from app.alpha_mining.contracts import (
    CandidateRenderer,
    DataQualification,
    EventProviderManifest,
    FeatureProviderManifest,
    FrozenSignalSpec,
    PointInTimeDataRequest,
    RendererManifest,
)


class ProviderContractError(ValueError):
    pass


class StandardFeatureProvider:
    """Validate an existing repository-backed loader before exposing its frame."""

    def __init__(
        self,
        manifest: FeatureProviderManifest,
        loader: Callable[[PointInTimeDataRequest], pl.DataFrame],
    ) -> None:
        self.manifest = manifest
        self._loader = loader

    def qualify(self, request: PointInTimeDataRequest) -> DataQualification:
        try:
            frame = self.load(request)
        except (OSError, ProviderContractError, ValueError) as exc:
            return DataQualification(False, (str(exc),), {"coverage": 0.0, "pit_verified": False})
        date_count = frame.get_column("date").n_unique() if "date" in frame.columns else 0
        return DataQualification(
            ready=not frame.is_empty(),
            reasons=() if not frame.is_empty() else ("数据集为空",),
            observations={
                "rows": frame.height,
                "dates": date_count,
                "coverage": 1.0 if frame.height else 0.0,
                "pit_verified": self.manifest.pit_capable,
                "provider_id": self.manifest.provider_id,
            },
        )

    def load(self, request: PointInTimeDataRequest) -> pl.DataFrame:
        if request.decision_clock not in {"after_close", "pre_open", "intraday"}:
            raise ProviderContractError("不支持的研究决策时钟")
        if "1d" not in self.manifest.supported_frequencies:
            raise ProviderContractError("特征Provider不支持日频")
        frame = self._loader(request)
        if not isinstance(frame, pl.DataFrame):
            raise ProviderContractError("特征Provider必须返回Polars DataFrame")
        required = {"symbol", "date", *request.columns}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ProviderContractError(f"特征Provider缺少字段: {missing}")
        normalized = frame.with_columns(pl.col("date").cast(pl.Date, strict=False))
        if normalized.get_column("date").null_count():
            raise ProviderContractError("特征Provider包含无效日期")
        start = date.fromisoformat(request.start)
        end = date.fromisoformat(request.end)
        if normalized.filter((pl.col("date") < start) | (pl.col("date") > end)).height:
            raise ProviderContractError("特征Provider返回了请求区间外的数据")
        if normalized.select(pl.struct("symbol", "date").is_duplicated().any()).item():
            raise ProviderContractError("特征Provider包含重复symbol/date键")
        return normalized.sort(["date", "symbol"])


class TimestampedEventProvider:
    """Expose timestamped events and map them to the next legal decision date."""

    def __init__(
        self,
        manifest: EventProviderManifest,
        loader: Callable[[PointInTimeDataRequest], pl.DataFrame],
        trading_dates: Callable[[PointInTimeDataRequest], Sequence[date]],
    ) -> None:
        self.manifest = manifest
        self._loader = loader
        self._trading_dates = trading_dates

    def qualify(self, request: PointInTimeDataRequest) -> DataQualification:
        try:
            frame = self.load_events(request)
        except (OSError, ProviderContractError, ValueError) as exc:
            return DataQualification(False, (str(exc),), {"coverage": 0.0, "pit_verified": False})
        return DataQualification(
            ready=not frame.is_empty(),
            reasons=() if not frame.is_empty() else ("事件数据为空",),
            observations={
                "rows": frame.height,
                "coverage": 1.0 if frame.height else 0.0,
                "pit_verified": True,
                "provider_id": self.manifest.provider_id,
            },
        )

    def load_events(self, request: PointInTimeDataRequest) -> pl.DataFrame:
        frame = self._loader(request)
        if not isinstance(frame, pl.DataFrame):
            raise ProviderContractError("事件Provider必须返回Polars DataFrame")
        published = self.manifest.published_at_field
        required = {"symbol", published}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ProviderContractError(f"事件Provider缺少字段: {missing}")
        normalized = frame.with_columns(
            pl.col(published).cast(pl.Datetime("us"), strict=False)
        )
        if normalized.get_column(published).null_count():
            raise ProviderContractError("事件发布时间不能为空")
        decision_dates = tuple(sorted(set(self._trading_dates(request))))
        if not decision_dates:
            raise ProviderContractError("缺少交易日历")
        effective = [
            _next_decision_date(value, decision_dates, request.decision_clock)
            for value in normalized.get_column(published).to_list()
        ]
        if any(value is None for value in effective):
            raise ProviderContractError("事件超出可映射的交易日范围")
        return normalized.with_columns(
            pl.Series(self.manifest.effective_date_field, effective, dtype=pl.Date)
        ).sort([self.manifest.effective_date_field, published, "symbol"])


class DeclarativeCandidateRenderer:
    manifest = RendererManifest(
        renderer_id="declarative.alpha.v1",
        version="1.0.0",
        candidate_types=(
            "factor_rank",
            "factor_rank_intersection",
            "market_regime_rank",
            "event_sequence_rank",
            "residual_rank",
        ),
        schema_version="1.0",
    )

    def render(self, candidate: FrozenSignalSpec) -> Mapping[str, Any]:
        if candidate.signal_kind not in self.manifest.candidate_types:
            raise ProviderContractError(
                f"渲染器不支持候选类型: {candidate.signal_kind}"
            )
        return {
            "schema_version": self.manifest.schema_version,
            "renderer_id": self.manifest.renderer_id,
            "candidate": candidate.to_dict(),
            "sections": ["thesis", "definition", "training_evidence", "validation"],
        }


def assert_provider_instance(provider: object) -> None:
    if not hasattr(provider, "manifest"):
        raise ProviderContractError("研究数据Provider缺少manifest")
    if not callable(getattr(provider, "qualify", None)):
        raise ProviderContractError("研究数据Provider缺少qualify")


def assert_renderer_instance(renderer: object) -> None:
    if not isinstance(renderer, CandidateRenderer):
        raise ProviderContractError("候选渲染器不符合CandidateRenderer契约")


def _next_decision_date(
    published_at: datetime,
    trading_dates: Sequence[date],
    decision_clock: str,
) -> date | None:
    published_day = published_at.date()
    for trading_day in trading_dates:
        if trading_day < published_day:
            continue
        if trading_day == published_day:
            if decision_clock == "after_close" and published_at.hour < 15:
                return trading_day
            if decision_clock == "pre_open" and published_at.hour < 9:
                return trading_day
            continue
        return trading_day
    return None
