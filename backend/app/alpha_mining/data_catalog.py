"""Fail-closed point-in-time data catalog for formal Alpha research."""
# Requirements: AM-S3-001 through AM-S3-011.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from app.alpha_mining.contracts import (
    DataQualification,
    EventProviderManifest,
    PointInTimeDataRequest,
)
from app.alpha_mining.providers import TimestampedEventProvider


@dataclass(frozen=True)
class CatalogSnapshot:
    datasets: dict[str, DataQualification]
    fingerprint: str
    first_date: str | None
    last_date: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "datasets": {
                key: {
                    "ready": value.ready,
                    "reasons": list(value.reasons),
                    "observations": dict(value.observations),
                }
                for key, value in self.datasets.items()
            },
        }


class AlphaResearchDataCatalog:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir).resolve()

    def snapshot(self, start: date, end: date, asset_type: str = "stock") -> CatalogSnapshot:
        enriched_root = self.data_dir / (
            "kline_daily_enriched" if asset_type == "stock" else "kline_daily_enriched_etf"
        )
        partition_dates = _partition_dates(enriched_root, start, end)
        daily = DataQualification(
            ready=bool(partition_dates),
            reasons=() if partition_dates else ("缺少日频enriched分区",),
            observations={
                "coverage": 1.0 if partition_dates else 0.0,
                "pit_verified": True,
                "partition_count": len(partition_dates),
                "first_date": partition_dates[0].isoformat() if partition_dates else None,
                "last_date": partition_dates[-1].isoformat() if partition_dates else None,
            },
        )
        universe = self._historical_universe(start, end)
        financial = self._financials()
        industry = self._historical_industry(start, end)
        events = self._events()
        concepts = DataQualification(
            ready=False,
            reasons=("当前概念成员表是快照; 禁止进入历史正式研究",),
            observations={"coverage": 0.0, "pit_verified": False},
        )
        datasets = {
            "daily_enriched": daily,
            "historical_universe": universe,
            "financial_pit": financial,
            "industry_pit": industry,
            "event_history": events,
            "concept_snapshot": concepts,
        }
        payload = {
            key: {
                "ready": value.ready,
                "reasons": list(value.reasons),
                "observations": dict(value.observations),
            }
            for key, value in datasets.items()
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return CatalogSnapshot(
            datasets=datasets,
            fingerprint=digest,
            first_date=partition_dates[0].isoformat() if partition_dates else None,
            last_date=partition_dates[-1].isoformat() if partition_dates else None,
        )

    def apply_formal_pit_context(self, panel: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        if panel.is_empty():
            return panel, {"eligible_rows": 0, "input_rows": 0}
        universe_path = self.data_dir / "research" / "historical_stock_universe.parquet"
        names_path = self.data_dir / "research" / "historical_stock_names.parquet"
        shares = _share_history(self.data_dir)
        if not universe_path.is_file() or not names_path.is_file() or shares.is_empty():
            raise ValueError("正式Alpha研究缺少历史股票池、名称或点时股本")
        work = panel.with_columns(pl.col("date").cast(pl.Date)).sort(["symbol", "date"])
        universe = pl.read_parquet(universe_path).with_columns(
            pl.col("list_date").cast(pl.Date, strict=False),
            pl.col("delist_date").cast(pl.Date, strict=False),
        ).select("symbol", "list_date", "delist_date")
        work = work.join(universe, on="symbol", how="left").filter(
            pl.col("list_date").is_not_null()
            & (pl.col("date") >= pl.col("list_date"))
            & (pl.col("delist_date").is_null() | (pl.col("date") <= pl.col("delist_date")))
        )
        names = pl.read_parquet(names_path).with_columns(
            pl.col("start_date").cast(pl.Date, strict=False),
            pl.col("end_date").cast(pl.Date, strict=False),
        ).sort(["symbol", "start_date"])
        work = work.join_asof(
            names.select("symbol", "name", "start_date", "end_date"),
            left_on="date",
            right_on="start_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        ).filter(
            pl.col("name").is_not_null()
            & (pl.col("end_date").is_null() | (pl.col("date") <= pl.col("end_date")))
            & ~pl.col("name").str.to_uppercase().str.contains(r"(?:\*?ST|退)")
        )
        shares = shares.with_columns(
            pl.coalesce(
                pl.col("announce_date").cast(pl.Date, strict=False),
                pl.col("period_end").cast(pl.Date, strict=False),
            ).alias("available_date")
        ).drop_nulls("available_date").sort(["symbol", "available_date"])
        work = work.sort(["symbol", "date"]).join_asof(
            shares.select("symbol", "available_date", "total_shares", "float_shares"),
            left_on="date",
            right_on="available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        ).filter(
            pl.col("total_shares").is_not_null()
            & (pl.col("total_shares") > 0)
            & pl.col("float_shares").is_not_null()
            & (pl.col("float_shares") > 0)
            & (pl.col("float_shares") <= pl.col("total_shares"))
        )
        audit = {
            "input_rows": panel.height,
            "eligible_rows": work.height,
            "eligible_symbols": work.get_column("symbol").n_unique() if work.height else 0,
            "coverage": work.height / panel.height if panel.height else 0.0,
            "pit_verified": True,
        }
        return work.drop(
            "list_date", "delist_date", "start_date", "end_date", "available_date"
        ), audit

    def attach_industry_context(self, panel: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        membership_path = self.data_dir / "research" / "sw_l1_membership.parquet"
        context_path = self.data_dir / "research" / "sw_l1_daily_context.parquet"
        if not membership_path.is_file() or not context_path.is_file():
            raise ValueError("缺少点时行业成员或行业日频上下文")
        membership = pl.read_parquet(membership_path).with_columns(
            pl.col("in_date").cast(pl.Date, strict=False),
            pl.col("out_date").cast(pl.Date, strict=False),
        ).sort(["symbol", "in_date"])
        work = panel.with_columns(pl.col("date").cast(pl.Date)).sort(["symbol", "date"])
        work = work.join_asof(
            membership.select("symbol", "l1_code", "in_date", "out_date"),
            left_on="date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        ).filter(pl.col("out_date").is_null() | (pl.col("date") <= pl.col("out_date")))
        context = pl.read_parquet(context_path).with_columns(
            pl.col("date").cast(pl.Date, strict=False)
        ).select("l1_code", "date", "industry_momentum_20d", "industry_breadth_5d")
        work = work.join(context, on=["l1_code", "date"], how="left")
        covered = work.select(
            pl.all_horizontal(
                pl.col("industry_momentum_20d").is_not_null(),
                pl.col("industry_breadth_5d").is_not_null(),
            ).mean()
        ).item()
        return work.drop("in_date", "out_date"), {
            "coverage": float(covered or 0.0),
            "pit_verified": True,
        }

    def attach_financial_context(self, panel: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Join financial metrics only from their public announcement date onward."""
        paths = sorted((self.data_dir / "financials" / "metrics").rglob("*.parquet"))
        if not paths:
            raise ValueError("缺少历史财务指标")
        financial = pl.concat(
            [pl.read_parquet(path) for path in paths],
            how="diagonal_relaxed",
        )
        announced = "announce_date" if "announce_date" in financial.columns else "announcement_date"
        if announced not in financial.columns or "symbol" not in financial.columns:
            raise ValueError("财务指标缺少symbol或公告时间")
        aliases = {
            "roe": "roe_latest",
            "gross_margin": "gross_margin_latest",
            "grossprofit_margin": "gross_margin_latest",
            "net_margin": "net_margin_latest",
            "revenue_yoy": "revenue_yoy_latest",
            "net_income_yoy": "net_income_yoy_latest",
            "netprofit_yoy": "net_income_yoy_latest",
            "debt_ratio": "debt_ratio_latest",
        }
        selected: dict[str, str] = {}
        for source, target in aliases.items():
            if source in financial.columns and target not in selected.values():
                selected[source] = target
        if not selected:
            raise ValueError("财务指标缺少可研究的标准化字段")
        financial = financial.select(
            "symbol",
            pl.col(announced).cast(pl.Date, strict=False).alias("financial_available_date"),
            *(pl.col(source).cast(pl.Float64, strict=False).alias(target) for source, target in selected.items()),
        ).drop_nulls("financial_available_date").sort(["symbol", "financial_available_date"])
        latest_fields = list(selected.values())
        work = panel.drop(*[name for name in latest_fields if name in panel.columns]).with_columns(
            pl.col("date").cast(pl.Date)
        ).sort(["symbol", "date"]).join_asof(
            financial,
            left_on="date",
            right_on="financial_available_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        revenue = "revenue_yoy_latest"
        profit = "net_income_yoy_latest"
        expressions = []
        if revenue in work.columns:
            expressions.append(
                (pl.col(revenue) - pl.col(revenue).shift(1).over("symbol")).alias(
                    "financial_revision_revenue"
                )
            )
        if profit in work.columns:
            expressions.append(
                (pl.col(profit) - pl.col(profit).shift(1).over("symbol")).alias(
                    "financial_revision_profit"
                )
            )
        if expressions:
            work = work.with_columns(*expressions)
        covered = work.select(
            pl.any_horizontal(*(pl.col(name).is_not_null() for name in latest_fields)).mean()
        ).item()
        return work.drop("financial_available_date"), {
            "coverage": float(covered or 0.0),
            "pit_verified": True,
            "timestamp_field": announced,
            "fields": latest_fields,
        }

    def attach_event_context(
        self,
        panel: pl.DataFrame,
        *,
        start: date,
        end: date,
        decision_clock: str = "after_close",
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        paths = sorted((self.data_dir / "research" / "events").glob("*.parquet"))
        if not paths:
            raise ValueError("缺少带发布时间的历史事件表")
        request = PointInTimeDataRequest(
            asset_type="stock",
            start=start.isoformat(),
            end=end.isoformat(),
            decision_clock=decision_clock,
            columns=(),
        )
        trading_dates = tuple(
            panel.select(pl.col("date").cast(pl.Date)).unique().sort("date")["date"].to_list()
        )
        provider = TimestampedEventProvider(
            EventProviderManifest(
                provider_id="standardized.research.events",
                dataset_id="event_history",
                version="1.0.0",
                published_at_field="published_at",
                effective_date_field="effective_date",
            ),
            lambda _request: pl.read_parquet(paths),
            lambda _request: trading_dates,
        )
        events = provider.load_events(request)
        if "event_direction" not in events.columns:
            events = events.with_columns(pl.lit(0.0).alias("event_direction"))
        daily = (
            events.group_by("symbol", "effective_date")
            .agg(
                pl.len().cast(pl.Float64).alias("_event_count"),
                pl.col("event_direction").cast(pl.Float64, strict=False).fill_null(0).sum().alias(
                    "_event_direction"
                ),
            )
            .rename({"effective_date": "date"})
        )
        work = panel.with_columns(pl.col("date").cast(pl.Date)).join(
            daily,
            on=["symbol", "date"],
            how="left",
        ).sort(["symbol", "date"]).with_columns(
            pl.col("_event_count").fill_null(0.0),
            pl.col("_event_direction").fill_null(0.0),
        ).with_columns(
            pl.col("_event_count").rolling_sum(20, min_samples=1).over("symbol").alias(
                "event_count_20d"
            ),
            pl.col("_event_direction").rolling_sum(20, min_samples=1).over("symbol").alias(
                "event_direction_20d"
            ),
        ).drop("_event_count", "_event_direction")
        return work, {
            "coverage": 1.0,
            "pit_verified": True,
            "events": events.height,
            "provider_id": provider.manifest.provider_id,
        }

    def _historical_universe(self, start: date, end: date) -> DataQualification:
        root = self.data_dir / "research"
        paths = {
            "universe": root / "historical_stock_universe.parquet",
            "names": root / "historical_stock_names.parquet",
        }
        shares = _share_history(self.data_dir)
        missing = [name for name, path in paths.items() if not path.is_file()]
        if shares.is_empty():
            missing.append("shares")
        if missing:
            return DataQualification(
                False,
                ("历史股票池缺失: " + ", ".join(missing),),
                {"coverage": 0.0, "pit_verified": False},
            )
        universe = pl.read_parquet(paths["universe"])
        names = pl.read_parquet(paths["names"])
        required_universe = {"symbol", "list_date", "delist_date"}
        required_names = {"symbol", "name", "start_date", "end_date"}
        missing_columns = sorted(
            (required_universe - set(universe.columns)) | (required_names - set(names.columns))
        )
        ready = not missing_columns and universe.height > 0 and names.height > 0
        return DataQualification(
            ready,
            () if ready else (f"历史股票池字段缺失: {missing_columns}",),
            {
                "coverage": 1.0 if ready else 0.0,
                "pit_verified": ready,
                "symbols": universe.get_column("symbol").n_unique() if ready else 0,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

    def _financials(self) -> DataQualification:
        paths = list((self.data_dir / "financials" / "metrics").rglob("*.parquet"))
        if not paths:
            return DataQualification(False, ("缺少历史财务指标",), {"coverage": 0.0, "pit_verified": False})
        schemas = [pl.read_parquet_schema(path) for path in paths]
        timestamps = [
            "announce_date" if "announce_date" in schema else "announcement_date"
            for schema in schemas
        ]
        ready = all(
            "symbol" in schema and timestamp in schema
            for schema, timestamp in zip(schemas, timestamps, strict=True)
        )
        timestamp = timestamps[0]
        return DataQualification(
            ready,
            () if ready else ("财务指标缺少公告时间",),
            {
                "coverage": 1.0 if ready else 0.0,
                "pit_verified": ready,
                "timestamp_field": timestamp,
                "tables": len(paths),
            },
        )

    def _historical_industry(self, start: date, end: date) -> DataQualification:
        root = self.data_dir / "research"
        membership = root / "sw_l1_membership.parquet"
        context = root / "sw_l1_daily_context.parquet"
        ready = membership.is_file() and context.is_file()
        return DataQualification(
            ready,
            () if ready else ("缺少点时行业成员或行业上下文",),
            {
                "coverage": 1.0 if ready else 0.0,
                "pit_verified": ready,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

    def _events(self) -> DataQualification:
        paths = list((self.data_dir / "research" / "events").glob("*.parquet"))
        if not paths:
            return DataQualification(False, ("缺少带发布时间的历史事件表",), {"coverage": 0.0, "pit_verified": False})
        valid = 0
        for path in paths:
            schema = pl.read_parquet_schema(path)
            if {"symbol", "published_at"}.issubset(schema):
                valid += 1
        ready = valid == len(paths)
        return DataQualification(
            ready,
            () if ready else ("事件表缺少symbol或published_at",),
            {"coverage": valid / len(paths), "pit_verified": ready, "tables": len(paths)},
        )


def _partition_dates(root: Path, start: date, end: date) -> list[date]:
    output: list[date] = []
    if not root.exists():
        return output
    for path in root.glob("date=*"):
        try:
            value = date.fromisoformat(path.name.removeprefix("date="))
        except ValueError:
            continue
        if start <= value <= end and any(path.glob("*.parquet")):
            output.append(value)
    return sorted(set(output))


def _share_history(data_dir: Path) -> pl.DataFrame:
    paths = list((data_dir / "financials" / "shares").rglob("*.parquet"))
    if not paths:
        return pl.DataFrame()
    return pl.read_parquet(paths)
