"""Collect point-in-time SW L1 membership and derive daily industry context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from app import secrets_store
from app.plugins.tushare.client import TushareClient

CLASSIFY_FIELDS = (
    "index_code",
    "industry_name",
    "parent_code",
    "level",
    "industry_code",
    "is_pub",
    "src",
)
MEMBER_FIELDS = (
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "ts_code",
    "name",
    "in_date",
    "out_date",
    "is_new",
)


def collect_membership(output_dir: Path) -> pl.DataFrame:
    token = secrets_store.get_env_backed_secret("tushare_api_key", "TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    client = TushareClient(token)
    try:
        industries = client.query(
            "index_classify", {"src": "SW2021", "level": "L1"}, CLASSIFY_FIELDS
        )
        rows: list[dict] = []
        for index, industry in enumerate(industries, start=1):
            code = str(industry.get("index_code") or "").strip()
            if not code:
                continue
            member_count = 0
            for membership_state in ("Y", "N"):
                members = client.query(
                    "index_member_all",
                    {"l1_code": code, "is_new": membership_state},
                    MEMBER_FIELDS,
                )
                rows.extend(members)
                member_count += len(members)
            print(
                f"industry={index}/{len(industries)} code={code} rows={member_count}",
                flush=True,
            )
    finally:
        client.close()

    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.col("name").cast(pl.Utf8).alias("name"),
            pl.col("l1_code").cast(pl.Utf8),
            pl.col("l1_name").cast(pl.Utf8),
            pl.col("in_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False)
            .fill_null(pl.date(1990, 1, 1)),
            pl.col("out_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("is_new").cast(pl.Utf8),
        )
        .drop_nulls(["symbol", "l1_code"])
        .unique(subset=["symbol", "l1_code", "in_date"], keep="last")
        .sort(["symbol", "in_date"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / "sw_l1_membership.parquet")
    (output_dir / "sw_l1_catalog.json").write_text(
        json.dumps(industries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame


def build_daily_context(data_dir: Path, output_dir: Path, membership: pl.DataFrame) -> pl.DataFrame:
    enriched_glob = str(data_dir / "kline_daily_enriched" / "**" / "*.parquet")
    daily = (
        pl.scan_parquet(enriched_glob)
        .select("symbol", "date", "close")
        .collect(engine="streaming")
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias(
                "change_pct"
            )
        )
        .filter(pl.col("change_pct").is_finite() & (pl.col("change_pct").abs() <= 0.25))
    )
    joined = (
        daily.join_asof(
            membership.select("symbol", "l1_code", "l1_name", "in_date", "out_date").sort(
                ["symbol", "in_date"]
            ),
            left_on="date",
            right_on="in_date",
            by="symbol",
            strategy="backward",
        )
        .filter(pl.col("l1_code").is_not_null())
        .filter(pl.col("out_date").is_null() | (pl.col("date") <= pl.col("out_date")))
    )
    context = (
        joined.group_by("l1_code", "l1_name", "date")
        .agg(
            pl.col("change_pct").median().alias("industry_return_1d"),
            (pl.col("change_pct") > 0).mean().alias("industry_breadth"),
            pl.len().alias("industry_member_count"),
        )
        .sort(["l1_code", "date"])
        .with_columns(
            (
                (pl.col("industry_return_1d") + 1.0)
                .log()
                .rolling_sum(window_size=20, min_samples=15)
                .over("l1_code")
                .exp()
                - 1.0
            ).alias("industry_momentum_20d"),
            pl.col("industry_breadth")
            .rolling_mean(window_size=5, min_samples=3)
            .over("l1_code")
            .alias("industry_breadth_5d"),
        )
    )
    context.write_parquet(output_dir / "sw_l1_daily_context.parquet")
    return context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/app/data/research"))
    args = parser.parse_args()
    membership = collect_membership(args.output_dir)
    context = build_daily_context(args.data_dir, args.output_dir, membership)
    print(
        json.dumps(
            {
                "membership_rows": membership.height,
                "membership_symbols": membership["symbol"].n_unique(),
                "industry_days": context.height,
                "industries": context["l1_code"].n_unique(),
                "start": str(context["date"].min()),
                "end": str(context["date"].max()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
