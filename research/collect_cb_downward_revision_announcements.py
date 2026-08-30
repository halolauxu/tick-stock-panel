"""Collect point-in-time CB conversion-price downward-revision announcements."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(RESEARCH))

import collect_equity_incentive_announcements as common  # noqa: E402

SEARCH_KEY = "向下修正 转股价格"
PAGE_SIZE = common.PAGE_SIZE
MAX_PAGES = common.MAX_PAGES


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _payload(year: int, page: int) -> dict[str, str]:
    return {
        "pageNum": str(page),
        "pageSize": str(PAGE_SIZE),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": SEARCH_KEY,
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{year}-01-01~{year}-12-31",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "false",
    }


def fetch_year(fetch_json, year: int) -> list[dict[str, Any]]:
    first, total, reported_pages = common._page_rows(
        fetch_json(_payload(year, 1))
    )
    expected_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE if total else 0
    accepted = {expected_pages, max(expected_pages - 1, 0)}
    if reported_pages not in accepted or expected_pages > MAX_PAGES:
        raise ValueError("CNInfo downward-revision page count is invalid")
    rows = list(first)
    for page in range(2, expected_pages + 1):
        current, current_total, current_reported = common._page_rows(
            fetch_json(_payload(year, page))
        )
        if current_total != total or current_reported not in accepted:
            raise ValueError("CNInfo downward-revision pagination changed")
        rows.extend(current)
    if len(rows) != total:
        raise ValueError(
            f"CNInfo pagination incomplete: expected {total}, got {len(rows)}"
        )
    return rows


def _phase(title: str) -> str | None:
    compact = title.replace(" ", "")
    if any(token in compact for token in ("不向下修正", "暂不下修", "不下修")):
        return None
    downward = "向下修正" in compact or "下修" in compact
    if not downward or not ("转股价格" in compact or "转股价" in compact):
        return None
    if "董事会" in compact and any(token in compact for token in ("提议", "建议")):
        return "proposal"
    return "implemented"


def normalize(rows: list[dict[str, Any]], year: int) -> pl.DataFrame:
    frame = common.normalize(rows, year)
    if frame.is_empty():
        return frame
    return (
        frame.with_columns(
            pl.col("title")
            .map_elements(_phase, return_dtype=pl.String)
            .alias("phase")
        )
        .drop_nulls("phase")
        .unique(
            subset=["ann_date", "symbol", "phase"],
            keep="first",
            maintain_order=True,
        )
        .sort(["ann_date", "symbol", "phase", "announcement_id"])
    )


def collect_year(fetch_json, root: Path, year: int) -> dict[str, Any]:
    rows = fetch_year(fetch_json, year)
    frame = normalize(rows, year)
    if frame.is_empty():
        raise ValueError(f"CNInfo returned no usable downward revisions for {year}")
    path = root / f"year={year}" / "part.parquet"
    _atomic_write(frame, path)
    by_phase = {
        row["phase"]: row["len"]
        for row in frame.group_by("phase").len().sort("phase").to_dicts()
    }
    return {
        "year": year,
        "path": str(path),
        "raw_rows": len(rows),
        "events": frame.height,
        "symbols": frame["symbol"].n_unique(),
        "announcement_days": frame["ann_date"].n_unique(),
        "by_phase": by_phase,
        "first_ann_date": frame["ann_date"].min(),
        "last_ann_date": frame["ann_date"].max(),
    }


def run(data_dir: Path, start_year: int, end_year: int) -> dict[str, Any]:
    if start_year > end_year or end_year - start_year > 1:
        raise ValueError("each collection run is bounded to at most two years")
    client = common.CNInfoClient()
    root = data_dir / "event_data" / "cb_downward_revision"
    results = [
        collect_year(client.fetch, root, year)
        for year in range(start_year, end_year + 1)
    ]
    payload = {
        "dataset": "cninfo_cb_conversion_price_downward_revision",
        "source": common.API_URL,
        "search_key": SEARCH_KEY,
        "outcome_fields_persisted": False,
        "start_year": start_year,
        "end_year": end_year,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
