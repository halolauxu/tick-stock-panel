"""Collect and verify the frozen evidence for the HFC quota repricing study.

This stage deliberately does not load security prices.  It persists regulator
PDFs, their hashes, the parsed quota table, issuer mappings, and the first
point-in-time earnings event that can activate the later return experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pymupdf


WINDOW_START = date(2025, 8, 26)
WINDOW_END = date(2026, 8, 26)

SOURCES: dict[int, dict[str, Any]] = {
    2025: {
        "notice_date": "2024-12-27",
        "notice_url": (
            "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk05/202501/"
            "t20250106_1099999.html"
        ),
        "pdf_url": (
            "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk05/202501/"
            "W020250106342729702074.pdf"
        ),
        "sha256": "0763da2fc16b9506ecca2fe22672eea749529630e6fbe2b1f6b07347cf7abf73",
        "producer_count": 34,
        "production_total_tonnes": 791_882,
    },
    2026: {
        "notice_date": "2025-12-30",
        "notice_url": (
            "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk05/202512/"
            "t20251230_1139340.html"
        ),
        "pdf_url": (
            "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk05/202512/"
            "W020251230595804836415.pdf"
        ),
        "sha256": "390f37f41b285bcdc93886cc886a9794be6df1e0aa6984fc038a135f87e98d81",
        "producer_count": 34,
        "production_total_tonnes": 797_845,
    },
}

ISSUER_MAPPINGS: dict[str, dict[str, Any]] = {
    "600160.SH": {
        "issuer": "巨化股份",
        "quota_entities": (
            "浙江衢化氟化学有限公司",
            "浙江巨化股份有限公司电化厂",
            "浙江兰溪巨化氟化学有限公司",
        ),
        "evidence_url": SOURCES[2025]["pdf_url"],
        "mapping_basis": "quota table names the issuer legal entity branch",
    },
    "603379.SH": {
        "issuer": "三美股份",
        "quota_entities": (
            "江苏三美化工有限公司",
            "浙江三美化工股份有限公司",
        ),
        "evidence_url": SOURCES[2025]["pdf_url"],
        "mapping_basis": "quota table names the issuer legal entity and subsidiary",
    },
    "605020.SH": {
        "issuer": "永和股份",
        "quota_entities": (
            "内蒙古永和氟化工有限公司",
            "金华永和氟化工有限公司",
            "邵武永和金塘新材料有限公司",
        ),
        "evidence_url": (
            "https://www.sse.com.cn/disclosure/listedinfo/announcement/c/new/"
            "2025-07-10/605020_20250710_0IQW.pdf"
        ),
        "mapping_basis": "consolidated subsidiaries",
    },
    "600378.SH": {
        "issuer": "昊华科技",
        "quota_entities": (
            "太仓中化环保化工有限公司",
            "中化蓝天氟材料有限公司",
            "江西兴氟中蓝新材料有限公司",
            "兴国兴氟化工有限公司",
            "陕西中化蓝天化工新材料有限公司",
        ),
        "evidence_url": (
            "https://www.sse.com.cn/disclosure/listedinfo/announcement/c/new/"
            "2024-04-30/600378_20240430_XHEF.pdf"
        ),
        "mapping_basis": "acquired and consolidated Sinochem Lantian group",
    },
    "002915.SZ": {
        "issuer": "中欣氟材",
        "quota_entities": ("江西中欣埃克盛新材料有限公司",),
        "evidence_url": (
            "https://static.cninfo.com.cn/finalpage/2025-04-22/1223197589.PDF"
        ),
        "mapping_basis": "consolidated subsidiary",
    },
}

CONTROL_SYMBOLS = (
    "002326.SZ",
    "002407.SZ",
    "603310.SH",
    "603505.SH",
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def fetch_verified_pdf(year: int, root: Path) -> tuple[Path, dict[str, Any]]:
    spec = SOURCES[year]
    path = root / "source" / f"mee-{year}-hfc-quota.pdf"
    if path.is_file():
        payload = path.read_bytes()
    else:
        response = httpx.get(spec["pdf_url"], timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        payload = response.content
        _atomic_write(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(
            f"{year} HFC quota PDF hash mismatch: {digest} != {spec['sha256']}"
        )
    return path, {
        "year": year,
        "notice_date": spec["notice_date"],
        "notice_url": spec["notice_url"],
        "pdf_url": spec["pdf_url"],
        "sha256": digest,
        "bytes": len(payload),
    }


def parse_production_quota(year: int, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = {"province": "", "sequence": "", "quota_entity": ""}
    document = pymupdf.open(path)
    try:
        for page_number, page in enumerate(document, start=1):
            for table in page.find_tables().tables:
                values = table.extract()
                if not values or len(values[0]) != 6:
                    continue
                header = "".join(cell or "" for cell in values[0])
                if "HFCs" not in header or "生产配额" not in header:
                    continue
                for raw in values[1:]:
                    cleaned = [
                        (cell or "").replace("\n", "").replace(" ", "").strip()
                        for cell in raw
                    ]
                    if cleaned[0]:
                        current["province"] = cleaned[0]
                    if cleaned[1]:
                        current["sequence"] = cleaned[1]
                    if cleaned[2]:
                        current["quota_entity"] = cleaned[2]
                    product = cleaned[3]
                    if not product.startswith("HFC-"):
                        continue
                    rows.append(
                        {
                            "quota_year": year,
                            "page": page_number,
                            **current,
                            "product": product,
                            "production_quota_tonnes": int(
                                cleaned[4].replace(",", "")
                            ),
                            "domestic_quota_tonnes": int(
                                cleaned[5].replace(",", "")
                            ),
                        }
                    )
    finally:
        document.close()
    return rows


def validate_quota_rows(year: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = SOURCES[year]
    producers = {row["sequence"] for row in rows}
    total = sum(row["production_quota_tonnes"] for row in rows)
    checks = {
        "producer_count_matches": len(producers) == expected["producer_count"],
        "production_total_matches": total == expected["production_total_tonnes"],
        "all_entities_present": all(row["quota_entity"] for row in rows),
        "all_products_present": all(row["product"] for row in rows),
    }
    return {
        "row_count": len(rows),
        "producer_count": len(producers),
        "production_total_tonnes": total,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_issuer_mapping(rows: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    year_rows = [row for row in rows if row["quota_year"] == year]
    result: list[dict[str, Any]] = []
    for symbol, spec in ISSUER_MAPPINGS.items():
        aliases = set(spec["quota_entities"])
        matched = [row for row in year_rows if row["quota_entity"] in aliases]
        result.append(
            {
                "symbol": symbol,
                "issuer": spec["issuer"],
                "quota_year": year,
                "mapping_basis": spec["mapping_basis"],
                "evidence_url": spec["evidence_url"],
                "declared_entities": sorted(aliases),
                "matched_entities": sorted(
                    {row["quota_entity"] for row in matched}
                ),
                "matched_products": sorted({row["product"] for row in matched}),
                "matched_rows": len(matched),
                "passed": bool(matched),
            }
        )
    return result


def qualify_forecast_reason(reason: str | None) -> dict[str, bool]:
    text = re.sub(r"\s+", "", reason or "")
    quota = bool(re.search(r"(?:生产)?配额(?:制|管理|政策|约束|缩减|削减)", text))
    price = bool(
        re.search(r"(?:价格|均价).{0,20}(?:上涨|上行|走高|高位)", text)
        or re.search(r"(?:上涨|上行|走高|高位).{0,20}(?:价格|均价)", text)
    )
    earnings = bool(re.search(r"(?:毛利|盈利|利润).{0,20}(?:增长|提升|改善|上升)", text))
    return {
        "quota_constraint": quota,
        "price_increase": price,
        "earnings_improvement": earnings,
        "qualified": quota and price and earnings,
    }


def load_fast_events(data_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((data_dir / "event_data" / "forecast").glob("year=*/part.parquet"))
    if not paths:
        raise ValueError("forecast event partitions are required")
    frame = pl.concat(
        [pl.read_parquet(path) for path in paths], how="diagonal_relaxed"
    ).filter(
        pl.col("symbol").is_in(list(ISSUER_MAPPINGS))
        & pl.col("ann_date").is_between(WINDOW_START, WINDOW_END, closed="both")
    )
    events: list[dict[str, Any]] = []
    for row in frame.sort(["ann_date", "symbol"]).to_dicts():
        qualification = qualify_forecast_reason(row.get("change_reason"))
        if not qualification["qualified"]:
            continue
        events.append(
            {
                "symbol": row["symbol"],
                "ann_date": row["ann_date"].isoformat(),
                "period_end": row["period_end"].isoformat(),
                "forecast_type": row.get("type"),
                "p_change_min": row.get("p_change_min"),
                "p_change_max": row.get("p_change_max"),
                "reason_sha256": hashlib.sha256(
                    (row.get("change_reason") or "").encode("utf-8")
                ).hexdigest(),
                "qualification": qualification,
            }
        )
    return events


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def collect(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "hfc_quota"
    source_meta: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    year_audits: dict[str, Any] = {}
    for year in sorted(SOURCES):
        path, metadata = fetch_verified_pdf(year, root)
        rows = parse_production_quota(year, path)
        audit = validate_quota_rows(year, rows)
        source_meta.append(metadata)
        all_rows.extend(rows)
        year_audits[str(year)] = audit

    quota_path = root / "production_quota.parquet"
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(all_rows, infer_schema_length=None).write_parquet(quota_path)

    issuer_mapping = []
    for year in sorted(SOURCES):
        issuer_mapping.extend(build_issuer_mapping(all_rows, year))
    fast_events = load_fast_events(data_dir)
    first_event = fast_events[0] if fast_events else None
    checks = {
        "all_regulator_tables_pass": all(
            audit["passed"] for audit in year_audits.values()
        ),
        "all_candidate_mappings_pass": all(
            row["passed"] for row in issuer_mapping
        ),
        "fast_event_exists": first_event is not None,
        "frozen_first_event_matches": bool(
            first_event
            and first_event["symbol"] == "605020.SH"
            and first_event["ann_date"] == "2025-10-09"
        ),
    }
    payload = {
        "schema_version": "p0-hfc-quota-evidence-v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "sources": source_meta,
        "quota_table": {
            "path": str(quota_path),
            "sha256": hashlib.sha256(quota_path.read_bytes()).hexdigest(),
            "rows": len(all_rows),
            "year_audits": year_audits,
        },
        "issuer_mapping": issuer_mapping,
        "control_symbols": list(CONTROL_SYMBOLS),
        "fast_events": fast_events,
        "activation_event": first_event,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "DATA_GAP",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, default=_json_default
    ).encode("utf-8")
    _atomic_write(output, encoded)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": checks,
                "quota_years": year_audits,
                "mapped_issuers": sum(row["passed"] for row in issuer_mapping),
                "mapping_rows": len(issuer_mapping),
                "qualifying_fast_events": len(fast_events),
                "activation_event": first_event,
                "output": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_hfc_quota_evidence_v1.json"),
    )
    args = parser.parse_args()
    collect(args.data_dir, args.output)


if __name__ == "__main__":
    main()
