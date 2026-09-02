"""Match inferred regular index additions to contemporaneous official notices.

This audit is metadata-only.  It deliberately refuses to read market prices or
returns.  Adjacent-month membership differences are admitted only when both
CSI 300 and CSI 500 counts match the official announcement for that cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import polars as pl

PRICE_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "return",
    "future_return",
    "forward_return",
    "net_return",
}
DETAIL_URL = (
    "https://www.csindex.com.cn/csindex-home/announcement/"
    "queryAnnouncementById?id={notice_id}"
)
MIN_MATCHED_CYCLES = 22


@dataclass(frozen=True)
class Notice:
    cycle_month: date
    announcement_date: date
    effective_date: date
    notice_id: int
    expected_csi300_additions: int
    expected_csi500_additions: int


NOTICES = (
    Notice(date(2013, 12, 1), date(2013, 12, 2), date(2013, 12, 16), 3470, 21, 50),
    Notice(date(2014, 6, 1), date(2014, 6, 3), date(2014, 6, 16), 6765, 26, 50),
    Notice(date(2014, 12, 1), date(2014, 12, 1), date(2014, 12, 15), 6779, 22, 50),
    Notice(date(2015, 6, 1), date(2015, 6, 1), date(2015, 6, 15), 3272, 18, 50),
    Notice(date(2015, 12, 1), date(2015, 11, 30), date(2015, 12, 14), 4275, 20, 50),
    Notice(date(2016, 6, 1), date(2016, 5, 30), date(2016, 6, 13), 3605, 24, 50),
    Notice(date(2016, 12, 1), date(2016, 11, 28), date(2016, 12, 12), 4409, 30, 50),
    Notice(date(2017, 6, 1), date(2017, 5, 31), date(2017, 6, 12), 12583, 30, 50),
    Notice(date(2017, 12, 1), date(2017, 11, 27), date(2017, 12, 11), 12589, 24, 50),
    Notice(date(2018, 6, 1), date(2018, 5, 28), date(2018, 6, 11), 12708, 27, 50),
    Notice(date(2018, 12, 1), date(2018, 12, 3), date(2018, 12, 17), 12807, 24, 50),
    Notice(date(2019, 6, 1), date(2019, 6, 3), date(2019, 6, 17), 12934, 19, 50),
    Notice(date(2019, 12, 1), date(2019, 12, 2), date(2019, 12, 16), 13044, 16, 50),
    Notice(date(2020, 6, 1), date(2020, 6, 1), date(2020, 6, 15), 13130, 21, 50),
    Notice(date(2020, 12, 1), date(2020, 11, 27), date(2020, 12, 14), 13247, 26, 50),
    Notice(date(2021, 6, 1), date(2021, 5, 28), date(2021, 6, 11), 12470, 25, 50),
    Notice(date(2021, 12, 1), date(2021, 11, 26), date(2021, 12, 10), 13888, 28, 50),
    Notice(date(2022, 6, 1), date(2022, 5, 27), date(2022, 6, 10), 14223, 28, 50),
    Notice(date(2022, 12, 1), date(2022, 11, 25), date(2022, 12, 9), 14497, 15, 50),
    Notice(date(2023, 6, 1), date(2023, 5, 26), date(2023, 6, 9), 14796, 9, 50),
    Notice(date(2023, 12, 1), date(2023, 11, 24), date(2023, 12, 8), 15044, 14, 50),
    Notice(date(2024, 6, 1), date(2024, 5, 31), date(2024, 6, 14), 15267, 12, 50),
    Notice(date(2024, 12, 1), date(2024, 11, 29), date(2024, 12, 13), 15471, 16, 50),
    Notice(date(2025, 6, 1), date(2025, 5, 30), date(2025, 6, 13), 15690, 7, 50),
    Notice(date(2025, 12, 1), date(2025, 11, 28), date(2025, 12, 12), 3006000, 11, 50),
    Notice(date(2026, 6, 1), date(2026, 5, 29), date(2026, 6, 12), 3006137, 19, 50),
)


def normalize_notice_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", "", text)


def extract_official_counts(content: str) -> dict[str, int | None]:
    text = normalize_notice_text(content)
    output: dict[str, int | None] = {}
    for code, label in (("000300.SH", "沪深300"), ("000905.SH", "中证500")):
        match = re.search(rf"{label}指数更换(\d+)只(?:股票|样本|样本股)", text)
        output[code] = int(match.group(1)) if match else None
    return output


def fetch_notice(notice_id: int) -> dict[str, Any]:
    request = urllib.request.Request(
        DETAIL_URL.format(notice_id=notice_id),
        headers={"User-Agent": "tick-stock-panel-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        payload = json.load(response)
    if payload.get("code") != "200" or not isinstance(payload.get("data"), dict):
        raise RuntimeError(f"official notice {notice_id} unavailable")
    return payload["data"]


def audit_notices(
    additions: pl.DataFrame,
    fetcher: Callable[[int], dict[str, Any]] = fetch_notice,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if PRICE_FIELDS & set(additions.columns):
        raise ValueError("price or outcome fields are forbidden in notice audit")
    counts = {
        (row["cycle_month"], row["index_code"]): row["len"]
        for row in additions.group_by(["cycle_month", "index_code"]).len().to_dicts()
    }
    cycle_rows: list[dict[str, Any]] = []
    accepted_cycles: list[date] = []
    notice_hashes: dict[str, str] = {}
    for notice in NOTICES:
        official = fetcher(notice.notice_id)
        official_date = date.fromisoformat(str(official.get("publishDate")))
        official_counts = extract_official_counts(str(official.get("content", "")))
        serialized = json.dumps(official, ensure_ascii=False, sort_keys=True).encode()
        notice_hashes[str(notice.notice_id)] = hashlib.sha256(serialized).hexdigest()
        expected = {
            "000300.SH": notice.expected_csi300_additions,
            "000905.SH": notice.expected_csi500_additions,
        }
        inferred = {
            code: int(counts.get((notice.cycle_month, code), 0)) for code in expected
        }
        checks = {
            "announcement_date_matches": official_date == notice.announcement_date,
            "official_text_counts_match_contract": official_counts == expected,
            "adjacent_membership_counts_match_notice": inferred == expected,
        }
        matched = all(checks.values())
        if matched:
            accepted_cycles.append(notice.cycle_month)
        cycle_rows.append(
            {
                **asdict(notice),
                "official_title": str(official.get("title", "")),
                "official_counts": official_counts,
                "inferred_counts": inferred,
                "checks": checks,
                "matched": matched,
                "source_url": DETAIL_URL.format(notice_id=notice.notice_id),
            }
        )
    metadata = pl.DataFrame(
        [
            {
                "cycle_month": notice.cycle_month,
                "announcement_date": notice.announcement_date,
                "effective_date": notice.effective_date,
                "notice_id": notice.notice_id,
                "notice_url": DETAIL_URL.format(notice_id=notice.notice_id),
            }
            for notice in NOTICES
            if notice.cycle_month in accepted_cycles
        ]
    )
    matched_additions = (
        additions.join(metadata, on="cycle_month", how="inner")
        .with_columns(
            pl.lit("OFFICIAL_COUNT_AND_ADJACENT_MEMBERSHIP").alias("match_level")
        )
        .sort(["cycle_month", "index_code", "symbol"])
    )
    checks = {
        "price_data_absent": not (PRICE_FIELDS & set(matched_additions.columns)),
        "at_least_22_notice_matched_cycles": len(accepted_cycles) >= MIN_MATCHED_CYCLES,
        "announcement_precedes_effective_date": all(
            notice.announcement_date < notice.effective_date for notice in NOTICES
        ),
        "all_admitted_cycles_match_both_indices": all(
            row["matched"]
            for row in cycle_rows
            if row["cycle_month"] in accepted_cycles
        ),
    }
    payload = {
        "status": "NOTICE_MATCH_SUFFICIENT" if all(checks.values()) else "DATA_GAP",
        "price_data_read": False,
        "future_returns_read": False,
        "match_method": "official count plus adjacent-month official membership diff",
        "matched_cycles": len(accepted_cycles),
        "rejected_cycles": len(NOTICES) - len(accepted_cycles),
        "matched_additions": matched_additions.height,
        "checks": checks,
        "cycles": cycle_rows,
        "official_notice_sha256": notice_hashes,
    }
    return matched_additions, payload


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    root = data_dir / "research" / "index_inclusion"
    additions = pl.read_parquet(root / "regular_additions.parquet")
    matched, payload = audit_notices(additions)
    matched_path = root / "notice_matched_additions.parquet"
    temporary = matched_path.with_suffix(".parquet.tmp")
    matched.write_parquet(temporary)
    temporary.replace(matched_path)
    payload = {
        "schema_version": "p0-index-inclusion-notice-audit-v2",
        "contract_frozen": "2026-09-03",
        **payload,
        "artifact": str(matched_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {**payload, "sha256": digest},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/data/research/p0_index_inclusion_notice_audit.json"),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
