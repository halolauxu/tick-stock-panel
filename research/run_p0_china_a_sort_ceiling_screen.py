"""Screen published China-A-Sort long legs for a 50% gross-return ceiling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

SORT_COMMIT = "b7b317eab7d535105dedc4057855ec4c6b7330d2"
CHAR_COMMIT = "ed1935598f8ae9ec4588fbe066e5e08d683c652c"
BASE_SORT_URL = (
    "https://raw.githubusercontent.com/mlfina/China-A-Sort/"
    f"{SORT_COMMIT}/data/output"
)
FILES = {
    "equal_weight": {
        "url": f"{BASE_SORT_URL}/sorted_portfolio_ew/unisort_returns.csv",
        "sha256": "61e80b31282fda317f0361283a549413edbb9c66b07aac2894641a44d02f6ebd",
    },
    "value_weight": {
        "url": f"{BASE_SORT_URL}/sorted_portfolio_vw/unisort_returns.csv",
        "sha256": "fb5fa5d751d2c6b065d893c8ea5eacaa4b5a5756bda796a75d45b270d2fe6740",
    },
    "characteristics": {
        "url": (
            "https://raw.githubusercontent.com/Quantactix/"
            f"ChinaAShareEquityCharacteristics/{CHAR_COMMIT}/char_list.csv"
        ),
        "sha256": "70e55cf63a087e69f1a328d82f13e780c63d03379c95f974df5502706b9648de",
    },
}
DISCOVERY_START = date(2000, 1, 1)
DISCOVERY_END = date(2013, 12, 31)
CONFIRMATION_START = date(2014, 1, 1)
CONFIRMATION_END = date(2018, 12, 31)
PARSE_END = CONFIRMATION_END


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_verified(root: Path, name: str) -> Path:
    specification = FILES[name]
    target = root / f"{name}.csv"
    if not target.exists() or _sha256(target) != specification["sha256"]:
        root.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".csv.tmp")
        request = urllib.request.Request(
            specification["url"], headers={"User-Agent": "tick-stock-panel-research"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            temporary.write_bytes(response.read())
        if _sha256(temporary) != specification["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {name}")
        temporary.replace(target)
    return target


def factor_names(fieldnames: list[str]) -> list[str]:
    buckets: dict[str, set[int]] = defaultdict(set)
    for field in fieldnames:
        match = re.fullmatch(r"(.+)([0-9])", field)
        if match:
            buckets[match.group(1)].add(int(match.group(2)))
    return sorted(name for name, values in buckets.items() if values == set(range(10)))


def read_returns_through(path: Path, cutoff: date) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = factor_names(reader.fieldnames or [])
        for raw in reader:
            row_date = datetime.strptime(raw["date"], "%Y-%m-%d").date()
            if row_date > cutoff:
                break
            row: dict[str, Any] = {"date": row_date}
            for factor in fields:
                row[factor] = [
                    float(raw[f"{factor}{bucket}"])
                    if raw[f"{factor}{bucket}"] != ""
                    else None
                    for bucket in range(10)
                ]
            output.append(row)
    return output


def characteristic_metadata(path: Path) -> dict[str, dict[str, str | None]]:
    output: dict[str, dict[str, str | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            abbreviation = row[2].strip().lower()
            if not abbreviation:
                continue
            output[abbreviation] = {
                "category": row[0].strip() or None,
                "group": row[1].strip() or None,
                "name": row[3].strip() or None,
            }
    return output


def metrics(rows: list[tuple[date, float]]) -> dict[str, Any]:
    if not rows:
        return {
            "months": 0,
            "annualized": None,
            "total_return": None,
            "max_drawdown": None,
            "positive_years": 0,
        }
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    yearly: dict[int, float] = defaultdict(lambda: 1.0)
    for row_date, monthly_return in rows:
        equity *= 1.0 + monthly_return
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        yearly[row_date.year] *= 1.0 + monthly_return
    total_return = equity - 1.0
    annualized = (
        equity ** (12.0 / len(rows)) - 1.0 if equity > 0 and rows else None
    )
    return {
        "months": len(rows),
        "annualized": annualized,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "positive_years": sum(value > 1.0 for value in yearly.values()),
        "yearly": {str(year): value - 1.0 for year, value in sorted(yearly.items())},
    }


def _period(row_date: date) -> str | None:
    if DISCOVERY_START <= row_date <= DISCOVERY_END:
        return "discovery"
    if CONFIRMATION_START <= row_date <= CONFIRMATION_END:
        return "confirmation"
    return None


def build_leg_record(
    factor: str,
    bucket: int,
    datasets: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "factor": factor,
        "direction": "LOW" if bucket == 0 else "HIGH",
        "bucket": bucket,
        "metadata": metadata.get(factor.lower(), {}),
    }
    for weighting, rows in datasets.items():
        leg: dict[str, list[tuple[date, float]]] = defaultdict(list)
        market: dict[str, list[tuple[date, float]]] = defaultdict(list)
        for row in rows:
            period = _period(row["date"])
            values = row.get(factor)
            if period is None or values is None or values[bucket] is None:
                continue
            available = [value for value in values if value is not None]
            if len(available) != 10:
                continue
            leg[period].append((row["date"], values[bucket]))
            market[period].append((row["date"], sum(available) / len(available)))
        record[weighting] = {
            period: metrics(leg[period])
            for period in ("discovery", "confirmation")
        }
        record[f"{weighting}_benchmark"] = {
            period: metrics(market[period])
            for period in ("discovery", "confirmation")
        }
    return record


def evaluate_candidate(record: dict[str, Any]) -> dict[str, Any]:
    ew = record["equal_weight"]
    ew_benchmark = record["equal_weight_benchmark"]
    vw = record["value_weight"]
    vw_benchmark = record["value_weight_benchmark"]
    checks: dict[str, bool] = {}
    for period, min_months, min_years in (
        ("discovery", 120, 10),
        ("confirmation", 48, 4),
    ):
        ew_annual = ew[period].get("annualized")
        ew_market = ew_benchmark[period].get("annualized")
        vw_annual = vw[period].get("annualized")
        vw_market = vw_benchmark[period].get("annualized")
        checks.update(
            {
                f"{period}_ew_annualized_at_least_50pct": (
                    ew_annual is not None and ew_annual >= 0.50
                ),
                f"{period}_ew_excess_at_least_20pp": (
                    ew_annual is not None
                    and ew_market is not None
                    and ew_annual - ew_market >= 0.20
                ),
                f"{period}_vw_annualized_at_least_20pct": (
                    vw_annual is not None and vw_annual >= 0.20
                ),
                f"{period}_vw_excess_at_least_5pp": (
                    vw_annual is not None
                    and vw_market is not None
                    and vw_annual - vw_market >= 0.05
                ),
                f"{period}_ew_drawdown_no_worse_than_50pct": (
                    ew[period].get("max_drawdown") is not None
                    and ew[period]["max_drawdown"] >= -0.50
                ),
                f"{period}_months_sufficient": ew[period].get("months", 0)
                >= min_months,
                f"{period}_positive_years_sufficient": ew[period].get(
                    "positive_years", 0
                )
                >= min_years,
            }
        )
    return {"passed": all(checks.values()), "checks": checks}


def run(data_dir: Path, output: Path) -> dict[str, Any]:
    source_root = data_dir / "research" / "china_a_sort_ceiling"
    paths = {name: fetch_verified(source_root, name) for name in FILES}
    datasets = {
        weighting: read_returns_through(paths[weighting], PARSE_END)
        for weighting in ("equal_weight", "value_weight")
    }
    metadata = characteristic_metadata(paths["characteristics"])
    factors = sorted(
        set.intersection(
            *[
                set(row.keys()) - {"date"}
                for rows in datasets.values()
                for row in rows[:1]
            ]
        )
    )
    records: list[dict[str, Any]] = []
    for factor in factors:
        for bucket in (0, 9):
            record = build_leg_record(factor, bucket, datasets, metadata)
            record["gate"] = evaluate_candidate(record)
            records.append(record)
    candidates = [record for record in records if record["gate"]["passed"]]
    candidates.sort(
        key=lambda record: min(
            record["equal_weight"]["discovery"]["annualized"],
            record["equal_weight"]["confirmation"]["annualized"],
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "p0-china-a-sort-ceiling-screen-v1",
        "contract_frozen": "2026-08-31",
        "source": {
            "sort_commit": SORT_COMMIT,
            "characteristics_commit": CHAR_COMMIT,
            "files": {
                name: {"sha256": specification["sha256"], "url": specification["url"]}
                for name, specification in FILES.items()
            },
        },
        "periods": {
            "discovery": {"start": DISCOVERY_START, "end": DISCOVERY_END},
            "confirmation": {
                "start": CONFIRMATION_START,
                "end": CONFIRMATION_END,
            },
            "numeric_rows_parsed_through": PARSE_END,
            "2019_2021_metrics_computed": False,
        },
        "semantics": {
            "gross_ceiling_only": True,
            "real_account_backtest": False,
            "costs_modeled": False,
            "counts_toward_50pct_goal": False,
        },
        "counts": {
            "factors": len(factors),
            "long_only_legs": len(records),
            "ceiling_candidates": len(candidates),
        },
        "candidates": candidates,
        "all_legs": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    digest = _sha256(output)
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "candidates": candidates,
                "semantics": payload["semantics"],
                "periods": payload["periods"],
                "output": str(output),
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/app/data/research/p0_china_a_sort_ceiling_screen.json"
        ),
    )
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
