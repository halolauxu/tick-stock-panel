"""Bounded seven-day pilot for the Serenity bottleneck hypothesis.

The pilot is deliberately isolated from the panel's production datasets.  It
freezes a 100-company universe, downloads only official CNINFO disclosures for
that universe, measures PDF/OCR storage economics, extracts rule-based evidence
candidates, freezes after-close research selections and settles them from the
next trading-day open.  It never places orders or calls a paid model.

The default mode is prospective.  ``run-historical`` is a deliberately labelled
retrospective engineering sample over the latest seven completed local trading
partitions.  It is not a clean-room Alpha replay because the local concept and
instrument snapshots are not effective-dated.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import duckdb
import httpx
import polars as pl

from app.config import settings
from app.plugins.tushare.client import TushareClient
from app.plugins.tushare.provider import get_api_key

PILOT_VERSION = "1.0.0"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_MAX_DOCUMENTS = 600
DEFAULT_MAX_RAW_BYTES = 2_000_000_000
DEFAULT_MAX_DOCUMENT_BYTES = 50_000_000
DEFAULT_MAX_DOCUMENTS_PER_COMPANY = 6
DEFAULT_MAX_OCR_PAGES = 8
DEFAULT_COST_BPS = 20.0
CNINFO_STOCK_LIST_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_ROOT = "https://static.cninfo.com.cn/"
HORIZONS = (1, 3, 5, 10, 20)
MODELS = ("serenity", "momentum", "random", "chain_equal")


@dataclass(frozen=True)
class ChainSpec:
    id: str
    name: str
    role: str
    size: int
    weights: dict[str, int]
    required_any: tuple[str, ...]


CHAIN_SPECS = (
    ChainSpec(
        id="semiconductor_frontend",
        name="半导体前道设备与材料",
        role="candidate",
        size=34,
        weights={"光刻机": 5, "光刻胶": 5, "第三代半导体": 3, "存储芯片": 2, "芯片概念": 1},
        required_any=("光刻机", "光刻胶", "第三代半导体", "存储芯片"),
    ),
    ChainSpec(
        id="ai_compute_infrastructure",
        name="AI算力互连与液冷",
        role="candidate",
        size=33,
        weights={
            "共封装光学(CPO)": 5,
            "液冷服务器": 5,
            "PCB概念": 3,
            "数据中心(AIDC)": 1,
            "东数西算(算力)": 1,
            "算力租赁": 1,
        },
        required_any=("共封装光学(CPO)", "液冷服务器", "PCB概念"),
    ),
    ChainSpec(
        id="lithium_materials_control",
        name="锂电材料过剩型负对照",
        role="negative_control",
        size=33,
        weights={"固态电池": 5, "钠离子电池": 4, "动力电池回收": 3, "锂电池概念": 1},
        required_any=("固态电池", "钠离子电池", "动力电池回收", "锂电池概念"),
    ),
)

FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "demand_order": ("订单", "合同", "中标", "客户", "销量", "需求", "交付"),
    "capacity_ramp": ("产能", "产量", "良率", "利用率", "扩产", "投产", "爬坡", "建设周期"),
    "supply_concentration": ("供应商", "独家", "唯一", "进口依赖", "国产替代", "市占率"),
    "customer_validation": ("认证", "验证", "定点", "导入", "批量供货", "复购"),
    "product_economics": ("主营业务", "产品收入", "营业收入", "毛利率", "销售成本", "单价"),
    "capex_project": ("项目投资", "固定资产", "在建工程", "环评", "开工", "竣工", "设备到位"),
}
_FACT_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|亿元|万元|元|吨|台|套|条|家|个月|年|天)")
_FACT_STRONG_RE = re.compile(r"独家|唯一|首家|国产替代|进口依赖|批量供货|正式投产")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。\uFF01\uFF1F\uFF1B;])|\n+")


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


@contextlib.contextmanager
def _pilot_lock(root: Path):
    """Prevent overlapping collectors from duplicating API and OCR work."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".pilot.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError) as exc:
            raise RuntimeError("serenity pilot is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _partition_dates(root: Path, *, not_after: date | None = None) -> list[date]:
    dates: list[date] = []
    for child in root.glob("date=*"):
        try:
            value = date.fromisoformat(child.name.removeprefix("date="))
        except ValueError:
            continue
        if not_after is None or value <= not_after:
            dates.append(value)
    return sorted(dates)


def _latest_partition(root: Path, not_after: date) -> tuple[date, Path]:
    dates = _partition_dates(root, not_after=not_after)
    if not dates:
        raise FileNotFoundError(f"no partition in {root} at or before {not_after}")
    latest = dates[-1]
    return latest, root / f"date={latest.isoformat()}" / "part.parquet"


def _historical_decision_dates(data_dir: Path, end_date: date, count: int) -> list[date]:
    """Return the latest explicit local trading partitions, never calendar-day guesses."""
    if count <= 0:
        raise ValueError("trading day count must be positive")
    dates = _partition_dates(data_dir / "kline_daily_enriched", not_after=end_date)
    if len(dates) < count:
        raise RuntimeError(
            f"historical pilot needs {count} trading partitions at or before {end_date}; "
            f"only {len(dates)} are available"
        )
    return dates[-count:]


def _concept_score(concepts: set[str], spec: ChainSpec) -> int:
    return sum(weight for name, weight in spec.weights.items() if name in concepts)


def _select_stratified(rows: list[dict[str, Any]], size: int, chain_id: str) -> list[dict[str, Any]]:
    """Select a deterministic small/mid/large-cap sample without outcome cherry-picking."""
    ordered = sorted(rows, key=lambda row: (float(row["market_cap"]), str(row["symbol"])))
    buckets: dict[str, list[dict[str, Any]]] = {"small": [], "mid": [], "large": []}
    labels = ("small", "mid", "large")
    for index, row in enumerate(ordered):
        bucket_index = min(2, index * 3 // max(1, len(ordered)))
        item = dict(row)
        item["market_cap_bucket"] = labels[bucket_index]
        buckets[labels[bucket_index]].append(item)

    base, remainder = divmod(size, 3)
    targets = {"small": base, "mid": base, "large": base + remainder}
    chosen: list[dict[str, Any]] = []
    for label in labels:
        ranked = sorted(
            buckets[label],
            key=lambda row: (
                -int(row["concept_score"]),
                _stable_hash(PILOT_VERSION, chain_id, str(row["symbol"])),
            ),
        )
        chosen.extend(ranked[: targets[label]])

    if len(chosen) < size:
        chosen_symbols = {str(row["symbol"]) for row in chosen}
        remainder_rows = [row for row in ordered if str(row["symbol"]) not in chosen_symbols]
        remainder_rows.sort(
            key=lambda row: (
                -int(row["concept_score"]),
                _stable_hash(PILOT_VERSION, chain_id, str(row["symbol"])),
            )
        )
        chosen.extend(remainder_rows[: size - len(chosen)])
    return chosen[:size]


def select_universe(data_dir: Path, as_of: date) -> tuple[date, list[dict[str, Any]]]:
    """Freeze the representative 34/33/33 sample from current local snapshots."""
    concepts_path = data_dir / "ext_data" / "ext_gn_ths" / "part.parquet"
    instruments_path = data_dir / "instruments" / "instruments.parquet"
    market_date, daily_path = _latest_partition(data_dir / "kline_daily_enriched", as_of)
    concepts = pl.read_parquet(concepts_path).select(
        pl.col("symbol"), pl.col("股票简称").alias("name"), pl.col("所属概念").alias("concepts")
    )
    instruments = pl.read_parquet(instruments_path).select(
        "symbol", "listing_date", "total_shares"
    )
    daily = pl.read_parquet(daily_path).select("symbol", "close", "amount")
    joined = concepts.join(instruments, on="symbol", how="inner").join(daily, on="symbol", how="inner")

    assigned: dict[str, list[dict[str, Any]]] = {spec.id: [] for spec in CHAIN_SPECS}
    cutoff = market_date - timedelta(days=120)
    for row in joined.iter_rows(named=True):
        name = str(row.get("name") or "")
        if "ST" in name.upper() or not name:
            continue
        try:
            listing_date = date.fromisoformat(str(row["listing_date"]))
            close = float(row["close"])
            amount = float(row["amount"])
            total_shares = float(row["total_shares"])
        except (TypeError, ValueError):
            continue
        market_cap = close * total_shares
        if listing_date > cutoff or not (3 <= close <= 300):
            continue
        if amount < 100_000_000 or market_cap < 2_000_000_000:
            continue
        concept_set = {item.strip() for item in str(row.get("concepts") or "").split(";") if item.strip()}
        matches: list[tuple[int, int, ChainSpec]] = []
        for order, spec in enumerate(CHAIN_SPECS):
            if not concept_set.intersection(spec.required_any):
                continue
            matches.append((_concept_score(concept_set, spec), -order, spec))
        if not matches:
            continue
        score, _, spec = max(matches, key=lambda value: (value[0], value[1]))
        assigned[spec.id].append(
            {
                "symbol": str(row["symbol"]),
                "code": str(row["symbol"]).split(".", 1)[0],
                "name": name,
                "chain_id": spec.id,
                "chain_name": spec.name,
                "chain_role": spec.role,
                "concept_score": score,
                "concepts": ";".join(sorted(concept_set)),
                "market_cap": market_cap,
                "amount": amount,
                "source_as_of": market_date.isoformat(),
            }
        )

    universe: list[dict[str, Any]] = []
    for spec in CHAIN_SPECS:
        candidates = assigned[spec.id]
        if len(candidates) < spec.size:
            raise RuntimeError(f"{spec.name} only has {len(candidates)} eligible candidates")
        selected = _select_stratified(candidates, spec.size, spec.id)
        for rank, row in enumerate(selected, start=1):
            row["sample_rank"] = rank
            universe.append(row)
    if len(universe) != DEFAULT_SAMPLE_SIZE:
        raise RuntimeError(f"pilot universe must contain {DEFAULT_SAMPLE_SIZE} companies")
    return market_date, universe


class PilotStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "pilot.duckdb"
        self.documents_dir = root / "documents"
        self.text_dir = root / "text"
        self.documents_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        self.connection = duckdb.connect(str(self.db_path))
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pilot_meta (
                key VARCHAR PRIMARY KEY,
                value_json VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS universe (
                symbol VARCHAR PRIMARY KEY,
                code VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                chain_id VARCHAR NOT NULL,
                chain_name VARCHAR NOT NULL,
                chain_role VARCHAR NOT NULL,
                sample_rank INTEGER NOT NULL,
                market_cap_bucket VARCHAR NOT NULL,
                concept_score INTEGER NOT NULL,
                concepts VARCHAR NOT NULL,
                market_cap DOUBLE NOT NULL,
                amount DOUBLE NOT NULL,
                source_as_of DATE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS announcements (
                announcement_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                announce_time TIMESTAMP,
                title VARCHAR NOT NULL,
                pdf_url VARCHAR NOT NULL,
                announced_size_kb DOUBLE,
                status VARCHAR NOT NULL,
                error VARCHAR,
                discovered_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_metrics (
                announcement_id VARCHAR PRIMARY KEY,
                sha256 VARCHAR NOT NULL,
                pages INTEGER NOT NULL,
                pdf_bytes BIGINT NOT NULL,
                embedded_text_bytes BIGINT NOT NULL,
                extracted_text_bytes BIGINT NOT NULL,
                ocr_text_bytes BIGINT NOT NULL,
                ocr_pages INTEGER NOT NULL,
                low_text_pages INTEGER NOT NULL,
                rendered_png_bytes BIGINT NOT NULL,
                persistent_inflation_pct DOUBLE NOT NULL,
                ocr_render_multiplier DOUBLE,
                fact_count INTEGER NOT NULL,
                parse_status VARCHAR NOT NULL,
                measured_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_facts (
                fact_id VARCHAR PRIMARY KEY,
                announcement_id VARCHAR NOT NULL,
                page_number INTEGER NOT NULL,
                category VARCHAR NOT NULL,
                evidence_sentence VARCHAR NOT NULL,
                review_status VARCHAR NOT NULL DEFAULT 'UNVALIDATED'
            );
            CREATE TABLE IF NOT EXISTS main_business (
                symbol VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                item VARCHAR NOT NULL,
                sales DOUBLE,
                profit DOUBLE,
                cost DOUBLE,
                currency VARCHAR,
                update_flag VARCHAR,
                collected_at TIMESTAMP NOT NULL,
                PRIMARY KEY (symbol, period_end, item)
            );
            CREATE TABLE IF NOT EXISTS main_business_status (
                symbol VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                row_count INTEGER NOT NULL,
                error VARCHAR,
                checked_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                chain_id VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                score DOUBLE,
                rank INTEGER,
                selected BOOLEAN NOT NULL,
                input_hash VARCHAR NOT NULL,
                frozen_at TIMESTAMP NOT NULL,
                PRIMARY KEY (decision_date, symbol, model)
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                decision_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                horizon INTEGER NOT NULL,
                entry_date DATE,
                exit_date DATE,
                gross_return DOUBLE,
                net_return DOUBLE,
                benchmark_return DOUBLE,
                chain_return DOUBLE,
                mae DOUBLE,
                mfe DOUBLE,
                status VARCHAR NOT NULL,
                settled_at TIMESTAMP,
                PRIMARY KEY (decision_date, symbol, model, horizon)
            );
            CREATE TABLE IF NOT EXISTS collection_runs (
                run_date DATE PRIMARY KEY,
                query_start DATE NOT NULL,
                query_end DATE NOT NULL,
                queried_companies INTEGER NOT NULL,
                query_failures INTEGER NOT NULL,
                discovered_documents INTEGER NOT NULL,
                downloaded_documents INTEGER NOT NULL,
                downloaded_bytes BIGINT NOT NULL,
                completed_at TIMESTAMP NOT NULL
            );
            """
        )

    def set_meta(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, default=_json_default, sort_keys=True)
        self.connection.execute(
            "INSERT OR REPLACE INTO pilot_meta VALUES (?, ?)", [key, encoded]
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value_json FROM pilot_meta WHERE key = ?", [key]
        ).fetchone()
        return json.loads(row[0]) if row else default

    def universe(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute("SELECT * FROM universe ORDER BY chain_id, sample_rank")
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class CninfoClient:
    """Low-rate official CNINFO disclosure client used only by the bounded pilot."""

    def __init__(self, *, min_interval_s: float = 0.45, timeout_s: float = 30.0) -> None:
        self._min_interval_s = max(0.0, min_interval_s)
        self._last_request_at = 0.0
        self._http = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.cninfo.com.cn/new/commonUrl?pageOfSearch=disclosure/list/search",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        self._org_map: dict[str, str] | None = None

    def close(self) -> None:
        self._http.close()

    def _wait(self) -> None:
        remaining = self._min_interval_s - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def org_map(self) -> dict[str, str]:
        if self._org_map is None:
            self._wait()
            response = self._http.get(CNINFO_STOCK_LIST_URL)
            response.raise_for_status()
            payload = response.json()
            stock_list = payload.get("stockList") if isinstance(payload, dict) else None
            if not isinstance(stock_list, list):
                raise RuntimeError("CNINFO stock list contract changed")
            self._org_map = {
                str(row["code"]): str(row["orgId"])
                for row in stock_list
                if isinstance(row, dict) and row.get("code") and row.get("orgId")
            }
        return self._org_map

    def announcements(self, code: str, start: date, end: date) -> list[dict[str, Any]]:
        org_id = self.org_map().get(code)
        if not org_id:
            raise RuntimeError(f"CNINFO has no orgId for {code}")
        self._wait()
        response = self._http.post(
            CNINFO_QUERY_URL,
            data={
                "pageNum": "1",
                "pageSize": "30",
                "column": "szse",
                "tabName": "fulltext",
                "plate": "",
                "stock": f"{code},{org_id}",
                "searchkey": "",
                "secid": "",
                "category": "",
                "trade": "",
                "seDate": f"{start.isoformat()}~{end.isoformat()}",
                "sortName": "time",
                "sortType": "desc",
                "isHLtitle": "true",
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("announcements") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("CNINFO announcement query contract changed")
        result: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict) or str(row.get("secCode")) != code:
                continue
            adjunct_url = str(row.get("adjunctUrl") or "").lstrip("/")
            if not adjunct_url.lower().endswith(".pdf"):
                continue
            raw_time = row.get("announcementTime")
            announce_time = None
            if isinstance(raw_time, (int, float)):
                announce_time = datetime.fromtimestamp(float(raw_time) / 1000)
            result.append(
                {
                    "announcement_id": str(row.get("announcementId") or row.get("id")),
                    "announce_time": announce_time,
                    "title": re.sub(r"<[^>]+>", "", str(row.get("announcementTitle") or "")).strip(),
                    "pdf_url": CNINFO_STATIC_ROOT + adjunct_url,
                    "announced_size_kb": float(row["adjunctSize"]) if row.get("adjunctSize") else None,
                }
            )
        return result

    def download_pdf(self, url: str, destination: Path, max_bytes: int) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._wait()
        total = 0
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle, self._http.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 256):
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(f"PDF exceeds {max_bytes} bytes")
                    handle.write(chunk)
            with open(temp_name, "rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise RuntimeError("download is not a PDF")
            os.replace(temp_name, destination)
            return total
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)


def extract_fact_candidates(page_texts: list[str], announcement_id: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for page_number, text in enumerate(page_texts, start=1):
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            compact = re.sub(r"\s+", " ", sentence).strip()
            if not 12 <= len(compact) <= 500:
                continue
            if not (_FACT_NUMBER_RE.search(compact) or _FACT_STRONG_RE.search(compact)):
                continue
            for category, keywords in FACT_PATTERNS.items():
                if not any(keyword in compact for keyword in keywords):
                    continue
                key = (category, compact)
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "fact_id": _stable_hash(announcement_id, str(page_number), category, compact),
                        "announcement_id": announcement_id,
                        "page_number": page_number,
                        "category": category,
                        "evidence_sentence": compact,
                        "review_status": "UNVALIDATED",
                    }
                )
    return facts


def analyze_pdf(
    pdf_path: Path,
    text_path: Path,
    announcement_id: str,
    *,
    max_ocr_pages: int = DEFAULT_MAX_OCR_PAGES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure one PDF; OCR only low-text pages and never persist rendered images."""
    import pymupdf as fitz
    import pytesseract
    from PIL import Image

    pdf_bytes = pdf_path.stat().st_size
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    embedded_text_bytes = 0
    ocr_text_bytes = 0
    rendered_png_bytes = 0
    low_text_pages = 0
    ocr_pages = 0
    page_texts: list[str] = []
    parse_status = "ok"

    with fitz.open(pdf_path) as document:
        for page in document:
            embedded = page.get_text("text") or ""
            embedded_bytes = len(embedded.encode("utf-8"))
            embedded_text_bytes += embedded_bytes
            visible_chars = len(re.sub(r"\s+", "", embedded))
            chosen = embedded
            if visible_chars < 80:
                low_text_pages += 1
                if ocr_pages < max_ocr_pages:
                    try:
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                        png = pixmap.tobytes("png")
                        rendered_png_bytes += len(png)
                        with Image.open(BytesIO(png)) as image:
                            ocr_text = pytesseract.image_to_string(
                                image, lang="chi_sim+eng", config="--psm 6"
                            )
                        ocr_pages += 1
                        ocr_text_bytes += len(ocr_text.encode("utf-8"))
                        if len(re.sub(r"\s+", "", ocr_text)) > visible_chars:
                            chosen = ocr_text
                    except Exception:
                        parse_status = "ocr_partial"
            page_texts.append(chosen)

    combined = "\n\n".join(
        f"--- PAGE {index} ---\n{text}" for index, text in enumerate(page_texts, start=1)
    )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{text_path.name}.", dir=text_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(combined)
        os.replace(temp_name, text_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
    extracted_text_bytes = text_path.stat().st_size
    facts = extract_fact_candidates(page_texts, announcement_id)
    metrics = {
        "announcement_id": announcement_id,
        "sha256": sha256,
        "pages": len(page_texts),
        "pdf_bytes": pdf_bytes,
        "embedded_text_bytes": embedded_text_bytes,
        "extracted_text_bytes": extracted_text_bytes,
        "ocr_text_bytes": ocr_text_bytes,
        "ocr_pages": ocr_pages,
        "low_text_pages": low_text_pages,
        "rendered_png_bytes": rendered_png_bytes,
        "persistent_inflation_pct": extracted_text_bytes / max(1, pdf_bytes) * 100,
        "ocr_render_multiplier": (
            rendered_png_bytes / pdf_bytes if ocr_pages and pdf_bytes else None
        ),
        "fact_count": len(facts),
        "parse_status": parse_status,
        "measured_at": datetime.now(),
    }
    return metrics, facts


def _linear_rating(value: float, lower: float, upper: float) -> float:
    return min(5.0, max(0.0, (value - lower) / (upper - lower) * 5.0))


def _score_serenity(row: dict[str, Any]) -> tuple[float | None, bool]:
    required = (
        "revenue_yoy",
        "net_income_yoy",
        "roe",
        "gross_margin",
        "debt_to_asset_ratio",
        "pb",
        "momentum_60d",
        "amount_ratio_5d",
    )
    if any(row.get(field) is None or not math.isfinite(float(row[field])) for field in required):
        return None, False
    demand = (
        _linear_rating(float(row["revenue_yoy"]), 0, 50) * 0.6
        + _linear_rating(float(row["net_income_yoy"]), 0, 50) * 0.4
    )
    roe_per_pb = float(row["roe"]) / float(row["pb"])
    valuation = _linear_rating(roe_per_pb, 1, 8)
    timing = (
        _linear_rating(float(row["momentum_60d"]), -0.10, 0.40) * 0.7
        + _linear_rating(float(row["amount_ratio_5d"]), -0.30, 1.20) * 0.3
    )
    score = demand / 5 * 15 + valuation / 5 * 11 + timing / 5 * 10
    gate = (
        float(row["revenue_yoy"]) >= 5
        and float(row["net_income_yoy"]) >= 0
        and float(row["roe"]) >= 8
        and float(row["gross_margin"]) >= 15
        and float(row["debt_to_asset_ratio"]) <= 75
        and 0 < float(row["pb"]) <= 15
        and float(row["momentum_60d"]) >= -0.10
        and score >= 12
    )
    return score, gate


def _daily_features(data_dir: Path, decision_date: date, symbols: set[str]) -> list[dict[str, Any]]:
    dates = _partition_dates(data_dir / "kline_daily_enriched", not_after=decision_date)
    dates = dates[-90:]
    if not dates or dates[-1] != decision_date:
        return []
    frames = [
        pl.read_parquet(data_dir / "kline_daily_enriched" / f"date={value}" / "part.parquet")
        .filter(pl.col("symbol").is_in(sorted(symbols)))
        .select("symbol", "date", "open", "high", "low", "close", "amount")
        for value in dates
    ]
    history = pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "date"])
    metrics = pl.read_parquet(data_dir / "financials" / "metrics" / "part.parquet").filter(
        pl.col("symbol").is_in(sorted(symbols))
        & (pl.col("announce_date") < decision_date.isoformat())
    )
    latest_metrics = metrics.sort(["symbol", "announce_date", "period_end"]).group_by(
        "symbol", maintain_order=True
    ).tail(1)
    metrics_map = {row["symbol"]: row for row in latest_metrics.iter_rows(named=True)}
    by_symbol = history.partition_by("symbol", as_dict=True)
    result: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        frame = by_symbol.get((symbol,))
        if frame is None or frame.height < 61:
            continue
        rows = frame.sort("date").to_dicts()
        today = rows[-1]
        previous_5 = rows[-6:-1]
        metric = metrics_map.get(symbol, {})
        bps = metric.get("bps")
        close = float(today["close"])
        pb = close / float(bps) if bps not in (None, 0) else None
        mean_amount = statistics.fmean(float(row["amount"]) for row in previous_5)
        result.append(
            {
                "symbol": symbol,
                "close": close,
                "momentum_60d": close / float(rows[-61]["close"]) - 1,
                "amount_ratio_5d": float(today["amount"]) / mean_amount - 1 if mean_amount else None,
                "pb": pb,
                "roe": metric.get("roe"),
                "gross_margin": metric.get("gross_margin"),
                "debt_to_asset_ratio": metric.get("debt_to_asset_ratio"),
                "revenue_yoy": metric.get("revenue_yoy"),
                "net_income_yoy": metric.get("net_income_yoy"),
            }
        )
    return result


def freeze_daily_decisions(store: PilotStore, data_dir: Path, decision_date: date) -> dict[str, Any]:
    already = store.connection.execute(
        "SELECT count(*) FROM decisions WHERE decision_date = ?", [decision_date]
    ).fetchone()[0]
    if already:
        return {"status": "already_frozen", "rows": int(already)}
    universe = store.universe()
    chain_map = {row["symbol"]: row["chain_id"] for row in universe}
    features = _daily_features(data_dir, decision_date, set(chain_map))
    if not features:
        return {"status": "no_market_partition", "rows": 0}
    feature_map = {row["symbol"]: row for row in features}
    input_hash = _stable_hash(
        decision_date.isoformat(),
        json.dumps(features, sort_keys=True, default=_json_default),
    )

    serenity_scores: dict[str, float | None] = {}
    serenity_eligible: list[tuple[float, str]] = []
    for symbol, row in feature_map.items():
        score, eligible = _score_serenity(row)
        serenity_scores[symbol] = score
        if eligible and score is not None:
            serenity_eligible.append((score, symbol))
    serenity_selected = {
        symbol for _, symbol in sorted(serenity_eligible, key=lambda item: (-item[0], item[1]))[:10]
    }
    momentum_selected = {
        row["symbol"]
        for row in sorted(
            features,
            key=lambda row: (-float(row["momentum_60d"]), row["symbol"]),
        )[:10]
    }
    random_selected = {
        row["symbol"]
        for row in sorted(
            features,
            key=lambda row: _stable_hash(decision_date.isoformat(), row["symbol"]),
        )[:10]
    }
    selections = {
        "serenity": serenity_selected,
        "momentum": momentum_selected,
        "random": random_selected,
        "chain_equal": set(feature_map),
    }
    frozen_at = datetime.now()
    inserts: list[list[Any]] = []
    for model in MODELS:
        if model == "serenity":
            ranks = {
                symbol: rank
                for rank, (_, symbol) in enumerate(
                    sorted(serenity_eligible, key=lambda item: (-item[0], item[1])), start=1
                )
            }
        elif model == "momentum":
            ranks = {
                row["symbol"]: rank
                for rank, row in enumerate(
                    sorted(features, key=lambda row: (-float(row["momentum_60d"]), row["symbol"])),
                    start=1,
                )
            }
        elif model == "random":
            ranks = {
                row["symbol"]: rank
                for rank, row in enumerate(
                    sorted(features, key=lambda row: _stable_hash(decision_date.isoformat(), row["symbol"])),
                    start=1,
                )
            }
        else:
            ranks = {row["symbol"]: 1 for row in features}
        for symbol, row in feature_map.items():
            score = serenity_scores[symbol] if model == "serenity" else (
                float(row["momentum_60d"]) if model == "momentum" else None
            )
            inserts.append(
                [
                    decision_date,
                    symbol,
                    chain_map[symbol],
                    model,
                    score,
                    ranks.get(symbol),
                    symbol in selections[model],
                    input_hash,
                    frozen_at,
                ]
            )
    store.connection.executemany(
        "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", inserts
    )
    return {
        "status": "frozen",
        "rows": len(inserts),
        "selected": {model: len(symbols) for model, symbols in selections.items()},
        "input_hash": input_hash,
    }


def _load_market_rows(data_dir: Path, symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    frames: list[pl.DataFrame] = []
    for value in _partition_dates(data_dir / "kline_daily_enriched"):
        frame = pl.read_parquet(
            data_dir / "kline_daily_enriched" / f"date={value}" / "part.parquet"
        ).filter(pl.col("symbol").is_in(sorted(symbols)))
        if not frame.is_empty():
            frames.append(frame.select("symbol", "date", "open", "high", "low", "close"))
    if not frames:
        return {}
    combined = pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "date"])
    return {
        key[0]: frame.to_dicts()
        for key, frame in combined.partition_by("symbol", as_dict=True).items()
    }


def _load_index_rows(data_dir: Path, symbol: str = "000300.SH") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in _partition_dates(data_dir / "kline_index_daily"):
        frame = pl.read_parquet(
            data_dir / "kline_index_daily" / f"date={value}" / "part.parquet"
        ).filter(pl.col("symbol") == symbol)
        rows.extend(frame.select("date", "open", "close").to_dicts())
    return sorted(rows, key=lambda row: row["date"])


def settle_outcomes(store: PilotStore, data_dir: Path, *, cost_bps: float) -> dict[str, Any]:
    cursor = store.connection.execute(
        """
        SELECT decision_date, symbol, model, chain_id
        FROM decisions
        WHERE selected
        ORDER BY decision_date, model, symbol
        """
    )
    selected = cursor.fetchall()
    if not selected:
        return {"settled": 0, "pending": 0}
    symbols = {row[1] for row in selected}
    market = _load_market_rows(data_dir, symbols)
    index_rows = _load_index_rows(data_dir)
    index_by_date = {row["date"]: row for row in index_rows}
    universe = store.universe()
    chain_symbols: dict[str, set[str]] = defaultdict(set)
    for row in universe:
        chain_symbols[row["chain_id"]].add(row["symbol"])

    settled = 0
    pending = 0
    now = datetime.now()
    for decision_date, symbol, model, chain_id in selected:
        symbol_rows = market.get(symbol, [])
        future = [row for row in symbol_rows if row["date"] > decision_date]
        for horizon in HORIZONS:
            existing = store.connection.execute(
                "SELECT status FROM outcomes WHERE decision_date=? AND symbol=? AND model=? AND horizon=?",
                [decision_date, symbol, model, horizon],
            ).fetchone()
            if existing and existing[0] == "settled":
                continue
            if len(future) < horizon:
                store.connection.execute(
                    "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'pending', NULL)",
                    [decision_date, symbol, model, horizon],
                )
                pending += 1
                continue
            window = future[:horizon]
            entry = window[0]
            exit_ = window[-1]
            entry_price = float(entry["open"])
            exit_price = float(exit_["close"])
            gross = exit_price / entry_price - 1
            net = gross - cost_bps / 10_000
            mae = min(float(row["low"]) / entry_price - 1 for row in window)
            mfe = max(float(row["high"]) / entry_price - 1 for row in window)
            index_entry = index_by_date.get(entry["date"])
            index_exit = index_by_date.get(exit_["date"])
            benchmark_return = None
            if index_entry and index_exit and float(index_entry["open"]):
                benchmark_return = float(index_exit["close"]) / float(index_entry["open"]) - 1

            chain_returns: list[float] = []
            for peer in chain_symbols[chain_id]:
                peer_future = [row for row in market.get(peer, []) if row["date"] > decision_date]
                if len(peer_future) >= horizon and float(peer_future[0]["open"]):
                    chain_returns.append(
                        float(peer_future[horizon - 1]["close"]) / float(peer_future[0]["open"]) - 1
                    )
            chain_return = statistics.fmean(chain_returns) if chain_returns else None
            store.connection.execute(
                """
                INSERT OR REPLACE INTO outcomes VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'settled', ?)
                """,
                [
                    decision_date,
                    symbol,
                    model,
                    horizon,
                    entry["date"],
                    exit_["date"],
                    gross,
                    net,
                    benchmark_return,
                    chain_return,
                    mae,
                    mfe,
                    now,
                ],
            )
            settled += 1
    return {"settled": settled, "pending": pending}


def initialize_pilot(
    root: Path,
    data_dir: Path,
    start_date: date,
    *,
    end_date: date | None = None,
    decision_dates: list[date] | None = None,
    mode: str = "prospective",
) -> dict[str, Any]:
    store = PilotStore(root)
    try:
        existing = store.get_meta("manifest")
        if existing:
            return existing
        source_as_of, universe = select_universe(data_dir, start_date)
        resolved_end = end_date or (start_date + timedelta(days=6))
        resolved_decision_dates = decision_dates or []
        if resolved_decision_dates:
            if resolved_decision_dates != sorted(set(resolved_decision_dates)):
                raise ValueError("decision dates must be unique and ascending")
            if resolved_decision_dates[0] != start_date or resolved_decision_dates[-1] != resolved_end:
                raise ValueError("decision dates must match the manifest start and end")
        retrospective = mode == "retrospective_historical"
        universe_hash = _stable_hash(
            *[f"{row['symbol']}:{row['chain_id']}" for row in sorted(universe, key=lambda row: row["symbol"])]
        )
        manifest = {
            "pilot_version": PILOT_VERSION,
            "pilot_id": root.name,
            "mode": mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": start_date.isoformat(),
            "end_date": resolved_end.isoformat(),
            "decision_dates": [value.isoformat() for value in resolved_decision_dates],
            "source_as_of": source_as_of.isoformat(),
            "sample_size": len(universe),
            "sample_hash": universe_hash,
            "chains": [asdict(spec) for spec in CHAIN_SPECS],
            "limits": {
                "max_documents": DEFAULT_MAX_DOCUMENTS,
                "max_raw_bytes": DEFAULT_MAX_RAW_BYTES,
                "max_document_bytes": DEFAULT_MAX_DOCUMENT_BYTES,
                "max_documents_per_company": DEFAULT_MAX_DOCUMENTS_PER_COMPANY,
                "max_ocr_pages_per_document": DEFAULT_MAX_OCR_PAGES,
            },
            "research_contract": {
                "decision_time": "after_close",
                "entry": "next_trading_day_open",
                "horizons": list(HORIZONS),
                "cost_bps": DEFAULT_COST_BPS,
                "benchmark": "000300.SH",
                "models": list(MODELS),
                "alpha_claim_gate": "minimum_60_trading_sessions_and_200_settled_positions",
                "seven_day_label": "UNVERIFIED_ALPHA",
            },
            "historical_replay_qualification": {
                "status": (
                    "RETROSPECTIVE_ENGINEERING_SAMPLE_NOT_CLEAN_ROOM"
                    if retrospective
                    else "NOT_APPLICABLE"
                ),
                "price_and_financial_inputs": "POINT_IN_TIME_BY_LOCAL_PARTITION",
                "concept_membership": (
                    "UNRESOLVED_CURRENT_SNAPSHOT"
                    if retrospective
                    else "CURRENT_SNAPSHOT_AT_PILOT_START"
                ),
                "instrument_snapshot": (
                    "UNRESOLVED_CURRENT_SNAPSHOT"
                    if retrospective
                    else "CURRENT_SNAPSHOT_AT_PILOT_START"
                ),
                "pdf_facts_in_strategy_score": False,
                "semantic_model_used": False,
                "claim_boundary": "DECISION_PROCESS_EVIDENCE_ONLY_NO_ALPHA_CLAIM",
            },
        }
        rows = [
            [
                row["symbol"],
                row["code"],
                row["name"],
                row["chain_id"],
                row["chain_name"],
                row["chain_role"],
                row["sample_rank"],
                row["market_cap_bucket"],
                row["concept_score"],
                row["concepts"],
                row["market_cap"],
                row["amount"],
                row["source_as_of"],
            ]
            for row in universe
        ]
        store.connection.executemany(
            "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        store.set_meta("manifest", manifest)
        _atomic_json(root / "pilot.json", manifest)
        return manifest
    finally:
        store.close()


def collect_documents(
    store: PilotStore,
    query_start: date,
    query_end: date,
) -> dict[str, Any]:
    universe = store.universe()
    client = CninfoClient()
    discovered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failures = 0
    try:
        for company in universe:
            try:
                for item in client.announcements(company["code"], query_start, query_end):
                    discovered.append((company, item))
                    store.connection.execute(
                        """
                        INSERT OR IGNORE INTO announcements VALUES
                        (?, ?, ?, ?, ?, ?, 'discovered', NULL, ?)
                        """,
                        [
                            item["announcement_id"],
                            company["symbol"],
                            item["announce_time"],
                            item["title"],
                            item["pdf_url"],
                            item["announced_size_kb"],
                            datetime.now(),
                        ],
                    )
            except Exception:
                failures += 1

        current_docs, current_bytes = store.connection.execute(
            "SELECT count(*), coalesce(sum(pdf_bytes), 0) FROM document_metrics"
        ).fetchone()
        per_symbol = dict(
            store.connection.execute(
                """
                SELECT a.symbol, count(*)
                FROM announcements a JOIN document_metrics d USING (announcement_id)
                GROUP BY a.symbol
                """
            ).fetchall()
        )
        chain_order = {spec.id: index for index, spec in enumerate(CHAIN_SPECS)}
        discovered.sort(
            key=lambda pair: (
                int(per_symbol.get(pair[0]["symbol"], 0)),
                chain_order[pair[0]["chain_id"]],
                pair[1]["announce_time"] or datetime.min,
                pair[1]["announcement_id"],
            )
        )
        downloaded = 0
        downloaded_bytes = 0
        for company, item in discovered:
            announcement_id = item["announcement_id"]
            if store.connection.execute(
                "SELECT 1 FROM document_metrics WHERE announcement_id=?", [announcement_id]
            ).fetchone():
                continue
            if current_docs >= DEFAULT_MAX_DOCUMENTS or current_bytes >= DEFAULT_MAX_RAW_BYTES:
                store.connection.execute(
                    "UPDATE announcements SET status='capped', error='pilot storage cap reached' WHERE announcement_id=?",
                    [announcement_id],
                )
                continue
            if int(per_symbol.get(company["symbol"], 0)) >= DEFAULT_MAX_DOCUMENTS_PER_COMPANY:
                store.connection.execute(
                    "UPDATE announcements SET status='capped', error='per-company document cap reached' WHERE announcement_id=?",
                    [announcement_id],
                )
                continue
            pdf_path = store.documents_dir / f"{announcement_id}.pdf"
            text_path = store.text_dir / f"{announcement_id}.txt"
            try:
                size = client.download_pdf(
                    item["pdf_url"], pdf_path, DEFAULT_MAX_DOCUMENT_BYTES
                )
                if current_bytes + size > DEFAULT_MAX_RAW_BYTES:
                    pdf_path.unlink(missing_ok=True)
                    store.connection.execute(
                        "UPDATE announcements SET status='capped', error='pilot raw-byte cap reached' WHERE announcement_id=?",
                        [announcement_id],
                    )
                    continue
                metrics, facts = analyze_pdf(pdf_path, text_path, announcement_id)
                store.connection.execute(
                    """
                    INSERT INTO document_metrics VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        metrics["announcement_id"],
                        metrics["sha256"],
                        metrics["pages"],
                        metrics["pdf_bytes"],
                        metrics["embedded_text_bytes"],
                        metrics["extracted_text_bytes"],
                        metrics["ocr_text_bytes"],
                        metrics["ocr_pages"],
                        metrics["low_text_pages"],
                        metrics["rendered_png_bytes"],
                        metrics["persistent_inflation_pct"],
                        metrics["ocr_render_multiplier"],
                        metrics["fact_count"],
                        metrics["parse_status"],
                        metrics["measured_at"],
                    ],
                )
                if facts:
                    store.connection.executemany(
                        "INSERT OR IGNORE INTO evidence_facts VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            [
                                fact["fact_id"],
                                fact["announcement_id"],
                                fact["page_number"],
                                fact["category"],
                                fact["evidence_sentence"],
                                fact["review_status"],
                            ]
                            for fact in facts
                        ],
                    )
                store.connection.execute(
                    "UPDATE announcements SET status='measured', error=NULL WHERE announcement_id=?",
                    [announcement_id],
                )
                current_docs += 1
                current_bytes += size
                downloaded += 1
                downloaded_bytes += size
                per_symbol[company["symbol"]] = int(per_symbol.get(company["symbol"], 0)) + 1
            except Exception as exc:
                pdf_path.unlink(missing_ok=True)
                text_path.unlink(missing_ok=True)
                store.connection.execute(
                    "UPDATE announcements SET status='failed', error=? WHERE announcement_id=?",
                    [str(exc)[:300], announcement_id],
                )
        return {
            "queried_companies": len(universe),
            "query_failures": failures,
            "discovered_documents": len({item["announcement_id"] for _, item in discovered}),
            "downloaded_documents": downloaded,
            "downloaded_bytes": downloaded_bytes,
        }
    finally:
        client.close()


def collect_main_business(store: PilotStore) -> dict[str, Any]:
    """Freeze the Tushare product-level business snapshot once per sample company."""
    checked = {
        row[0]
        for row in store.connection.execute(
            "SELECT symbol FROM main_business_status WHERE status IN ('ok', 'empty')"
        ).fetchall()
    }
    companies = [row for row in store.universe() if row["symbol"] not in checked]
    if not companies:
        covered, rows = store.connection.execute(
            "SELECT count(*), coalesce(sum(row_count), 0) FROM main_business_status"
        ).fetchone()
        return {"queried": 0, "covered_companies": covered, "rows": rows, "failures": 0}
    client = TushareClient(get_api_key())
    failures = 0
    inserted = 0
    try:
        for company in companies:
            symbol = company["symbol"]
            try:
                records = client.main_business_records(symbol, kind="P")
                values: list[list[Any]] = []
                for row in records:
                    item = str(row.get("bz_item") or "").strip()
                    raw_period = str(row.get("end_date") or "")
                    if not item or not re.fullmatch(r"\d{8}", raw_period):
                        continue
                    values.append(
                        [
                            symbol,
                            datetime.strptime(raw_period, "%Y%m%d").date(),
                            item,
                            row.get("bz_sales"),
                            row.get("bz_profit"),
                            row.get("bz_cost"),
                            row.get("curr_type"),
                            row.get("update_flag"),
                            datetime.now(),
                        ]
                    )
                if values:
                    store.connection.executemany(
                        "INSERT OR REPLACE INTO main_business VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        values,
                    )
                store.connection.execute(
                    "INSERT OR REPLACE INTO main_business_status VALUES (?, ?, ?, NULL, ?)",
                    [symbol, "ok" if values else "empty", len(values), datetime.now()],
                )
                inserted += len(values)
            except Exception as exc:
                failures += 1
                store.connection.execute(
                    "INSERT OR REPLACE INTO main_business_status VALUES (?, 'failed', 0, ?, ?)",
                    [symbol, str(exc)[:300], datetime.now()],
                )
        covered, total_rows = store.connection.execute(
            "SELECT count(*), coalesce(sum(row_count), 0) FROM main_business_status"
        ).fetchone()
        return {
            "queried": len(companies),
            "covered_companies": covered,
            "inserted_rows": inserted,
            "rows": total_rows,
            "failures": failures,
        }
    finally:
        client.close()


def build_report(store: PilotStore) -> dict[str, Any]:
    manifest = store.get_meta("manifest", {})
    doc_rows = store.connection.execute(
        """
        SELECT pdf_bytes, pages, persistent_inflation_pct, ocr_render_multiplier,
               ocr_pages, fact_count, parse_status
        FROM document_metrics
        """
    ).fetchall()
    pdf_sizes = [int(row[0]) for row in doc_rows]
    facts = [int(row[5]) for row in doc_rows]
    pages = [int(row[1]) for row in doc_rows]
    discovered = store.connection.execute("SELECT count(*) FROM announcements").fetchone()[0]
    measured = len(doc_rows)
    failed = store.connection.execute(
        "SELECT count(*) FROM announcements WHERE status='failed'"
    ).fetchone()[0]
    capped = store.connection.execute(
        "SELECT count(*) FROM announcements WHERE status='capped'"
    ).fetchone()[0]
    decision_days = store.connection.execute(
        "SELECT count(DISTINCT decision_date) FROM decisions"
    ).fetchone()[0]
    selected_positions = store.connection.execute(
        "SELECT count(*) FROM decisions WHERE model='serenity' AND selected"
    ).fetchone()[0]
    main_business_companies, main_business_rows = store.connection.execute(
        "SELECT count(*), coalesce(sum(row_count), 0) FROM main_business_status"
    ).fetchone()
    settled_rows = store.connection.execute(
        """
        SELECT model, horizon, count(*), avg(net_return), avg(net_return-benchmark_return),
               avg(net_return-chain_return), min(mae), max(mfe)
        FROM outcomes
        WHERE status='settled'
        GROUP BY model, horizon
        ORDER BY model, horizon
        """
    ).fetchall()
    outcomes = [
        {
            "model": row[0],
            "horizon": row[1],
            "positions": row[2],
            "mean_net_return": row[3],
            "mean_alpha_vs_csi300": row[4],
            "mean_alpha_vs_chain": row[5],
            "worst_mae": row[6],
            "best_mfe": row[7],
        }
        for row in settled_rows
    ]
    serenity_three_day_positions = sum(
        int(row["positions"])
        for row in outcomes
        if row["model"] == "serenity" and row["horizon"] == 3
    )
    mean_size = statistics.fmean(pdf_sizes) if pdf_sizes else None
    sorted_sizes = sorted(pdf_sizes)
    p95_index = max(0, math.ceil(len(sorted_sizes) * 0.95) - 1) if sorted_sizes else 0
    report = {
        "pilot_id": manifest.get("pilot_id"),
        "mode": manifest.get("mode", "prospective"),
        "period": {"start": manifest.get("start_date"), "end": manifest.get("end_date")},
        "decision_dates": manifest.get("decision_dates", []),
        "universe": {
            "companies": store.connection.execute("SELECT count(*) FROM universe").fetchone()[0],
            "chains": dict(
                store.connection.execute(
                    "SELECT chain_id, count(*) FROM universe GROUP BY chain_id ORDER BY chain_id"
                ).fetchall()
            ),
            "sample_hash": manifest.get("sample_hash"),
        },
        "documents": {
            "discovered": discovered,
            "measured": measured,
            "failed": failed,
            "capped": capped,
            "raw_bytes": sum(pdf_sizes),
            "mean_pdf_bytes": mean_size,
            "median_pdf_bytes": statistics.median(pdf_sizes) if pdf_sizes else None,
            "p95_pdf_bytes": sorted_sizes[p95_index] if sorted_sizes else None,
            "mean_persistent_text_inflation_pct": (
                statistics.fmean(float(row[2]) for row in doc_rows) if doc_rows else None
            ),
            "ocr_documents": sum(1 for row in doc_rows if int(row[4]) > 0),
            "mean_ocr_render_multiplier": (
                statistics.fmean(float(row[3]) for row in doc_rows if row[3] is not None)
                if any(row[3] is not None for row in doc_rows)
                else None
            ),
            "parse_success_rate": (
                sum(1 for row in doc_rows if row[6] == "ok") / measured if measured else None
            ),
        },
        "evidence": {
            "candidate_facts": sum(facts),
            "facts_per_document": statistics.fmean(facts) if facts else None,
            "facts_per_100_pages": sum(facts) / sum(pages) * 100 if sum(pages) else None,
            "reviewed_facts": store.connection.execute(
                "SELECT count(*) FROM evidence_facts WHERE review_status<>'UNVALIDATED'"
            ).fetchone()[0],
            "status": "UNVALIDATED_RULE_CANDIDATES",
            "main_business_companies": main_business_companies,
            "main_business_rows": main_business_rows,
        },
        "strategy": {
            "decision_days": decision_days,
            "selected_positions": selected_positions,
            "outcomes": outcomes,
            "alpha_status": (
                "UNVERIFIED_ALPHA"
                if serenity_three_day_positions >= 20
                else "INSUFFICIENT_OBSERVATION"
            ),
            "claim_gate": manifest.get("research_contract", {}).get("alpha_claim_gate"),
        },
        "qualification": {
            "storage_cap_respected": sum(pdf_sizes) <= DEFAULT_MAX_RAW_BYTES,
            "document_coverage_ge_80pct": (
                measured / discovered >= 0.80 if discovered else None
            ),
            "parse_success_ge_95pct": (
                sum(1 for row in doc_rows if row[6] == "ok") / measured >= 0.95
                if measured
                else None
            ),
            "fact_precision_gate": "PENDING_50_FACT_MANUAL_REVIEW",
            "historical_replay": manifest.get("historical_replay_qualification", {}),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_json(store.root / "report.json", report)
    return report


def run_daily(root: Path, data_dir: Path, as_of: date) -> dict[str, Any]:
    store = PilotStore(root)
    try:
        manifest = store.get_meta("manifest")
        if not manifest:
            raise RuntimeError("pilot is not initialized")
        start = date.fromisoformat(manifest["start_date"])
        end = date.fromisoformat(manifest["end_date"])
        query_end = min(as_of, end)
        decision = (
            freeze_daily_decisions(store, data_dir, as_of)
            if start <= as_of <= end
            else {"status": "outside_pilot_window"}
        )
        settlement = settle_outcomes(
            store,
            data_dir,
            cost_bps=float(manifest["research_contract"]["cost_bps"]),
        )
        collection: dict[str, Any]
        if query_end < start:
            collection = {"status": "before_start"}
        else:
            collection = collect_documents(store, start, query_end)
            store.connection.execute(
                """
                INSERT OR REPLACE INTO collection_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    as_of,
                    start,
                    query_end,
                    collection["queried_companies"],
                    collection["query_failures"],
                    collection["discovered_documents"],
                    collection["downloaded_documents"],
                    collection["downloaded_bytes"],
                    datetime.now(),
                ],
            )
        main_business = collect_main_business(store)
        report = build_report(store)
        result = {
            "pilot_id": manifest["pilot_id"],
            "as_of": as_of.isoformat(),
            "collection": collection,
            "main_business": main_business,
            "decision": decision,
            "settlement": settlement,
            "report": report,
        }
        _atomic_json(root / "status.json", result)
        return result
    finally:
        store.close()


def run_historical(
    root: Path,
    data_dir: Path,
    *,
    end_date: date,
    trading_days: int = 7,
) -> dict[str, Any]:
    """Backfill one bounded retrospective sample and settle only available horizons."""
    decision_dates = _historical_decision_dates(data_dir, end_date, trading_days)
    start = decision_dates[0]
    end = decision_dates[-1]
    manifest = initialize_pilot(
        root,
        data_dir,
        start,
        end_date=end,
        decision_dates=decision_dates,
        mode="retrospective_historical",
    )
    expected_dates = [value.isoformat() for value in decision_dates]
    if manifest.get("mode") != "retrospective_historical":
        raise RuntimeError("historical command cannot reuse a prospective pilot root")
    if manifest.get("decision_dates") != expected_dates:
        raise RuntimeError("historical pilot root is bound to a different trading window")

    store = PilotStore(root)
    try:
        collection = collect_documents(store, start, end)
        store.connection.execute(
            """
            INSERT OR REPLACE INTO collection_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                end,
                start,
                end,
                collection["queried_companies"],
                collection["query_failures"],
                collection["discovered_documents"],
                collection["downloaded_documents"],
                collection["downloaded_bytes"],
                datetime.now(),
            ],
        )
        main_business = collect_main_business(store)
        decisions = {
            value.isoformat(): freeze_daily_decisions(store, data_dir, value)
            for value in decision_dates
        }
        settlement = settle_outcomes(
            store,
            data_dir,
            cost_bps=float(manifest["research_contract"]["cost_bps"]),
        )
        report = build_report(store)
        result = {
            "pilot_id": manifest["pilot_id"],
            "mode": manifest["mode"],
            "decision_dates": expected_dates,
            "collection": collection,
            "main_business": main_business,
            "decisions": decisions,
            "settlement": settlement,
            "report": report,
        }
        _atomic_json(root / "status.json", result)
        return result
    finally:
        store.close()


def _default_root(start: date) -> Path:
    return settings.data_dir / "research" / "serenity_pilot" / f"serenity-7d-{start:%Y%m%d}"


def _historical_root(end: date, trading_days: int) -> Path:
    return (
        settings.data_dir
        / "research"
        / "serenity_pilot"
        / f"serenity-historical-{trading_days}td-{end:%Y%m%d}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run-daily", "run-historical", "status"))
    parser.add_argument("--start-date", default=date.today().isoformat())
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--trading-days", type=int, default=7)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    historical_end = date.fromisoformat(args.end_date)
    start = date.fromisoformat(args.start_date)
    root = args.root or (
        _historical_root(historical_end, args.trading_days)
        if args.command == "run-historical"
        else _default_root(start)
    )
    with _pilot_lock(root):
        if args.command == "init":
            payload = initialize_pilot(root, settings.data_dir, start)
        elif args.command == "run-daily":
            payload = run_daily(root, settings.data_dir, date.fromisoformat(args.as_of))
        elif args.command == "run-historical":
            payload = run_historical(
                root,
                settings.data_dir,
                end_date=historical_end,
                trading_days=args.trading_days,
            )
        else:
            store = PilotStore(root)
            try:
                payload = build_report(store)
            finally:
                store.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
