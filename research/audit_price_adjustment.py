"""除权污染审计 — 阶段一验收脚本。

用法:
    cd backend && .venv/bin/python ../research/audit_price_adjustment.py
    cd backend && .venv/bin/python ../research/audit_price_adjustment.py --json

验收标准: 复权接入完成后, `复权已生效` 为 True 且 `异常行数` 为 0。
只要 close 恒等于 raw_close, 除权日的价格断层就会被当成真实跌幅,
制造假新低信号并误触发止损, 此时任何长周期回测结论都不可信。
"""
# 报告面向中文读者, 正文使用全角标点。
# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.data_quality import (  # noqa: E402
    audit_price_adjustment,
    audit_signal_contamination,
)

DEFAULT_DATA_DIR = ROOT / "data"
ENRICHED_DIRNAME = "kline_daily_enriched"
PANEL_COLUMNS = ["symbol", "date", "close", "raw_close"]


def load_panel(data_dir: Path) -> pl.DataFrame:
    paths = sorted((data_dir / ENRICHED_DIRNAME).glob("date=*/part.parquet"))
    if not paths:
        return pl.DataFrame()
    return pl.concat(
        [pl.read_parquet(path, columns=PANEL_COLUMNS) for path in paths],
        how="vertical_relaxed",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json", action="store_true", help="输出机器可读结果")
    args = parser.parse_args()

    panel = load_panel(args.data_dir)
    if panel.is_empty():
        print(f"没有找到日线数据: {args.data_dir / ENRICHED_DIRNAME}", file=sys.stderr)
        return 2

    audit = audit_price_adjustment(panel)
    impact = audit_signal_contamination(panel)
    passed = audit.adjustment_applied and audit.suspect_rows == 0

    if args.json:
        print(json.dumps(
            {
                "passed": passed,
                "total_rows": audit.total_rows,
                "adjustment_applied": audit.adjustment_applied,
                "suspect_rows": audit.suspect_rows,
                "suspect_symbols": audit.suspect_symbols,
                "new_listing_rows": audit.new_listing_rows,
                "suspension_rows": audit.suspension_rows,
                "monthly_counts": audit.monthly_counts,
                "fake_new_lows": impact.fake_new_lows,
                "stop_loss_hits": impact.stop_loss_hits,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if passed else 1

    ratio = audit.suspect_rows / audit.total_rows * 100 if audit.total_rows else 0.0
    print("除权污染审计")
    print("=" * 52)
    print(f"总行数        {audit.total_rows:,}")
    print(f"复权已生效    {'是' if audit.adjustment_applied else '否'}")
    print(f"异常行数      {audit.suspect_rows:,}  ({ratio:.3f}%)")
    print(f"涉及标的      {audit.suspect_symbols:,}")
    print(f"次新股豁免    {audit.new_listing_rows:,}  上市初期不设涨跌幅，属合法交易")
    print(f"停牌复牌豁免  {audit.suspension_rows:,}  停牌期间信息累积，复牌跳空属合法")

    if audit.monthly_counts:
        peak = max(audit.monthly_counts.values())
        print("\n按月分布（A股除权集中在4-7月，若此处出现尖峰即为除权污染）")
        for month, count in sorted(audit.monthly_counts.items()):
            bar = "█" * max(1, round(count / peak * 32)) if count else ""
            print(f"  {month}  {count:>5}  {bar}")

    if audit.samples:
        print("\n最严重的样本（这些跌幅在任何板块的涨跌停规则下都不可能由交易产生）")
        for sample in audit.samples[:10]:
            print(
                f"  {sample['symbol']:<12} {sample['date']}  "
                f"{sample['return_pct'] * 100:>7.2f}%   收盘 {sample['raw_close']}"
            )

    print("\n对策略信号的污染")
    fake_ratio = (
        impact.fake_new_lows / impact.suspect_rows * 100 if impact.suspect_rows else 0.0
    )
    print(f"  制造假的60日新低    {impact.fake_new_lows:>5}  ({fake_ratio:.1f}% 的假跌幅)")
    print(f"  穿透 -6% 止损线     {impact.stop_loss_hits:>5}  持仓会被除权断层误止损")

    print()
    if passed:
        print("通过：复权已生效，且没有无法由交易解释的跌幅。")
    else:
        print("未通过：在接入除权因子之前，任何长周期回测结论都不可信。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
