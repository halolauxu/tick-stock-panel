import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  ArrowRight,
  BarChart3,
  Check,
  Clock3,
  Database,
  FileCheck2,
  ShieldAlert,
} from 'lucide-react'
import { Link } from 'react-router-dom'

import { api, type ManagedForwardStrategy } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

const pct = (value: number) => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`

function lifecycleTone(code: string) {
  if (code === 'BLOCKED' || code === 'DATA_DELAYED' || code === 'INPUT_DELAYED') {
    return 'border-amber-400/35 bg-amber-400/10 text-amber-300'
  }
  if (code === 'NOT_STARTED' || code === 'PAUSED') {
    return 'border-border bg-elevated text-muted'
  }
  return 'border-accent/35 bg-accent/10 text-accent'
}

function HistoricalResults({ strategy }: { strategy: ManagedForwardStrategy }) {
  return (
    <div className="grid gap-2 md:grid-cols-2">
      {strategy.historical_results.map(period => (
        <div key={period.id} className="rounded-card border border-border bg-base/60 px-3 py-2.5">
          <div className="text-[10px] font-medium text-secondary">{period.label}</div>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <div><div className="text-[9px] text-muted">年化</div><div className="mt-0.5 font-mono text-xs text-emerald-400">{pct(period.annualized)}</div></div>
            <div><div className="text-[9px] text-muted">累计</div><div className="mt-0.5 font-mono text-xs text-emerald-400">{pct(period.total_return)}</div></div>
            <div><div className="text-[9px] text-muted">最大回撤</div><div className="mt-0.5 font-mono text-xs text-red-400">{pct(period.max_drawdown)}</div></div>
          </div>
          <div className="mt-2 text-[9px] text-muted">逐年 {period.yearly.map(pct).join(' / ')}</div>
        </div>
      ))}
    </div>
  )
}

export function ManagedForwardStrategyPanel({ view }: { view: 'strategy' | 'backtest' }) {
  const query = useQuery({
    queryKey: QK.managedForwardStrategies,
    queryFn: api.paperManagedStrategies,
    refetchInterval: 30_000,
  })
  const strategy = query.data?.items[0]
  if (!strategy && !query.isError) return null

  if (query.isError) {
    return (
      <section className="rounded-card border border-amber-400/30 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300">
        前向研究策略状态加载失败，普通策略仍可继续使用。
      </section>
    )
  }
  if (!strategy) return null

  const live = strategy.live
  return (
    <section className="overflow-hidden rounded-card border border-accent/25 bg-surface">
      <div className="flex flex-wrap items-start gap-3 px-3.5 py-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-accent/30 bg-accent/10 text-accent">
          {view === 'backtest' ? <BarChart3 className="h-4 w-4" /> : <Activity className="h-4 w-4" />}
        </div>
        <div className="min-w-[16rem] flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs font-semibold text-foreground">{strategy.name}</span>
            <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[9px] text-accent">前向研究 · {strategy.version}</span>
            <span className="rounded bg-elevated px-1.5 py-0.5 text-[9px] text-secondary">专用组合</span>
            <span className={`rounded border px-1.5 py-0.5 text-[9px] ${lifecycleTone(live.lifecycle.code)}`}>{live.lifecycle.label}</span>
          </div>
          <div className="mt-1 text-[10px] leading-4 text-muted">{strategy.description}</div>
          <div className="mt-1 text-[10px] text-secondary">{live.lifecycle.detail} 下一步：{live.lifecycle.next_action}</div>
        </div>
        <Link
          to={`/paper-trading?account=${encodeURIComponent(strategy.account_id)}`}
          className="inline-flex h-8 items-center gap-1 rounded-btn border border-accent/35 bg-accent/10 px-2.5 text-[10px] font-medium text-accent hover:bg-accent/15"
        >
          查看模拟运行 <ArrowRight className="h-3 w-3" />
        </Link>
      </div>

      <details className="group border-t border-border/70 bg-base/25" open={view === 'backtest' ? true : undefined}>
        <summary className="flex cursor-pointer list-none items-center justify-between px-3.5 py-2 text-[10px] text-secondary hover:text-foreground">
          <span>{view === 'backtest' ? '冻结历史验证 · 不参与普通单策略回测' : '查看策略来源、冻结规则和历史验证'}</span>
          <span className="text-muted group-open:hidden">展开</span>
          <span className="hidden text-muted group-open:inline">收起</span>
        </summary>
        <div className="space-y-3 border-t border-border/60 px-3.5 py-3">
          <HistoricalResults strategy={strategy} />
          <div className="grid gap-2 text-[10px] text-secondary md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded border border-border bg-surface px-2.5 py-2"><span className="text-muted">账户：</span>¥{strategy.contract.initial_capital.toLocaleString('zh-CN')} · {strategy.contract.total_slots} 槽</div>
            <div className="rounded border border-border bg-surface px-2.5 py-2"><span className="text-muted">仓位：</span>微盘 {pct(strategy.contract.microcap_weight)} · 事件 {pct(strategy.contract.event_weight)}</div>
            <div className="rounded border border-border bg-surface px-2.5 py-2"><span className="text-muted">调仓：</span>{strategy.contract.rebalance} · 事件最多{strategy.contract.event_lifetime_days}日</div>
            <div className="rounded border border-border bg-surface px-2.5 py-2"><span className="text-muted">前向：</span>至少 {strategy.contract.observation_trading_days} 个真实交易日</div>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[9px] text-muted">
            <span>来源：{strategy.provenance.created_by} · 提交 {strategy.provenance.introduced_commit}</span>
            <span className={strategy.provenance.artifact_verified ? 'text-emerald-400' : 'text-amber-300'}>
              {strategy.provenance.artifact_verified ? '冻结研究文件校验通过' : '冻结研究文件未在当前环境校验'}
            </span>
            <span>{strategy.provenance.note}</span>
          </div>
        </div>
      </details>
    </section>
  )
}

const FLOW = [
  { id: 'data', label: '数据准备', icon: Database },
  { id: 'signal', label: '信号封板', icon: FileCheck2 },
  { id: 'order', label: '订单生成', icon: FileCheck2 },
  { id: 'execution', label: '开盘执行', icon: Clock3 },
  { id: 'observe', label: '观察结算', icon: Activity },
] as const

function flowState(strategy: ManagedForwardStrategy, index: number) {
  const { code, stage } = strategy.live.lifecycle
  if (code === 'BLOCKED') return index === 0 ? 'blocked' : 'waiting'
  if (code === 'SEALED_NO_ORDER') {
    if (index <= 1 || index === 4) return 'complete'
    return 'idle'
  }
  const active = stage === 'data' ? 0 : stage === 'signal' ? 1 : stage === 'execution' ? 3 : 4
  if (index < active) return 'complete'
  if (index === active) return 'active'
  return 'waiting'
}

export function ManagedForwardAccountPanel({ strategy }: { strategy: ManagedForwardStrategy }) {
  const live = strategy.live
  return (
    <section className="overflow-hidden rounded-card border border-accent/30 bg-surface">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-3.5 py-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-foreground">专用前向研究账户</span>
            <span className={`rounded border px-1.5 py-0.5 text-[9px] ${lifecycleTone(live.lifecycle.code)}`}>{live.lifecycle.label}</span>
          </div>
          <div className="mt-1 text-[10px] text-muted">{live.lifecycle.detail}</div>
        </div>
        <div className="text-right text-[10px]"><div className="text-muted">下一步</div><div className="mt-0.5 font-medium text-accent">{live.lifecycle.next_action}</div></div>
      </div>

      <div className="grid grid-cols-2 gap-px bg-border lg:grid-cols-4">
        {[
          ['最近完整数据', live.latest_enriched_date ?? '尚无'],
          ['业绩预告覆盖', live.forecast_covered_through ?? '尚无'],
          ['最近信号封板', live.last_signal_date ?? '尚未封板'],
          ['本账户进展', `${live.signal_count} 信号 · ${live.order_count} 订单 · ${live.fill_count} 成交`],
        ].map(([label, value]) => (
          <div key={label} className="bg-surface px-3 py-2.5"><div className="text-[9px] text-muted">{label}</div><div className="mt-1 font-mono text-[10px] text-secondary">{value}</div></div>
        ))}
      </div>

      <div className="grid grid-cols-5 border-t border-border px-3 py-3">
        {FLOW.map((item, index) => {
          const state = flowState(strategy, index)
          const Icon = state === 'complete' ? Check : state === 'blocked' ? ShieldAlert : item.icon
          const tone = state === 'complete'
            ? 'border-emerald-400/35 bg-emerald-400/10 text-emerald-400'
            : state === 'active'
              ? 'border-accent/40 bg-accent/10 text-accent'
              : state === 'blocked'
                ? 'border-red-400/40 bg-red-400/10 text-red-400'
                : 'border-border bg-base text-muted'
          return (
            <div key={item.id} className="relative flex flex-col items-center gap-1 text-center">
              {index > 0 && <span className="absolute right-1/2 top-3.5 h-px w-full bg-border" aria-hidden="true" />}
              <span className={`relative z-[1] grid h-7 w-7 place-items-center rounded-full border ${tone}`}><Icon className="h-3 w-3" /></span>
              <span className={`text-[9px] ${state === 'active' || state === 'complete' ? 'text-secondary' : 'text-muted'}`}>{item.label}</span>
            </div>
          )
        })}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border bg-base/25 px-3.5 py-2 text-[9px] text-muted">
        <span>{strategy.contract.execution}</span>
        <span>历史结果已冻结 · 前向收益尚未验证</span>
      </div>
    </section>
  )
}
