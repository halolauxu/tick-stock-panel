import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  BriefcaseBusiness,
  CheckCircle2,
  Clock3,
  Database,
  FileClock,
  Gauge,
  History,
  ListChecks,
  Loader2,
  Pause,
  Play,
  Plus,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  WalletCards,
  X,
} from 'lucide-react'

import { EmptyState } from '@/components/EmptyState'
import { Modal } from '@/components/Modal'
import { toast } from '@/components/Toast'
import { fmtPct, priceColorClass } from '@/lib/format'
import {
  api,
  type PaperTradingAccount,
  type PaperTradingConfig,
  type PaperTradingEvent,
  type StrategyDetail,
  type StrategyParamDef,
} from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import {
  BACKTEST_INPUT_CLS,
  buildDefaultOverrides,
  normalizeStrategyOverrides,
  strategyDefaultParams,
} from './StrategyBacktest'
import { StrategyNavChart } from './charts/StrategyNavChart'

type StrategyGroup = 'all' | 'builtin' | 'custom' | 'ai' | 'composite'
type OpsTab = 'positions' | 'orders' | 'fills' | 'cash' | 'performance' | 'incidents'
type ExitMode = 'eod' | 'intraday'

const STRATEGY_GROUPS: { id: StrategyGroup; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'custom', label: '自定义' },
  { id: 'ai', label: 'AI' },
  { id: 'composite', label: '叠加' },
  { id: 'builtin', label: '内置' },
]

const PHASE_LABELS: Record<string, string> = {
  CLOSED: '闭市',
  PRE_MARKET: '盘前等待',
  PREFLIGHT: '盘前校验',
  TRADING: '交易时段',
  LUNCH_BREAK: '午间休市',
  CLOSE_PENDING: '等待结算',
  SETTLEMENT: '收盘结算',
  SIGNAL_SEAL: '信号封板',
}

const STATUS_LABELS: Record<string, string> = {
  PLANNED: '已计划',
  PREFLIGHT_OK: '盘前校验通过',
  FILLED: '已成交',
  PARTIALLY_FILLED: '部分成交',
  REJECTED_LIMIT_UP: '涨停无法买入',
  REJECTED_LIMIT_DOWN: '跌停无法卖出',
  REJECTED_SUSPENDED: '停牌 / 无成交',
  REJECTED_INSUFFICIENT_CASH: '资金不足',
  UNKNOWN_MARKET_DATA: '行情不足 · 等待自动复核',
  EXECUTION_FAILED: '执行失败',
  MISSED_EXECUTION: '错过执行窗口',
  CANCELLED: '已取消',
}

const QUALITY_LABELS: Record<string, string> = {
  ON_TIME: '按时执行',
  RECOVERED_LATE: '延迟恢复',
  NO_RELIABLE_OPEN_DATA: '无可靠开盘证据 · 持续自动复核',
  MISSED_EXECUTION: '错过开盘时钟',
}

const EVENT_ICONS: Record<string, typeof Clock3> = {
  ACCOUNT_CREATED: WalletCards,
  SIGNAL_FROZEN: FileClock,
  ORDER_PLANNED: ListChecks,
  PREFLIGHT_PASSED: ShieldCheck,
  FILLED: CheckCircle2,
  PARTIALLY_FILLED: CheckCircle2,
  ACCOUNT_SETTLED: Database,
  SETTLEMENT_RESTATED: Database,
  MISSED_EXECUTION: ShieldAlert,
  UNKNOWN_MARKET_DATA: AlertTriangle,
}

function money(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '—'
  return value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function shortTime(value: string | null | undefined) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false, timeZone: 'Asia/Shanghai',
  })
}

function clockTime(value: string | null | undefined) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleTimeString('zh-CN', {
    hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Shanghai',
  })
}

function NumericField({
  label, value, onChange, min, max, step, suffix,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  min?: number
  max?: number
  step?: number
  suffix?: string
}) {
  return (
    <label className="min-w-0">
      <span className="mb-1 block text-[11px] font-medium text-secondary">{label}</span>
      <div className="relative">
        <input
          type="number"
          value={value}
          onChange={event => onChange(event.target.value)}
          min={min}
          max={max}
          step={step}
          className={`${BACKTEST_INPUT_CLS} ${suffix ? 'pr-8' : ''}`}
        />
        {suffix && <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-muted">{suffix}</span>}
      </div>
    </label>
  )
}

function StrategyParamInput({ param, value, onChange }: {
  param: StrategyParamDef
  value: unknown
  onChange: (value: unknown) => void
}) {
  if (param.type === 'bool') {
    return (
      <label className="flex items-center justify-between rounded-input border border-border bg-base px-2.5 py-2 text-xs">
        <span className="text-secondary">{param.label}</span>
        <input type="checkbox" checked={Boolean(value)} onChange={event => onChange(event.target.checked)} />
      </label>
    )
  }
  if (param.type === 'select') {
    return (
      <label>
        <span className="mb-1 block text-[11px] text-secondary">{param.label}</span>
        <select value={String(value ?? '')} onChange={event => onChange(event.target.value)} className={BACKTEST_INPUT_CLS}>
          {(param.options ?? []).map(option => <option key={option} value={option}>{option}</option>)}
        </select>
      </label>
    )
  }
  return (
    <label>
      <span className="mb-1 block text-[11px] text-secondary">{param.label}</span>
      <input
        type="number"
        value={String(value ?? '')}
        min={param.min}
        max={param.max}
        step={param.step ?? (param.type === 'int' ? 1 : 'any')}
        onChange={event => onChange(param.type === 'int' ? Number.parseInt(event.target.value, 10) : Number(event.target.value))}
        className={BACKTEST_INPUT_CLS}
      />
    </label>
  )
}

function Metric({ label, value, sub, tone }: { label: string; value: string; sub: string; tone?: string }) {
  return (
    <div className="rounded-card border border-border bg-surface px-3 py-3">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={`mt-1 num text-lg font-semibold ${tone ?? 'text-foreground'}`}>{value}</div>
      <div className="mt-0.5 truncate text-[10px] text-muted">{sub}</div>
    </div>
  )
}

function EventTimeline({ account, events }: { account: PaperTradingAccount; events: PaperTradingEvent[] }) {
  if (!events.length) {
    return <div className="py-10 text-center text-xs text-muted">今天还没有交易事件</div>
  }

  const today = events[0]?.occurred_at.slice(0, 10)
  const pendingOrders = account.orders.filter(order => (
    ['PLANNED', 'PREFLIGHT_OK'].includes(order.status) && order.created_at.slice(0, 10) === today
  ))
  const todayFills = account.fills.filter(fill => fill.executed_at.slice(0, 10) === today)
  const latestSettlement = events.find(event => (
    event.event_type === 'SETTLEMENT_RESTATED' || event.event_type === 'ACCOUNT_SETTLED'
  ))
  const orderById = new Map(account.orders.map(order => [order.id, order]))
  const resolvedOrderIds = new Set(account.orders.filter(order => (
    !['MISSED_EXECUTION', 'UNKNOWN_MARKET_DATA'].includes(order.status)
  )).map(order => order.id))
  const nextPlanDate = pendingOrders.map(order => order.scheduled_date).find(Boolean)
  const nextOpenOnly = pendingOrders.length > 0 && pendingOrders.every(order => order.planned_session === 'NEXT_OPEN')
  const fillFees = todayFills.reduce((sum, fill) => sum + fill.fee_amount, 0)

  const moments: {
    id: string
    at: string
    title: string
    detail: string
    badge?: string
    icon: typeof Clock3
    tone: string
    rows?: string[]
  }[] = []

  if (pendingOrders.length) {
    moments.push({
      id: 'pending-orders',
      at: pendingOrders.reduce((latest, order) => order.created_at > latest ? order.created_at : latest, pendingOrders[0].created_at),
      title: '下一交易日订单已就绪',
      detail: nextOpenOnly ? `${nextPlanDate ?? '下一交易日'} 09:25 自动校验 · 09:30 开盘执行` : '订单将按账户配置的下一有效交易时钟执行',
      badge: `${pendingOrders.length} 笔`,
      icon: ListChecks,
      tone: 'border-accent/35 bg-accent/10 text-accent',
      rows: pendingOrders.map(order => `${order.name || order.symbol} · ${order.side === 'BUY' ? '买入' : '卖出'} ${order.requested_qty.toLocaleString()} 股`),
    })
  }

  if (latestSettlement) {
    moments.push({
      id: 'latest-settlement',
      at: latestSettlement.occurred_at,
      title: latestSettlement.event_type === 'SETTLEMENT_RESTATED' ? '收盘结算已更新' : '今日收盘结算完成',
      detail: latestSettlement.detail,
      badge: '已完成',
      icon: Database,
      tone: 'border-emerald-400/35 bg-emerald-400/10 text-emerald-400',
    })
  }

  if (todayFills.length) {
    moments.push({
      id: 'today-fills',
      at: todayFills.reduce((latest, fill) => fill.executed_at > latest ? fill.executed_at : latest, todayFills[0].executed_at),
      title: '今日成交已入账',
      detail: `${todayFills.length} 笔成交 · 总费用 ¥ ${money(fillFees)}${todayFills.some(fill => fill.quality === 'RECOVERED_LATE') ? ' · 含延迟恢复成交' : ''}`,
      badge: `${todayFills.length} 笔`,
      icon: CheckCircle2,
      tone: 'border-amber-400/35 bg-amber-400/10 text-amber-400',
      rows: todayFills.map(fill => {
        const order = orderById.get(fill.order_id)
        return `${order?.name || fill.symbol} · ${fill.side === 'BUY' ? '买入' : '卖出'} ${fill.quantity.toLocaleString()} 股 @ ${money(fill.price)}`
      }),
    })
  }

  if (account.incidents.some(item => item.status === 'open')) {
    const incidents = account.incidents.filter(item => item.status === 'open')
    moments.push({
      id: 'open-incidents',
      at: incidents.reduce((latest, item) => item.opened_at > latest ? item.opened_at : latest, incidents[0].opened_at),
      title: '存在需要处理的异常',
      detail: incidents[0].detail,
      badge: `${incidents.length} 个`,
      icon: ShieldAlert,
      tone: 'border-red-500/40 bg-red-500/10 text-red-400',
    })
  }

  moments.sort((left, right) => right.at.localeCompare(left.at))

  return (
    <div>
      <div className="divide-y divide-border/60">
        {moments.map(moment => {
          const Icon = moment.icon
          return (
            <div key={moment.id} className="grid grid-cols-[3.25rem_1.75rem_minmax(0,1fr)] gap-3 px-4 py-3.5">
              <time className="pt-1 font-mono text-[11px] text-muted">{clockTime(moment.at)}</time>
              <div className={`grid h-7 w-7 place-items-center rounded-full border ${moment.tone}`}>
                <Icon className="h-3.5 w-3.5" />
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold text-foreground">{moment.title}</span>
                  {moment.badge && <span className="rounded-full bg-elevated px-2 py-0.5 text-[10px] text-secondary">{moment.badge}</span>}
                </div>
                <div className="mt-0.5 text-[11px] leading-4 text-muted">{moment.detail}</div>
                {moment.rows && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {moment.rows.map(row => <span key={row} className="rounded border border-border bg-base px-2 py-1 text-[10px] text-secondary">{row}</span>)}
                  </div>
                )}
              </div>
            </div>
          )
        })}
        {!moments.length && <div className="py-10 text-center text-xs text-muted">今天没有需要关注的账户变化</div>}
      </div>

      <details className="group border-t border-border bg-base/35">
        <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-2.5 text-[11px] text-secondary hover:text-foreground">
          <span>完整审计记录 <span className="ml-1 text-muted">{events.length} 条</span></span>
          <span className="text-[10px] text-muted group-open:hidden">展开</span>
          <span className="hidden text-[10px] text-muted group-open:inline">收起</span>
        </summary>
        <div className="border-t border-border/60 px-4 py-2">
          {events.slice(0, 30).map(event => {
            const Icon = EVENT_ICONS[event.event_type] ?? Clock3
            const resolved = Boolean(
              event.entity_id
              && ['MISSED_EXECUTION', 'UNKNOWN_MARKET_DATA'].includes(event.event_type)
              && resolvedOrderIds.has(event.entity_id)
            )
            const displayTitle = STATUS_LABELS[event.event_type]
              ? `${event.title.split('·')[0].trim()} · ${STATUS_LABELS[event.event_type]}`
              : event.title
            const tone = resolved
              ? 'border-border bg-elevated text-muted'
              : event.severity === 'critical'
                ? 'border-red-500/40 bg-red-500/10 text-red-400'
                : event.severity === 'warning'
                  ? 'border-amber-400/40 bg-amber-400/10 text-amber-400'
                  : 'border-accent/30 bg-accent/10 text-accent'
            return (
              <div key={event.id} className="grid grid-cols-[3.25rem_1.5rem_minmax(0,1fr)] gap-2.5 border-b border-border/40 py-2 last:border-0">
                <time className="pt-0.5 font-mono text-[10px] text-muted">{clockTime(event.occurred_at)}</time>
                <div className={`grid h-6 w-6 place-items-center rounded-full border ${tone}`}><Icon className="h-3 w-3" /></div>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-secondary">
                    <span>{displayTitle}</span>
                    {resolved && <span className="rounded-full bg-elevated px-1.5 py-0.5 text-[9px] text-muted">后续已恢复</span>}
                  </div>
                  <div className="mt-0.5 text-[10px] leading-4 text-muted">{event.detail || event.event_type}</div>
                </div>
              </div>
            )
          })}
        </div>
      </details>
    </div>
  )
}

export function PaperTrading() {
  const queryClient = useQueryClient()
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null)
  const [opsTab, setOpsTab] = useState<OpsTab>('positions')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<PaperTradingAccount | null>(null)
  const [assetType, setAssetType] = useState<'stock' | 'etf'>('stock')
  const [strategyGroup, setStrategyGroup] = useState<StrategyGroup>('all')
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [accountName, setAccountName] = useState('')
  const [exitMode, setExitMode] = useState<ExitMode | null>(null)
  const [initialCapital, setInitialCapital] = useState('200000')
  const [maxPositions, setMaxPositions] = useState('10')
  const [maxExposure, setMaxExposure] = useState('100')
  const [commission, setCommission] = useState('2')
  const [stampTax, setStampTax] = useState('1')
  const [slippage, setSlippage] = useState('5')
  const [positionSizing, setPositionSizing] = useState<'equal' | 'score_weight'>('equal')
  const [strategyParams, setStrategyParams] = useState<Record<string, any>>({})
  const [overrides, setOverrides] = useState<Record<string, any>>({})

  const accounts = useQuery({
    queryKey: QK.paperAccounts,
    queryFn: api.paperAccounts,
    refetchInterval: 10_000,
  })
  const accountItems = useMemo(() => accounts.data?.items ?? [], [accounts.data])
  const account = accountItems.find(item => item.id === selectedAccountId) ?? accountItems[0] ?? null
  const system = accounts.data?.system ?? account?.system

  const strategies = useQuery({
    queryKey: QK.screenerStrategies(assetType),
    queryFn: () => api.screenerStrategies(assetType),
    enabled: drawerOpen,
  })
  const strategyList = useMemo(() => strategies.data?.presets ?? [], [strategies.data])
  const filteredStrategies = useMemo(() => (
    strategyGroup === 'all' ? strategyList : strategyList.filter(item => item.source === strategyGroup)
  ), [strategyGroup, strategyList])
  const strategyDetail = useQuery({
    queryKey: QK.strategyDetail(selectedStrategy ?? ''),
    queryFn: () => api.strategyGet(selectedStrategy!),
    enabled: drawerOpen && Boolean(selectedStrategy),
  })
  const detail = strategyDetail.data as StrategyDetail | undefined

  useEffect(() => {
    if (!selectedAccountId && accountItems[0]) setSelectedAccountId(accountItems[0].id)
    if (selectedAccountId && accountItems.length && !accountItems.some(item => item.id === selectedAccountId)) {
      setSelectedAccountId(accountItems[0].id)
    }
  }, [accountItems, selectedAccountId])

  useEffect(() => {
    if (!detail) return
    setStrategyParams(strategyDefaultParams(detail))
    setOverrides(buildDefaultOverrides(detail))
    setAccountName(`${detail.name} 模拟盘`)
  }, [detail])

  const refresh = async (id?: string) => {
    await queryClient.invalidateQueries({ queryKey: QK.paperAccounts })
    if (id) setSelectedAccountId(id)
  }

  const createAccount = useMutation({
    mutationFn: async () => {
      if (!selectedStrategy || !detail) throw new Error('请先选择策略')
      if (!exitMode) throw new Error('必须选择退出模式')
      const payload: PaperTradingConfig & { name: string } = {
        name: accountName.trim() || `${detail.name} 模拟盘`,
        strategy_id: selectedStrategy,
        strategy_name: detail.name,
        asset_type: assetType,
        symbols: null,
        params: strategyParams,
        overrides: normalizeStrategyOverrides(detail, overrides),
        entry_fill: 'open_t+1',
        exit_fill: exitMode === 'intraday' ? 'signal_next_minute' : 'open_t+1',
        exit_mode: exitMode,
        commission_pct: Number(commission) / 10000,
        stamp_tax_pct: Number(stampTax) / 1000,
        slippage_bps: Number(slippage),
        max_positions: Number(maxPositions),
        max_exposure_pct: Number(maxExposure) / 100,
        initial_capital: Number(initialCapital),
        position_sizing: positionSizing,
        holding_days: Number(overrides.max_hold_days ?? 5) || 5,
        minute_fill: exitMode === 'intraday',
        regime_filter: null,
        enforce_t_plus_one: true,
      }
      return api.paperAccountCreate(payload)
    },
    onSuccess: async created => {
      setDrawerOpen(false)
      await refresh(created.id)
      toast('账户已进入事件账本；不会倒填已过去的成交', 'success')
    },
    onError: error => toast(`创建失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const recoverAccount = useMutation({
    mutationFn: (id: string) => api.paperAccountRun(id),
    onSuccess: async updated => {
      await refresh(updated.id)
      toast('账本核对与错过事件恢复完成', 'success')
    },
    onError: error => toast(`核对失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const toggleAccount = useMutation({
    mutationFn: (target: PaperTradingAccount) => target.status === 'active'
      ? api.paperAccountPause(target.id)
      : api.paperAccountResume(target.id),
    onSuccess: async updated => {
      await refresh(updated.id)
      toast(updated.status === 'active' ? '账户已恢复' : '账户已暂停', 'success')
    },
    onError: error => toast(`操作失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const reconcileAccount = useMutation({
    mutationFn: (id: string) => api.paperAccountReconcile(id),
    onSuccess: async result => {
      await refresh()
      toast(result.ok ? '资金、持仓、成交三账一致' : '对账发现异常，请查看异常标签', result.ok ? 'success' : 'error')
    },
    onError: error => toast(`对账失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const deleteAccount = useMutation({
    mutationFn: (id: string) => api.paperAccountDelete(id),
    onSuccess: async deleted => {
      const next = accountItems.find(item => item.id !== deleted.id)
      setDeleteTarget(null)
      setSelectedAccountId(next?.id ?? null)
      await refresh(next?.id)
      toast(`已从运营界面移除「${deleted.name}」，审计账本保留`, 'success')
    },
    onError: error => toast(`删除失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const nowDate = system?.beijing_time?.slice(0, 10)
  const todayEvents = (account?.timeline ?? []).filter(event => !nowDate || event.occurred_at.slice(0, 10) === nowDate)
  const openIncidents = (account?.incidents ?? []).filter(item => item.status === 'open')
  const quoteAge = system?.tracked_symbol_count === 0 ? '无待执行 / 持仓' : system?.quote_age_ms == null ? '暂无行情' : system.quote_age_ms < 1_000
    ? `${Math.round(system.quote_age_ms)} ms`
    : `${Math.round(system.quote_age_ms / 1000)} 秒`
  const healthTone = system?.executor_health === 'ERROR'
    ? 'border-red-500/40 bg-red-500/10 text-red-400'
    : system?.executor_health === 'DEGRADED'
      ? 'border-amber-400/40 bg-amber-400/10 text-amber-400'
      : 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400'

  const tabs: { id: OpsTab; label: string; count?: number; icon: typeof Clock3 }[] = [
    { id: 'positions', label: '持仓', count: account?.positions.length, icon: BriefcaseBusiness },
    { id: 'orders', label: '订单', count: account?.orders.length, icon: ListChecks },
    { id: 'fills', label: '成交', count: account?.fills.length, icon: CheckCircle2 },
    { id: 'cash', label: '资金流水', count: account?.cash_entries.length, icon: WalletCards },
    { id: 'performance', label: '收益分析', icon: Gauge },
    { id: 'incidents', label: '异常', count: openIncidents.length, icon: ShieldAlert },
  ]

  return (
    <div className="min-h-full space-y-3">
      <section className="rounded-card border border-border bg-surface px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted">市场阶段</div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs font-medium text-foreground">
              <Clock3 className="h-3.5 w-3.5 text-accent" />
              {PHASE_LABELS[system?.market_phase ?? ''] ?? '状态加载中'}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted">北京时间</div>
            <div className="mt-0.5 font-mono text-xs text-foreground">{shortTime(system?.beijing_time)}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted">行情新鲜度</div>
            <div className={`mt-0.5 text-xs font-medium ${system?.quote_stale && system?.market_phase === 'TRADING' ? 'text-red-400' : 'text-secondary'}`}>{quoteAge} · {system?.quote_source_mode ?? '未接入'}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted">执行器</div>
            <div className={`mt-0.5 inline-flex rounded border px-1.5 py-0.5 text-[10px] font-semibold ${healthTone}`}>{system?.executor_health ?? 'LOADING'}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted">订阅范围</div>
            <div className="mt-0.5 text-xs text-secondary">{system?.tracked_symbol_count ?? 0} 个订单 / 持仓标的</div>
          </div>
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-semibold text-white"
          >
            <Plus className="h-3.5 w-3.5" />新建模拟账户
          </button>
        </div>
      </section>

      {(system?.executor_health === 'ERROR' || (system?.quote_stale && system.market_phase === 'TRADING') || account?.reconciliation?.ok === false) && (
        <section className="flex items-start gap-2 rounded-card border border-red-500/40 bg-red-500/10 px-3 py-2.5 text-xs text-red-300">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <div className="font-semibold">交易状态存在阻断风险，不能视为正常自动运行</div>
            <div className="mt-0.5 text-[11px] text-red-300/80">
              {account?.reconciliation?.ok === false ? '三账对账不一致。' : ''}
              {system?.quote_stale && system.market_phase === 'TRADING' ? '交易时段行情已过期。' : ''}
              {system?.critical_incident_count ? `仍有 ${system.critical_incident_count} 个关键异常。` : ''}
            </div>
          </div>
        </section>
      )}

      <section className="rounded-card border border-border bg-surface p-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <WalletCards className="h-4 w-4 text-accent" />
          <span className="text-xs font-medium text-foreground">账户</span>
          <span className="rounded-full bg-elevated px-1.5 py-0.5 text-[10px] text-muted">{accountItems.length}</span>
          <div className="flex min-w-0 flex-1 gap-1.5 overflow-x-auto">
            {accountItems.map(item => (
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedAccountId(item.id)}
                className={`shrink-0 rounded-btn border px-3 py-1.5 text-xs ${account?.id === item.id ? 'border-accent bg-accent/15 text-accent' : 'border-border bg-base text-secondary'}`}
              >
                <span className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full ${item.status === 'active' ? 'bg-emerald-400' : 'bg-muted'}`} />
                {item.name}
              </button>
            ))}
          </div>
        </div>
      </section>

      {accounts.isLoading && <div className="grid min-h-[24rem] place-items-center text-sm text-muted"><Loader2 className="h-5 w-5 animate-spin" /></div>}
      {!accounts.isLoading && !account && (
        <div className="rounded-card border border-border bg-surface">
          <EmptyState icon={WalletCards} title="还没有模拟账户" hint="新建账户时会冻结策略与执行口径；账户只记录创建后真实发生的事件。" />
        </div>
      )}

      {account && (
        <>
          <section className="rounded-card border border-border bg-surface px-4 py-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-[10px] font-medium uppercase tracking-wide text-accent">当前账户</div>
                <div className="mt-1 flex flex-wrap items-center gap-2">
                  <h2 className="text-base font-semibold text-foreground">{account.name}</h2>
                  <span className={`rounded border px-1.5 py-0.5 text-[10px] ${account.status === 'active' ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400' : 'border-border bg-elevated text-muted'}`}>{account.status === 'active' ? '事件时钟运行中' : '已暂停'}</span>
                  <span className="rounded bg-elevated px-1.5 py-0.5 text-[10px] text-secondary">{account.config.exit_mode === 'intraday' ? '盘中风险模式' : '盘后退出模式'}</span>
                </div>
                <div className="mt-1 text-[10px] text-muted">
                  {account.config.strategy_name ?? account.config.strategy_id} · 信号起始 {account.signal_start_date} · 最近封板 {account.last_processed_date ?? '尚未封板'}
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <button type="button" onClick={() => toggleAccount.mutate(account)} className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-border px-2.5 text-xs text-secondary">
                  {account.status === 'active' ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}{account.status === 'active' ? '暂停' : '恢复'}
                </button>
                <button type="button" onClick={() => recoverAccount.mutate(account.id)} className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-accent/40 bg-accent/10 px-2.5 text-xs text-accent">
                  <RefreshCw className={`h-3.5 w-3.5 ${recoverAccount.isPending ? 'animate-spin' : ''}`} />核对 / 恢复
                </button>
                <button type="button" onClick={() => reconcileAccount.mutate(account.id)} className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-border px-2.5 text-xs text-secondary">
                  <ShieldCheck className="h-3.5 w-3.5" />三账对账
                </button>
                <button type="button" onClick={() => setDeleteTarget(account)} className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-red-500/30 bg-red-500/5 px-2.5 text-xs text-red-400">
                  <Trash2 className="h-3.5 w-3.5" />删除此账户
                </button>
              </div>
            </div>
          </section>

          {openIncidents.length > 0 && (
            <section className="rounded-card border border-amber-400/35 bg-amber-400/5 px-3 py-2.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-amber-300"><ShieldAlert className="h-4 w-4" />{openIncidents.length} 个未关闭异常</div>
                <button type="button" onClick={() => setOpsTab('incidents')} className="text-[10px] text-amber-300">查看原因</button>
              </div>
              <div className="mt-1 text-[11px] text-amber-200/80">{openIncidents[0].title} · {openIncidents[0].detail}</div>
            </section>
          )}

          <section className="grid grid-cols-2 gap-2 lg:grid-cols-5">
            <Metric label="账户权益" value={`¥ ${money(account.summary.equity)}`} sub={`初始 ¥ ${money(account.config.initial_capital)}`} />
            <Metric label="可用现金" value={`¥ ${money(account.summary.cash)}`} sub={`现金占比 ${fmtPct(account.summary.equity ? account.summary.cash / account.summary.equity : 0)}`} />
            <Metric label="持仓市值" value={`¥ ${money(account.summary.market_value)}`} sub={`${account.summary.position_count} / ${account.config.max_positions} 个持仓`} />
            <Metric
              label="今日盈亏"
              value={account.summary.today_pnl_available && account.summary.today_pnl != null ? `${account.summary.today_pnl >= 0 ? '+' : ''}¥ ${money(account.summary.today_pnl)}` : '—'}
              sub={account.summary.today_pnl_available ? `风险敞口 ${fmtPct(account.summary.exposure)}` : '等待当日有效行情'}
              tone={account.summary.today_pnl != null ? priceColorClass(account.summary.today_pnl) : 'text-muted'}
            />
            <Metric label="订单 / 异常" value={`${account.summary.pending_order_count} / ${account.summary.open_incident_count}`} sub={account.reconciliation.ok ? '三账一致' : '三账不平'} tone={account.reconciliation.ok ? 'text-foreground' : 'text-red-400'} />
          </section>

          <section className="overflow-hidden rounded-card border border-border bg-surface">
            <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
              <div>
                <div className="flex items-center gap-2"><History className="h-4 w-4 text-accent" /><span className="text-xs font-semibold text-foreground">今日关键进展</span></div>
                <div className="mt-0.5 pl-6 text-[10px] text-muted">默认只展示影响账户结果的节点</div>
              </div>
              <span className="text-[10px] text-muted">北京时间 · 完整技术记录可展开</span>
            </div>
            <EventTimeline account={account} events={todayEvents} />
          </section>

          <section className="overflow-hidden rounded-card border border-border bg-surface">
            <div className="flex gap-1 overflow-x-auto border-b border-border px-2 py-2">
              {tabs.map(({ id, label, count, icon: Icon }) => (
                <button key={id} type="button" onClick={() => setOpsTab(id)} className={`inline-flex shrink-0 items-center gap-1 rounded-btn px-2.5 py-1.5 text-[11px] ${opsTab === id ? 'bg-accent text-white' : 'text-secondary hover:bg-elevated'}`}>
                  <Icon className="h-3.5 w-3.5" />{label}{count != null && <span className="opacity-70">{count}</span>}
                </button>
              ))}
            </div>

            {opsTab === 'positions' && (
              account.positions.length ? <div className="overflow-x-auto"><table className="w-full text-xs">
                <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">标的</th><th className="px-3 py-2 text-right font-medium">成交成本</th><th className="px-3 py-2 text-right font-medium">数量 / 可卖</th><th className="px-3 py-2 text-right font-medium">现价 / 市值</th><th className="px-3 py-2 text-right font-medium">浮动盈亏</th><th className="px-3 py-2 text-right font-medium">行情与风险</th></tr></thead>
                <tbody>{account.positions.map(position => {
                  const pnlPct = position.cost_basis ? position.unrealized_pnl / position.cost_basis : 0
                  const positionQuoteAge = position.quote_at && system?.beijing_time
                    ? Date.parse(system.beijing_time) - Date.parse(position.quote_at)
                    : Number.POSITIVE_INFINITY
                  const positionQuoteStale = system?.market_phase === 'TRADING' && positionQuoteAge > 90_000
                  return <tr key={position.symbol} className="border-t border-border align-top">
                    <td className="px-3 py-2"><div className="font-medium text-foreground">{position.name || position.symbol}</div><div className="font-mono text-[10px] text-muted">{position.symbol}</div></td>
                    <td className="px-3 py-2 text-right num"><div>{money(position.average_price)}</div><div className="text-[10px] text-muted">{position.acquired_on}</div></td>
                    <td className="px-3 py-2 text-right num"><div>{position.quantity.toLocaleString()} 股</div><div className="text-[10px] text-muted">可卖 {position.available_qty} · 锁定 {position.locked_qty}</div></td>
                    <td className="px-3 py-2 text-right num"><div>{money(position.last_price)}</div><div className="text-[10px] text-muted">¥ {money(position.market_value)}</div></td>
                    <td className={`px-3 py-2 text-right num ${priceColorClass(position.unrealized_pnl)}`}><div>{position.unrealized_pnl >= 0 ? '+' : ''}{money(position.unrealized_pnl)}</div><div className="text-[10px]">{fmtPct(pnlPct)}</div></td>
                    <td className="px-3 py-2 text-right"><div className={positionQuoteStale ? 'text-red-400' : position.pending_exit_reason ? 'text-amber-400' : 'text-emerald-400'}>{positionQuoteStale ? '行情已过期' : position.pending_exit_reason ? `已触发 ${position.pending_exit_reason}` : '监控中'}</div><div className="mt-0.5 text-[10px] text-muted">{shortTime(position.quote_at)} · {position.quote_source ?? '无行情源'}</div></td>
                  </tr>
                })}</tbody>
              </table></div> : <div className="py-12 text-center text-xs text-muted">当前没有真实持仓</div>
            )}

            {opsTab === 'orders' && (
              account.orders.length ? <div className="overflow-x-auto"><table className="w-full text-xs">
                <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">订单</th><th className="px-3 py-2 text-right font-medium">信号 / 计划</th><th className="px-3 py-2 text-right font-medium">数量</th><th className="px-3 py-2 text-right font-medium">生命周期终态</th><th className="px-3 py-2 text-right font-medium">真实原因</th></tr></thead>
                <tbody>{account.orders.map(order => {
                  const checkedAt = typeof order.preflight.checked_at === 'string' ? order.preflight.checked_at : null
                  const lifecycle = [
                    `信号冻结 ${order.signal_date}`,
                    `订单计划 ${shortTime(order.created_at)}`,
                    checkedAt ? `盘前校验 ${shortTime(checkedAt)}` : '待盘前校验',
                    order.terminal_at ? `${STATUS_LABELS[order.status] ?? order.status} ${shortTime(order.terminal_at)}` : '等待执行时钟',
                  ].join(' → ')
                  return <tr key={order.id} className="border-t border-border align-top">
                    <td className="px-3 py-2"><div className="font-medium text-foreground">{order.name || order.symbol}</div><div className="font-mono text-[10px] text-muted">{order.symbol} · {order.side}</div></td>
                    <td className="px-3 py-2 text-right"><div>{order.signal_date}</div><div className="text-[10px] text-muted">{order.planned_session === 'NEXT_OPEN' ? '下一交易日开盘' : '下一有效行情'}</div></td>
                    <td className="px-3 py-2 text-right num"><div>{order.filled_qty} / {order.requested_qty}</div><div className="text-[10px] text-muted">评分 {order.score?.toFixed(2) ?? '—'}</div></td>
                    <td className="max-w-[24rem] px-3 py-2 text-right"><div className={order.status.includes('FAILED') || order.status.includes('MISSED') || order.status.includes('UNKNOWN') ? 'text-red-400' : order.status.startsWith('REJECTED') ? 'text-amber-400' : order.status === 'FILLED' ? 'text-emerald-400' : 'text-accent'}>{STATUS_LABELS[order.status] ?? order.status}</div><div className="text-[10px] text-muted">{order.execution_quality ? `${QUALITY_LABELS[order.execution_quality] ?? order.execution_quality} · ${order.execution_quality}` : '尚未执行'}</div><div className="mt-1 text-[9px] leading-3 text-muted">{lifecycle}</div></td>
                    <td className="max-w-[22rem] px-3 py-2 text-right text-[11px] leading-4 text-secondary">{order.reason}</td>
                  </tr>
                })}</tbody>
              </table></div> : <div className="py-12 text-center text-xs text-muted">当前没有订单</div>
            )}

            {opsTab === 'fills' && (
              account.fills.length ? <div className="overflow-x-auto"><table className="w-full text-xs">
                <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">标的</th><th className="px-3 py-2 text-right font-medium">方向 / 数量</th><th className="px-3 py-2 text-right font-medium">成交价 / 金额</th><th className="px-3 py-2 text-right font-medium">费用</th><th className="px-3 py-2 text-right font-medium">质量 / 行情证据</th></tr></thead>
                <tbody>{account.fills.map(fill => <tr key={fill.id} className="border-t border-border">
                  <td className="px-3 py-2 font-mono text-foreground">{fill.symbol}</td>
                  <td className="px-3 py-2 text-right"><span className={fill.side === 'BUY' ? 'text-red-400' : 'text-emerald-400'}>{fill.side}</span> · {fill.quantity.toLocaleString()} 股</td>
                  <td className="px-3 py-2 text-right num"><div>{money(fill.price)}</div><div className="text-[10px] text-muted">¥ {money(fill.gross_amount)}</div></td>
                  <td className="px-3 py-2 text-right num">¥ {money(fill.fee_amount)}</td>
                  <td className="px-3 py-2 text-right"><div className={fill.quality === 'RECOVERED_LATE' ? 'text-amber-400' : 'text-emerald-400'}>{QUALITY_LABELS[fill.quality] ?? fill.quality}</div><div className="text-[10px] text-muted">{fill.quality} · {shortTime(fill.quote_at)} · {fill.source}</div></td>
                </tr>)}</tbody>
              </table></div> : <div className="py-12 text-center text-xs text-muted">尚无真实成交</div>
            )}

            {opsTab === 'cash' && (
              account.cash_entries.length ? <div className="overflow-x-auto"><table className="w-full text-xs">
                <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">时间 / 类型</th><th className="px-3 py-2 text-left font-medium">说明</th><th className="px-3 py-2 text-right font-medium">发生额</th><th className="px-3 py-2 text-right font-medium">变动后余额</th></tr></thead>
                <tbody>{account.cash_entries.map(entry => <tr key={entry.id} className="border-t border-border">
                  <td className="px-3 py-2"><div>{shortTime(entry.occurred_at)}</div><div className="text-[10px] text-muted">{entry.event_type}</div></td><td className="px-3 py-2 text-secondary">{entry.detail}</td><td className={`px-3 py-2 text-right num ${priceColorClass(entry.amount)}`}>{entry.amount >= 0 ? '+' : ''}¥ {money(entry.amount)}</td><td className="px-3 py-2 text-right num">¥ {money(entry.balance_after)}</td>
                </tr>)}</tbody>
              </table></div> : <div className="py-12 text-center text-xs text-muted">没有资金流水</div>
            )}

            {opsTab === 'performance' && (
              <div className="p-3">
                <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Metric label="累计收益" value={fmtPct(account.summary.total_return)} sub="事件账本净值" tone={priceColorClass(account.summary.total_return)} />
                  <Metric label="浮动盈亏" value={`¥ ${money(account.summary.unrealized_pnl)}`} sub="当前持仓" tone={priceColorClass(account.summary.unrealized_pnl)} />
                  <Metric label="风险敞口" value={fmtPct(account.summary.exposure)} sub={`上限 ${fmtPct(account.config.max_exposure_pct)}`} />
                  <Metric label="成交笔数" value={`${account.fills.length}`} sub="不把待执行订单计为成交" />
                </div>
                {account.result.equity_curve.length ? <StrategyNavChart result={account.result} /> : <div className="py-14 text-center text-xs text-muted">完成首次 15:05 结算后生成净值曲线</div>}
              </div>
            )}

            {opsTab === 'incidents' && (
              account.incidents.length ? <div className="divide-y divide-border">{account.incidents.map(incident => <div key={incident.id} className="flex items-start gap-3 px-3 py-3">
                <ShieldAlert className={`mt-0.5 h-4 w-4 shrink-0 ${incident.severity === 'critical' ? 'text-red-400' : 'text-amber-400'}`} />
                <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="text-xs font-medium text-foreground">{incident.title}</span><span className={`rounded px-1.5 py-0.5 text-[9px] ${incident.status === 'open' ? 'bg-red-500/10 text-red-400' : 'bg-elevated text-muted'}`}>{incident.status === 'open' ? '未关闭' : '已解决'}</span><span className="font-mono text-[9px] text-muted">{incident.code}</span></div><div className="mt-1 text-[11px] leading-4 text-secondary">{incident.detail}</div><div className="mt-1 text-[10px] text-muted">发生 {shortTime(incident.opened_at)}{incident.resolved_at ? ` · 解决 ${shortTime(incident.resolved_at)}` : ''}</div></div>
              </div>)}</div> : <div className="py-12 text-center text-xs text-muted"><ShieldCheck className="mx-auto mb-2 h-5 w-5 text-emerald-400" />没有异常记录</div>
            )}
          </section>
        </>
      )}

      {drawerOpen && (
        <Modal
          onClose={() => !createAccount.isPending && setDrawerOpen(false)}
          labelledBy="paper-create-title"
          overlayClassName="fixed inset-0 z-50 flex justify-end bg-black/55 backdrop-blur-sm"
          panelClassName="h-full w-[96vw] max-w-xl overflow-y-auto border-l border-border bg-surface shadow-2xl"
        >
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-surface/95 px-4 py-3 backdrop-blur">
            <div><h3 id="paper-create-title" className="text-sm font-semibold text-foreground">新建模拟账户</h3><p className="mt-0.5 text-[10px] text-muted">冻结策略、风控与执行口径；只记录创建后真实发生的事件</p></div>
            <button type="button" onClick={() => setDrawerOpen(false)} className="rounded-btn p-1.5 text-muted hover:bg-elevated"><X className="h-4 w-4" /></button>
          </div>
          <div className="space-y-4 p-4">
            <div>
              <div className="mb-2 flex items-center justify-between"><span className="text-xs font-semibold text-foreground">1. 选择策略</span><div className="inline-flex rounded-btn border border-border p-0.5 text-[10px]">{(['stock', 'etf'] as const).map(type => <button key={type} type="button" onClick={() => { setAssetType(type); setSelectedStrategy(null) }} className={`rounded px-2 py-1 ${assetType === type ? 'bg-accent/15 text-accent' : 'text-muted'}`}>{type === 'stock' ? '股票' : 'ETF'}</button>)}</div></div>
              <div className="rounded-input border border-border bg-base/50"><div className="flex border-b border-border p-1">{STRATEGY_GROUPS.map(group => <button key={group.id} type="button" onClick={() => setStrategyGroup(group.id)} className={`flex-1 rounded px-1 py-1 text-[10px] ${strategyGroup === group.id ? 'bg-accent/15 text-accent' : 'text-muted'}`}>{group.label}</button>)}</div><div className="flex max-h-36 flex-wrap gap-1 overflow-y-auto p-2">{filteredStrategies.map(strategy => <button key={strategy.id} type="button" onClick={() => setSelectedStrategy(strategy.id)} className={`rounded-btn border px-2 py-1 text-[11px] ${selectedStrategy === strategy.id ? 'border-accent/50 bg-accent/10 text-accent' : 'border-border text-secondary'}`}>{strategy.name}</button>)}{strategies.isLoading && <span className="text-xs text-muted">加载策略…</span>}</div></div>
            </div>

            <div>
              <div className="mb-2 text-xs font-semibold text-foreground">2. 必选退出模式</div>
              <div className="grid gap-2 md:grid-cols-2">
                <button type="button" onClick={() => setExitMode('eod')} className={`rounded-card border p-3 text-left ${exitMode === 'eod' ? 'border-accent bg-accent/10' : 'border-border bg-base/40'}`}><div className="text-xs font-semibold text-foreground">盘后退出模式</div><div className="mt-1 text-[10px] leading-4 text-muted">收盘后判断止损、止盈与卖点，下一交易日开盘执行。</div></button>
                <button type="button" onClick={() => setExitMode('intraday')} className={`rounded-card border p-3 text-left ${exitMode === 'intraday' ? 'border-accent bg-accent/10' : 'border-border bg-base/40'}`}><div className="text-xs font-semibold text-foreground">盘中风险模式</div><div className="mt-1 text-[10px] leading-4 text-muted">实时或分钟行情触发，下一有效行情执行；当天买入仍受 T+1 锁定。</div></button>
              </div>
            </div>

            <div>
              <div className="mb-2 text-xs font-semibold text-foreground">3. 账户与成交口径</div>
              <input value={accountName} onChange={event => setAccountName(event.target.value)} placeholder="模拟账户名称" className={BACKTEST_INPUT_CLS} />
              <div className="mt-2 grid grid-cols-2 gap-2"><NumericField label="初始资金" value={initialCapital} onChange={setInitialCapital} min={10000} /><label><span className="mb-1 block text-[11px] text-secondary">买入权重</span><select value={positionSizing} onChange={event => setPositionSizing(event.target.value as typeof positionSizing)} className={BACKTEST_INPUT_CLS}><option value="equal">等权买入</option><option value="score_weight">评分加权</option></select></label><NumericField label="最大持仓数" value={maxPositions} onChange={setMaxPositions} min={1} max={100} /><NumericField label="最大总仓位" value={maxExposure} onChange={setMaxExposure} min={1} max={100} suffix="%" /></div>
              <div className="mt-2 grid grid-cols-3 gap-2"><NumericField label="佣金" value={commission} onChange={setCommission} min={0} suffix="‱" /><NumericField label="印花税" value={stampTax} onChange={setStampTax} min={0} suffix="‰" /><NumericField label="滑点" value={slippage} onChange={setSlippage} min={0} suffix="bps" /></div>
            </div>

            {detail && (
              <details className="rounded-card border border-border bg-base/40 p-3">
                <summary className="cursor-pointer text-xs font-semibold text-foreground">4. 策略参数与风控（{detail.params.length} 项）</summary>
                <div className="mt-3 grid grid-cols-2 gap-2">{detail.params.map(param => <StrategyParamInput key={param.id} param={param} value={strategyParams[param.id]} onChange={value => setStrategyParams(current => ({ ...current, [param.id]: value }))} />)}</div>
              </details>
            )}

            <div className="rounded-card border border-amber-400/25 bg-amber-400/5 p-3 text-[10px] leading-4 text-amber-200/80">订单只会在真实时钟到达后推进。服务错过 09:30 时会记录 MISSED_EXECUTION；只有存在可靠开盘或分钟证据才允许 RECOVERED_LATE，绝不倒填为准时成交。</div>
            <button type="button" onClick={() => createAccount.mutate()} disabled={!selectedStrategy || !detail || !exitMode || createAccount.isPending} className="flex h-11 w-full items-center justify-center gap-2 rounded-btn bg-accent text-sm font-semibold text-white disabled:opacity-50">{createAccount.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}{createAccount.isPending ? '写入事件账本…' : '创建并冻结账户'}</button>
          </div>
        </Modal>
      )}

      {deleteTarget && (
        <Modal onClose={() => !deleteAccount.isPending && setDeleteTarget(null)} labelledBy="paper-delete-title" closeOnBackdrop={!deleteAccount.isPending} panelClassName="w-[92vw] max-w-md rounded-card border border-border bg-surface p-5 shadow-2xl">
          <h3 id="paper-delete-title" className="text-sm font-semibold text-foreground">从运营界面移除「{deleteTarget.name}」</h3>
          <p className="mt-2 text-xs leading-5 text-secondary">操作只针对当前高亮账户。账户会停止运行并从页面隐藏，但信号、订单、成交、资金流水和异常审计仍保留在事件账本中。</p>
          <div className="mt-5 flex justify-end gap-2"><button type="button" onClick={() => setDeleteTarget(null)} className="rounded-btn bg-elevated px-3 py-1.5 text-xs text-secondary">取消</button><button type="button" onClick={() => deleteAccount.mutate(deleteTarget.id)} disabled={deleteAccount.isPending} className="inline-flex items-center gap-1.5 rounded-btn bg-red-500/15 px-3 py-1.5 text-xs font-medium text-red-400 disabled:opacity-50">{deleteAccount.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}确认移除</button></div>
        </Modal>
      )}
    </div>
  )
}
