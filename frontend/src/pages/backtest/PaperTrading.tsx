import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  BriefcaseBusiness,
  CalendarCheck,
  Clock3,
  Gauge,
  ListChecks,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  Settings2,
  Trash2,
  WalletCards,
} from 'lucide-react'
import {
  api,
  type PaperTradingAccount,
  type PaperTradingConfig,
  type StrategyDetail,
  type StrategyParamDef,
  REGIME_STATE_COLORS,
  REGIME_STATE_LABELS,
} from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { fmtPct, priceColorClass } from '@/lib/format'
import { toast } from '@/components/Toast'
import { EmptyState } from '@/components/EmptyState'
import { Modal } from '@/components/Modal'
import { StrategyNavChart } from './charts/StrategyNavChart'
import {
  BACKTEST_INPUT_CLS,
  buildDefaultOverrides,
  normalizeStrategyOverrides,
  strategyDefaultParams,
} from './StrategyBacktest'

type StrategyGroup = 'all' | 'builtin' | 'custom' | 'ai' | 'composite'
type ResultTab = 'positions' | 'orders' | 'trades'

const STRATEGY_GROUPS: { id: StrategyGroup; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'custom', label: '自定义' },
  { id: 'ai', label: 'AI' },
  { id: 'composite', label: '叠加' },
  { id: 'builtin', label: '内置' },
]

const formatMoney = (value: number | null | undefined) => {
  if (value == null || !Number.isFinite(value)) return '—'
  return value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function NumericField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  suffix,
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
        {suffix && <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-muted">{suffix}</span>}
      </div>
    </label>
  )
}

function StrategyParamInput({
  param,
  value,
  onChange,
}: {
  param: StrategyParamDef
  value: unknown
  onChange: (value: unknown) => void
}) {
  if (param.type === 'bool') {
    return (
      <label className="flex items-center justify-between gap-3 rounded-input border border-border bg-surface px-2.5 py-2 text-xs">
        <span className="text-secondary">{param.label}</span>
        <input type="checkbox" checked={Boolean(value)} onChange={event => onChange(event.target.checked)} />
      </label>
    )
  }
  if (param.type === 'select') {
    return (
      <label>
        <span className="mb-1 block text-[11px] font-medium text-secondary">{param.label}</span>
        <select value={String(value ?? '')} onChange={event => onChange(event.target.value)} className={BACKTEST_INPUT_CLS}>
          {(param.options ?? []).map(option => <option key={option} value={option}>{option}</option>)}
        </select>
      </label>
    )
  }
  return (
    <label>
      <span className="mb-1 block text-[11px] font-medium text-secondary">{param.label}</span>
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

function AccountStat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-card border border-border bg-surface px-3 py-2.5">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={`mt-1 num text-lg font-semibold ${tone ?? 'text-foreground'}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[10px] text-muted">{sub}</div>}
    </div>
  )
}

export function PaperTrading() {
  const queryClient = useQueryClient()
  const [assetType, setAssetType] = useState<'stock' | 'etf'>('stock')
  const [strategyGroup, setStrategyGroup] = useState<StrategyGroup>('all')
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [accountName, setAccountName] = useState('')
  const [initialCapital, setInitialCapital] = useState('200000')
  const [maxPositions, setMaxPositions] = useState('10')
  const [maxExposure, setMaxExposure] = useState('100')
  const [commission, setCommission] = useState('2')
  const [stampTax, setStampTax] = useState('1')
  const [slippage, setSlippage] = useState('5')
  const [entryFill, setEntryFill] = useState<'close_t' | 'open_t+1'>('open_t+1')
  const [exitFill, setExitFill] = useState<'close_t' | 'open_t+1'>('open_t+1')
  const [positionSizing, setPositionSizing] = useState<'equal' | 'score_weight'>('equal')
  const [regimeStates, setRegimeStates] = useState<string[]>([])
  const [regimeMinScore, setRegimeMinScore] = useState<number | ''>('')
  const [strategyParams, setStrategyParams] = useState<Record<string, any>>({})
  const [overrides, setOverrides] = useState<Record<string, any>>({})
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null)
  const [resultTab, setResultTab] = useState<ResultTab>('positions')
  const [deleteTarget, setDeleteTarget] = useState<PaperTradingAccount | null>(null)

  const strategies = useQuery({
    queryKey: QK.screenerStrategies(assetType),
    queryFn: () => api.screenerStrategies(assetType),
  })
  const strategyList = useMemo(() => strategies.data?.presets ?? [], [strategies.data])
  const filteredStrategies = useMemo(() => (
    strategyGroup === 'all' ? strategyList : strategyList.filter(strategy => strategy.source === strategyGroup)
  ), [strategyGroup, strategyList])
  const strategyDetail = useQuery({
    queryKey: QK.strategyDetail(selectedStrategy ?? ''),
    queryFn: () => api.strategyGet(selectedStrategy!),
    enabled: Boolean(selectedStrategy),
  })
  const accounts = useQuery({
    queryKey: QK.paperAccounts,
    queryFn: api.paperAccounts,
    refetchInterval: 60_000,
  })
  const accountItems = useMemo(() => accounts.data?.items ?? [], [accounts.data])
  const account = accountItems.find(item => item.id === selectedAccountId) ?? accountItems[0] ?? null

  useEffect(() => {
    if (!selectedAccountId && accountItems[0]) setSelectedAccountId(accountItems[0].id)
    if (selectedAccountId && accountItems.length && !accountItems.some(item => item.id === selectedAccountId)) {
      setSelectedAccountId(accountItems[0].id)
    }
  }, [accountItems, selectedAccountId])

  useEffect(() => {
    const detail = strategyDetail.data
    if (!detail) return
    setStrategyParams(strategyDefaultParams(detail))
    setOverrides(buildDefaultOverrides(detail))
    setAccountName(`${detail.name} 模拟盘`)
  }, [strategyDetail.data])

  const refreshAccounts = async (selectedId?: string) => {
    await queryClient.invalidateQueries({ queryKey: QK.paperAccounts })
    if (selectedId) setSelectedAccountId(selectedId)
  }

  const createAccount = useMutation({
    mutationFn: async () => {
      if (!selectedStrategy) throw new Error('请先选择策略')
      const payload: PaperTradingConfig & { name: string } = {
        name: accountName.trim() || `${strategyDetail.data?.name ?? selectedStrategy} 模拟盘`,
        strategy_id: selectedStrategy,
        asset_type: assetType,
        symbols: null,
        params: strategyParams,
        overrides: detail ? normalizeStrategyOverrides(detail, overrides) : overrides,
        entry_fill: entryFill,
        exit_fill: exitFill,
        commission_pct: Number(commission) / 10000,
        stamp_tax_pct: Number(stampTax) / 1000,
        slippage_bps: Number(slippage),
        max_positions: Number(maxPositions),
        max_exposure_pct: Number(maxExposure) / 100,
        initial_capital: Number(initialCapital),
        position_sizing: positionSizing,
        holding_days: Number(overrides.max_hold_days ?? 5) || 5,
        minute_fill: false,
        regime_filter: regimeStates.length > 0 || regimeMinScore !== ''
          ? {
              ...(regimeStates.length > 0 ? { states: regimeStates } : {}),
              ...(regimeMinScore !== '' ? { min_score: Number(regimeMinScore) } : {}),
            }
          : null,
        enforce_t_plus_one: true,
      }
      const created = await api.paperAccountCreate(payload)
      return api.paperAccountRun(created.id)
    },
    onSuccess: async created => {
      await refreshAccounts(created.id)
      toast('模拟账户已创建，并同步到最新完整交易日', 'success')
    },
    onError: error => toast(`创建失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const runAccount = useMutation({
    mutationFn: (id: string) => api.paperAccountRun(id),
    onSuccess: async updated => {
      await refreshAccounts(updated.id)
      toast('模拟账户已同步', 'success')
    },
    onError: error => toast(`同步失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const toggleAccount = useMutation({
    mutationFn: (target: PaperTradingAccount) => target.status === 'active'
      ? api.paperAccountPause(target.id)
      : api.paperAccountResume(target.id),
    onSuccess: async updated => {
      await refreshAccounts(updated.id)
      toast(updated.status === 'active' ? '已恢复每日自动运行' : '已暂停每日自动运行', 'success')
    },
    onError: error => toast(`操作失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const deleteAccount = useMutation({
    mutationFn: (id: string) => api.paperAccountDelete(id),
    onSuccess: async deleted => {
      const nextAccount = accountItems.find(item => item.id !== deleted.id) ?? null
      setDeleteTarget(null)
      setSelectedAccountId(nextAccount?.id ?? null)
      await refreshAccounts(nextAccount?.id)
      toast(`已删除模拟账户 · ${deleted.name}`, 'success')
    },
    onError: error => toast(`删除失败 · ${String((error as Error).message || error)}`, 'error'),
  })

  const detail = strategyDetail.data as StrategyDetail | undefined
  const result = account?.result
  const lastPoint = result?.equity_curve?.[result.equity_curve.length - 1]
  const equity = Number(lastPoint?.value ?? account?.config.initial_capital ?? 0)
  const cash = Number(lastPoint?.cash ?? account?.config.initial_capital ?? 0)
  const marketValue = Math.max(equity - cash, 0)
  const totalReturn = Number(result?.stats?.total_return ?? (account ? equity / account.config.initial_capital - 1 : 0))
  const openPositions = result?.open_positions ?? []
  const pendingOrders = result?.pending_orders ?? []
  const trades = result?.trades ?? []
  const accountStrategyName = account
    ? strategyList.find(strategy => strategy.id === account.config.strategy_id)?.name
      ?? account.config.strategy_name
      ?? result?.strategy_info?.name
      ?? account.config.strategy_id
    : ''

  useEffect(() => {
    setResultTab(current => (
      current === 'positions' && openPositions.length === 0 && pendingOrders.length > 0
        ? 'orders'
        : current
    ))
  }, [account?.id, openPositions.length, pendingOrders.length])

  const overrideValue = (key: string) => overrides[key] == null ? '' : String(overrides[key])
  const overridePercentValue = (key: string) => (
    overrides[key] == null ? '' : String(Math.abs(Number(overrides[key])) * 100)
  )
  const updateNullableOverride = (key: string, value: string, transform: (value: number) => number = number => number) => {
    setOverrides(current => ({
      ...current,
      [key]: value === '' ? null : transform(Number(value)),
    }))
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-1 overflow-hidden rounded-card border border-border bg-surface/80 xl:grid-cols-[18rem_minmax(0,1fr)]">
      <section className="space-y-3 border-b border-border bg-base/25 px-3 py-3 xl:overflow-y-auto xl:border-b-0 xl:border-r">
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <label className="text-xs font-medium text-secondary">选择策略</label>
            <div className="inline-flex rounded-btn border border-border bg-surface p-0.5 text-[10px]">
              {(['stock', 'etf'] as const).map(type => (
                <button
                  key={type}
                  type="button"
                  onClick={() => { setAssetType(type); setSelectedStrategy(null) }}
                  className={`rounded-[5px] px-2 py-0.5 ${assetType === type ? 'bg-accent/15 text-accent' : 'text-muted'}`}
                >
                  {type === 'stock' ? '股票' : 'ETF'}
                </button>
              ))}
            </div>
          </div>
          <div className="overflow-hidden rounded-input border border-border bg-surface">
            <div className="flex border-b border-border/60 bg-base/30 p-0.5">
              {STRATEGY_GROUPS.map(group => (
                <button
                  key={group.id}
                  type="button"
                  onClick={() => setStrategyGroup(group.id)}
                  className={`flex-1 rounded-[6px] px-1 py-1 text-[10px] font-medium ${strategyGroup === group.id ? 'bg-accent/15 text-accent' : 'text-muted hover:text-secondary'}`}
                >
                  {group.label}
                </button>
              ))}
            </div>
            <div className="flex max-h-[128px] flex-wrap gap-1 overflow-y-auto p-1">
              {strategies.isLoading && <span className="px-2 py-1 text-xs text-muted">加载中…</span>}
              {!strategies.isLoading && filteredStrategies.length === 0 && <span className="px-2 py-1 text-xs text-muted">当前分组暂无策略</span>}
              {filteredStrategies.map(strategy => (
                <button
                  key={strategy.id}
                  type="button"
                  onClick={() => setSelectedStrategy(strategy.id)}
                  className={`rounded-btn border px-2 py-1 text-[11px] transition-colors ${selectedStrategy === strategy.id ? 'border-accent/50 bg-accent/10 text-accent' : 'border-border bg-base text-secondary hover:border-accent/40'}`}
                >
                  {strategy.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={() => detail && setSettingsOpen(value => !value)}
          disabled={!detail}
          className="w-full rounded-btn border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-accent/40 disabled:opacity-50"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-foreground"><Settings2 className="h-3.5 w-3.5 text-accent" />策略设置</span>
            <span className="text-[10px] text-muted">{settingsOpen ? '收起' : '编辑'}</span>
          </div>
          <div className="mt-1.5 text-[11px] font-medium text-secondary">{detail?.name ?? '选择策略后配置'}</div>
          {detail && (
            <div className="mt-1 text-[10px] leading-4 text-muted">
              参数 {detail.params.length} · 买点 {detail.entry_signals.length} · 卖点 {detail.exit_signals.length} · 最长 {detail.max_hold_days ?? '不限'} 天
            </div>
          )}
        </button>
        {settingsOpen && detail && (
          <div className="grid grid-cols-2 gap-2 rounded-btn border border-accent/20 bg-accent/5 p-2.5">
            {detail.params.length === 0 && <div className="col-span-2 text-[11px] text-muted">该策略没有独立参数，仍可调整评分与风控口径。</div>}
            {detail.params.map(param => (
              <StrategyParamInput
                key={param.id}
                param={param}
                value={strategyParams[param.id]}
                onChange={value => setStrategyParams(current => ({ ...current, [param.id]: value }))}
              />
            ))}
            <div className="col-span-2 mt-1 border-t border-border/60 pt-2 text-[10px] font-medium text-secondary">评分与风控</div>
            <NumericField label="最小评分" value={overrideValue('score_min')} onChange={value => updateNullableOverride('score_min', value)} min={0} max={100} />
            <NumericField label="最大评分" value={overrideValue('score_max')} onChange={value => updateNullableOverride('score_max', value)} min={0} max={100} />
            <NumericField label="止损" value={overridePercentValue('stop_loss')} onChange={value => updateNullableOverride('stop_loss', value, number => -Math.abs(number) / 100)} min={0} max={99} step={0.5} suffix="%" />
            <NumericField label="止盈" value={overridePercentValue('take_profit')} onChange={value => updateNullableOverride('take_profit', value, number => Math.abs(number) / 100)} min={0} max={500} step={0.5} suffix="%" />
            <NumericField label="移动止损" value={overridePercentValue('trailing_stop')} onChange={value => updateNullableOverride('trailing_stop', value, number => -Math.abs(number) / 100)} min={0} max={50} step={0.5} suffix="%" />
            <NumericField label="回撤止盈启动" value={overridePercentValue('trailing_take_profit_activate')} onChange={value => updateNullableOverride('trailing_take_profit_activate', value, number => Math.abs(number) / 100)} min={0} max={200} step={0.5} suffix="%" />
            <NumericField label="回撤止盈回撤" value={overridePercentValue('trailing_take_profit_drawdown')} onChange={value => updateNullableOverride('trailing_take_profit_drawdown', value, number => Math.abs(number) / 100)} min={0} max={50} step={0.5} suffix="点" />
            <NumericField label="最长持仓" value={overrideValue('max_hold_days')} onChange={value => updateNullableOverride('max_hold_days', value, number => Math.round(number))} min={1} step={1} suffix="天" />
          </div>
        )}

        <div className="rounded-btn border border-border bg-surface p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-foreground"><CalendarCheck className="h-3.5 w-3.5 text-accent" />运行口径</div>
          <p className="mt-1.5 text-[10px] leading-4 text-muted">创建时只冻结数据基线，不执行基线日的历史信号。首个新交易日数据完整后，才开始严格前向选股、成交和更新净值。</p>
          <input value={accountName} onChange={event => setAccountName(event.target.value)} placeholder="模拟账户名称" className={`${BACKTEST_INPUT_CLS} mt-2`} />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <label>
            <span className="mb-1 block text-[11px] font-medium text-secondary">建仓口径</span>
            <select value={entryFill} onChange={event => setEntryFill(event.target.value as typeof entryFill)} className={BACKTEST_INPUT_CLS}>
              <option value="open_t+1">次日开盘（推荐）</option>
            </select>
          </label>
          <label>
            <span className="mb-1 block text-[11px] font-medium text-secondary">清仓口径</span>
            <select value={exitFill} onChange={event => setExitFill(event.target.value as typeof exitFill)} className={BACKTEST_INPUT_CLS}>
              <option value="open_t+1">次日开盘</option>
            </select>
          </label>
          <NumericField label="初始资金" value={initialCapital} onChange={setInitialCapital} min={10000} />
          <label>
            <span className="mb-1 block text-[11px] font-medium text-secondary">买入权重</span>
            <select value={positionSizing} onChange={event => setPositionSizing(event.target.value as typeof positionSizing)} className={BACKTEST_INPUT_CLS}>
              <option value="equal">等权买入</option>
              <option value="score_weight">评分加权</option>
            </select>
          </label>
          <NumericField label="最大持仓数" value={maxPositions} onChange={setMaxPositions} min={1} max={100} />
          <NumericField label="最大总仓位" value={maxExposure} onChange={setMaxExposure} min={1} max={100} suffix="%" />
        </div>
        <div className="grid grid-cols-3 gap-2">
          <NumericField label="佣金" value={commission} onChange={setCommission} min={0} suffix="‱" />
          <NumericField label="印花税" value={stampTax} onChange={setStampTax} min={0} suffix="‰" />
          <NumericField label="滑点" value={slippage} onChange={setSlippage} min={0} suffix="bps" />
        </div>
        <div className="text-[10px] leading-4 text-muted">盘后模式仅使用完整落盘数据；次日开盘订单要等下一交易日行情完整后，按当日开盘价、T+1、停牌及涨跌停约束入账，不会倒填账户创建前的成交。</div>

        <button
          type="button"
          onClick={() => createAccount.mutate()}
          disabled={!selectedStrategy || !detail || createAccount.isPending}
          className="flex h-12 w-full items-center justify-center gap-2 rounded-btn bg-accent text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {createAccount.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {createAccount.isPending ? '创建并同步中…' : '创建模拟账户'}
        </button>
      </section>

      <section className="min-h-0 overflow-y-auto bg-base/10 p-3 lg:p-4">
        <div className="mb-3 grid gap-2 rounded-card border border-border bg-surface px-3 py-2.5 text-[10px] leading-4 text-muted md:grid-cols-3">
          <div><span className="font-medium text-foreground">盘后选股</span><br />仅在日线与指标全部落盘后生成新信号。</div>
          <div><span className="font-medium text-foreground">开盘成交</span><br />下一完整交易日按开盘价撮合，并检查 T+1、停牌及涨跌停。</div>
          <div><span className="font-medium text-foreground">退出处理</span><br />当前为盘后日线重放；不伪装成盘中实时盯盘或实时委托。</div>
        </div>

        <div className="mb-3 space-y-1.5 rounded-btn border border-border bg-surface/50 px-3 py-2">
          <div className="flex items-center gap-2">
            <Gauge className="h-3.5 w-3.5 text-accent" />
            <span className="text-xs font-medium text-foreground">环境过滤</span>
            <span className="text-[10px] text-muted">仅在前一日环境满足时入场（防未来函数）</span>
            <div className="ml-auto flex items-center gap-1">
              <span className="text-[10px] text-muted">最低分</span>
              <input
                type="number"
                min={0}
                max={100}
                value={regimeMinScore}
                placeholder="不限"
                onChange={event => setRegimeMinScore(event.target.value ? Number(event.target.value) : '')}
                className="h-6 w-14 rounded border border-border bg-base px-1 text-center text-[11px] text-foreground focus:border-accent/50 focus:outline-none"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            {(Object.keys(REGIME_STATE_LABELS) as (keyof typeof REGIME_STATE_LABELS)[]).map(state => {
              const active = regimeStates.includes(state)
              return (
                <button
                  key={state}
                  type="button"
                  onClick={() => setRegimeStates(current => active ? current.filter(item => item !== state) : [...current, state])}
                  className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[11px] transition-colors ${active ? 'border-transparent text-white' : 'border-border text-muted hover:text-secondary'}`}
                  style={active ? { backgroundColor: REGIME_STATE_COLORS[state] } : undefined}
                >
                  <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: active ? '#fff' : REGIME_STATE_COLORS[state] }} />
                  {REGIME_STATE_LABELS[state]}
                </button>
              )
            })}
            {(regimeStates.length > 0 || regimeMinScore !== '') && (
              <button type="button" onClick={() => { setRegimeStates([]); setRegimeMinScore('') }} className="px-1 text-[10px] text-muted hover:text-danger">清除</button>
            )}
          </div>
        </div>

        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex min-w-0 flex-1 gap-1 overflow-x-auto">
            {accountItems.map(item => (
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedAccountId(item.id)}
                className={`shrink-0 rounded-btn border px-3 py-1.5 text-xs ${account?.id === item.id ? 'border-accent/50 bg-accent/10 text-accent' : 'border-border bg-surface text-secondary'}`}
              >
                <span className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full ${item.status === 'active' ? 'bg-emerald-400' : 'bg-muted'}`} />
                {item.name}
              </button>
            ))}
          </div>
          {account && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => toggleAccount.mutate(account)}
                disabled={toggleAccount.isPending}
                className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-border bg-surface px-2.5 text-xs text-secondary hover:border-accent/40"
              >
                {account.status === 'active' ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                {account.status === 'active' ? '暂停' : '恢复'}
              </button>
              <button
                type="button"
                onClick={() => runAccount.mutate(account.id)}
                disabled={runAccount.isPending}
                title="仅使用本地已完整落盘的数据重算账户，不触发行情补采"
                className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-accent/40 bg-accent/10 px-2.5 text-xs text-accent"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${runAccount.isPending ? 'animate-spin' : ''}`} />
                按现有数据同步
              </button>
              <button
                type="button"
                onClick={() => setDeleteTarget(account)}
                disabled={deleteAccount.isPending}
                title="永久删除该模拟账户及其持仓、订单和成交记录"
                className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-red-500/30 bg-red-500/5 px-2.5 text-xs text-red-400 hover:bg-red-500/10 disabled:opacity-50"
              >
                <Trash2 className="h-3.5 w-3.5" />
                删除
              </button>
            </div>
          )}
        </div>

        {accounts.isLoading && <div className="grid h-full place-items-center text-sm text-muted"><Loader2 className="mr-2 inline h-4 w-4 animate-spin" />加载模拟账户…</div>}
        {!accounts.isLoading && !account && (
          <EmptyState
            icon={WalletCards}
            title="选择策略并创建模拟账户"
            hint="系统会冻结创建时的策略与成交配置；盘后数据完整后自动选股、生成订单并更新持仓与净值。"
          />
        )}

        {account && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-3 rounded-card border border-border bg-surface px-4 py-3">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-semibold text-foreground">{account.name}</h2>
                  <span className={`rounded border px-1.5 py-0.5 text-[10px] ${account.status === 'active' ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400' : 'border-border bg-elevated text-muted'}`}>
                    {account.status === 'active' ? '自动运行' : '已暂停'}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-muted">{accountStrategyName} · {account.config.asset_type === 'stock' ? '股票' : 'ETF'} · 前向信号起始 {account.signal_start_date ?? account.start_date}</div>
              </div>
              <div className="text-right text-[11px] text-muted">
                <div className="flex items-center justify-end gap-1"><CalendarCheck className="h-3.5 w-3.5" />数据截至 {account.last_processed_date ?? '尚未运行'}</div>
                <div className="mt-1 flex items-center justify-end gap-1"><Clock3 className="h-3.5 w-3.5" />{account.last_run_at ? new Date(account.last_run_at).toLocaleString('zh-CN', { hour12: false }) : '等待首次同步'}</div>
              </div>
            </div>

            {account.execution_state && (
              <div className={`rounded-card border px-3 py-2.5 ${account.execution_state.code === 'error' ? 'border-red-500/30 bg-red-500/5' : account.execution_state.code === 'waiting_first_data' || account.execution_state.code === 'waiting_open' || account.execution_state.code === 'waiting_exit' ? 'border-amber-400/30 bg-amber-400/5' : 'border-accent/25 bg-accent/5'}`}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-medium text-foreground">{account.execution_state.label}</span>
                  <span className="text-[10px] text-muted">下一步：{account.execution_state.next_action}</span>
                </div>
                <div className="mt-1 text-[11px] leading-4 text-secondary">{account.execution_state.detail}</div>
              </div>
            )}

            {account.last_error && (
              <div className="flex items-start gap-2 rounded-card border border-red-500/30 bg-red-500/5 px-3 py-2 text-xs text-red-400">
                <AlertTriangle className="mt-px h-4 w-4 shrink-0" />
                <span>{account.last_error}</span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2 lg:grid-cols-5">
              <AccountStat label="账户权益" value={`¥ ${formatMoney(equity)}`} sub={`初始 ¥ ${formatMoney(account.config.initial_capital)}`} />
              <AccountStat label="累计收益" value={fmtPct(totalReturn)} tone={priceColorClass(totalReturn)} sub={`已完成 ${trades.length} 笔`} />
              <AccountStat label="可用现金" value={`¥ ${formatMoney(cash)}`} sub={`现金占比 ${equity > 0 ? fmtPct(cash / equity) : '—'}`} />
              <AccountStat label="持仓市值" value={`¥ ${formatMoney(marketValue)}`} sub={`${openPositions.length} / ${account.config.max_positions} 个持仓`} />
              <AccountStat label="待执行订单" value={`${pendingOrders.length}`} sub={account.config.entry_fill === 'open_t+1' ? '下一交易日开盘' : '当日收盘'} />
            </div>

            {result && result.equity_curve.length > 0 && (
              <div className="rounded-card border border-border bg-surface p-3">
                <div className="mb-2 text-xs font-medium text-foreground">模拟账户净值</div>
                <StrategyNavChart result={result} />
              </div>
            )}

            <div className="overflow-hidden rounded-card border border-border bg-surface">
              <div className="flex items-center justify-between border-b border-border px-3 py-2">
                <div className="inline-flex rounded-btn border border-border bg-base/50 p-0.5">
                  {([
                    ['positions', `当前持仓 ${openPositions.length}`, BriefcaseBusiness],
                    ['orders', `待执行 ${pendingOrders.length}`, ListChecks],
                    ['trades', `历史成交 ${trades.length}`, RefreshCw],
                  ] as const).map(([tab, label, Icon]) => (
                    <button
                      key={tab}
                      type="button"
                      onClick={() => setResultTab(tab)}
                      className={`inline-flex items-center gap-1 rounded-[5px] px-2.5 py-1 text-[11px] ${resultTab === tab ? 'bg-accent text-white' : 'text-secondary'}`}
                    >
                      <Icon className="h-3 w-3" />{label}
                    </button>
                  ))}
                </div>
                <div className="text-[10px] text-muted">与回测相同的交易、费用和风控口径</div>
              </div>

              {resultTab === 'positions' && (
                openPositions.length ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">标的</th><th className="px-3 py-2 text-right font-medium">建仓</th><th className="px-3 py-2 text-right font-medium">持仓</th><th className="px-3 py-2 text-right font-medium">现价 / 市值</th><th className="px-3 py-2 text-right font-medium">浮动盈亏</th><th className="px-3 py-2 text-right font-medium">状态</th></tr></thead>
                      <tbody>{openPositions.map(position => (
                        <tr key={position.symbol} className="border-t border-border hover:bg-elevated/40">
                          <td className="px-3 py-2"><div className="font-medium text-foreground">{position.name || position.symbol}</div><div className="font-mono text-[10px] text-muted">{position.symbol}</div></td>
                          <td className="px-3 py-2 text-right num"><div>{formatMoney(position.entry_price)}</div><div className="text-[10px] text-muted">{position.entry_date}</div></td>
                          <td className="px-3 py-2 text-right num"><div>{position.shares.toLocaleString('zh-CN')} 股</div><div className="text-[10px] text-muted">{position.hold_days} 交易日</div></td>
                          <td className="px-3 py-2 text-right num"><div>{formatMoney(position.market_price)}</div><div className="text-[10px] text-muted">¥ {formatMoney(position.market_value)}</div></td>
                          <td className={`px-3 py-2 text-right num ${priceColorClass(position.unrealized_pnl)}`}><div>{position.unrealized_pnl > 0 ? '+' : ''}{formatMoney(position.unrealized_pnl)}</div><div className="text-[10px]">{fmtPct(position.unrealized_pnl_pct)}</div></td>
                          <td className="px-3 py-2 text-right">{position.pending_exit_reason ? <span className="text-amber-400">待卖 · {position.pending_exit_reason}</span> : <span className="text-emerald-400">持有中</span>}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                ) : <div className="px-4 py-10 text-center text-xs text-muted">当前没有持仓</div>
              )}

              {resultTab === 'orders' && (
                pendingOrders.length ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">标的</th><th className="px-3 py-2 text-right font-medium">信号日</th><th className="px-3 py-2 text-right font-medium">评分</th><th className="px-3 py-2 text-right font-medium">计划成交</th></tr></thead>
                      <tbody>{pendingOrders.map(order => (
                        <tr key={`${order.symbol}-${order.signal_date}`} className="border-t border-border">
                          <td className="px-3 py-2"><div className="font-medium text-foreground">{order.name || order.symbol}</div><div className="font-mono text-[10px] text-muted">{order.symbol}</div></td>
                          <td className="px-3 py-2 text-right num">{order.signal_date}</td>
                          <td className="px-3 py-2 text-right num">{order.score.toFixed(2)}</td>
                          <td className="px-3 py-2 text-right"><div className="text-accent">下一交易日开盘</div><div className="mt-0.5 text-[10px] text-muted">{order.reason ?? '等待完整行情'}</div></td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                ) : <div className="px-4 py-10 text-center text-xs text-muted">当前没有待执行订单</div>
              )}

              {resultTab === 'trades' && (
                trades.length ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead className="bg-elevated text-secondary"><tr><th className="px-3 py-2 text-left font-medium">标的</th><th className="px-3 py-2 text-right font-medium">买入</th><th className="px-3 py-2 text-right font-medium">卖出</th><th className="px-3 py-2 text-right font-medium">净盈亏</th><th className="px-3 py-2 text-right font-medium">原因</th></tr></thead>
                      <tbody>{trades.slice().reverse().map((trade, index) => (
                        <tr key={`${trade.symbol}-${trade.entry_date}-${index}`} className="border-t border-border">
                          <td className="px-3 py-2"><div className="font-medium text-foreground">{trade.name || trade.symbol}</div><div className="font-mono text-[10px] text-muted">{trade.symbol}</div></td>
                          <td className="px-3 py-2 text-right num"><div>{formatMoney(trade.entry_price)}</div><div className="text-[10px] text-muted">{trade.entry_date}</div></td>
                          <td className="px-3 py-2 text-right num"><div>{formatMoney(trade.exit_price)}</div><div className="text-[10px] text-muted">{trade.exit_date}</div></td>
                          <td className={`px-3 py-2 text-right num ${priceColorClass(trade.pnl_pct)}`}><div>{formatMoney(trade.pnl_amount)}</div><div className="text-[10px]">{fmtPct(trade.pnl_pct)}</div></td>
                          <td className="px-3 py-2 text-right text-secondary">{trade.exit_reason}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                ) : <div className="px-4 py-10 text-center text-xs text-muted">尚无已完成交易</div>
              )}
            </div>
          </div>
        )}

        {deleteTarget && (
          <Modal
            onClose={() => !deleteAccount.isPending && setDeleteTarget(null)}
            labelledBy="paper-account-delete-title"
            closeOnBackdrop={!deleteAccount.isPending}
            panelClassName="w-[92vw] max-w-md rounded-card border border-border bg-surface p-5 shadow-2xl"
          >
            <h3 id="paper-account-delete-title" className="text-sm font-semibold text-foreground">删除模拟账户</h3>
            <p className="mt-2 text-xs leading-5 text-secondary">
              确认永久删除「{deleteTarget.name}」？该账户的配置、持仓、待执行订单、净值和历史成交都会一并删除，此操作不可撤销。
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                disabled={deleteAccount.isPending}
                className="rounded-btn bg-elevated px-3 py-1.5 text-xs text-secondary hover:bg-elevated/80 disabled:opacity-50"
              >
                取消
              </button>
              <button
                type="button"
                onClick={() => deleteAccount.mutate(deleteTarget.id)}
                disabled={deleteAccount.isPending}
                className="inline-flex items-center gap-1.5 rounded-btn bg-red-500/15 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/25 disabled:opacity-50"
              >
                {deleteAccount.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                {deleteAccount.isPending ? '删除中…' : '确认永久删除'}
              </button>
            </div>
          </Modal>
        )}
      </section>
    </div>
  )
}
