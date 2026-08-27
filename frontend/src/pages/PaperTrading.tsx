import { PageHeader } from '@/components/PageHeader'
import { PaperTrading as PaperTradingWorkspace } from './backtest/PaperTrading'

export function PaperTrading() {
  return (
    <div className="flex min-h-full flex-col bg-base">
      <PageHeader
        title="模拟交易"
        subtitle={<span className="hidden md:inline">事件驱动执行 · 真实时钟 · 不倒填成交</span>}
        className="shrink-0 bg-base/95 px-3 lg:px-5"
      />
      <main className="min-h-0 flex-1 px-3 pb-3 pt-3 lg:px-4 lg:pb-4">
        <PaperTradingWorkspace />
      </main>
    </div>
  )
}
