import { useState, useCallback, useEffect, useRef } from 'react'
import { api } from '@/lib/api'
import { storage } from '@/lib/storage'

export function useStrategyPool() {
  const [pool, setPool] = useState<string[]>(() => storage.strategyPool.get([]))
  const [hydrated, setHydrated] = useState(false)
  const [needsMigration, setNeedsMigration] = useState(false)
  const dirtyRef = useRef(false)
  const poolRef = useRef(pool)

  // 服务端为权威来源，localStorage 仅作为离线兜底与旧版迁移来源。
  useEffect(() => {
    let cancelled = false
    api.strategyPoolPreferences()
      .then(({ strategy_ids }) => {
        if (cancelled || dirtyRef.current) return
        if (strategy_ids == null) {
          setNeedsMigration(true)
          return
        }
        storage.strategyPool.set(strategy_ids)
        poolRef.current = strategy_ids
        setPool(strategy_ids)
      })
      .catch(() => {
        // 后端不可用时继续使用本地缓存，不阻断策略页。
      })
      .finally(() => {
        if (!cancelled) setHydrated(true)
      })
    return () => { cancelled = true }
  }, [])

  const commit = useCallback((updater: (previous: string[]) => string[]) => {
    const previous = poolRef.current
    const next = updater(previous)
    if (next === previous) return
    dirtyRef.current = true
    poolRef.current = next
    storage.strategyPool.set(next)
    setPool(next)
    void api.updateStrategyPoolPreferences(next).catch(() => {})
  }, [])

  const initializePool = useCallback((ids: string[]) => {
    if (!needsMigration) return
    setNeedsMigration(false)
    commit(() => ids)
  }, [commit, needsMigration])

  const addToPool = useCallback((id: string) => {
    commit(prev => prev.includes(id) ? prev : [...prev, id])
  }, [commit])

  const removeFromPool = useCallback((id: string) => {
    commit(prev => prev.includes(id) ? prev.filter(x => x !== id) : prev)
  }, [commit])

  const reorderPool = useCallback((newOrder: string[]) => {
    commit(prev => (
      prev.length === newOrder.length && prev.every((id, index) => id === newOrder[index])
        ? prev
        : newOrder
    ))
  }, [commit])

  // 清除池中不存在于 validIds 的失效策略(如本地开发残留的自定义策略)。
  // 仅当确实有失效项时才更新,避免无谓重渲染。
  const prune = useCallback((validIds: Iterable<string>) => {
    const validSet = validIds instanceof Set ? validIds : new Set(validIds)
    commit(prev => {
      if (prev.length === 0) return prev
      const next = prev.filter(id => validSet.has(id))
      return next.length === prev.length ? prev : next
    })
  }, [commit])

  const isInPool = useCallback((id: string) => pool.includes(id), [pool])

  return {
    pool,
    hydrated,
    needsMigration,
    initializePool,
    addToPool,
    removeFromPool,
    reorderPool,
    prune,
    isInPool,
  }
}
