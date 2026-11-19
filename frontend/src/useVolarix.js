import { useState, useEffect, useCallback } from 'react'
import {
  fetchLatest, fetchPredict, fetchAnalogues,
  fetchShap, fetchBacktest, fetchBulletin,
} from './api'

const POLL_INTERVAL = 60_000

function useEndpoint(fetcher, deps = []) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const load = useCallback(async () => {
    try {
      setError(null)
      const res = await fetcher()
      setData(res.data)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }, deps) // eslint-disable-line

  return { data, loading, error, reload: load }
}

const BULLETIN_CACHE_KEY = 'volarix-bulletin-last-success-v1'
const BULLETIN_RETRY_KEY = 'volarix-bulletin-retry-after-v1'
const BULLETIN_COOLDOWN_MS = 60 * 60 * 1000

function readStored(key) {
  try { return JSON.parse(localStorage.getItem(key) || 'null') } catch { return null }
}

function useCachedBulletin() {
  const [cached, setCached] = useState(() => readStored(BULLETIN_CACHE_KEY))
  const [data, setData] = useState(() => readStored(BULLETIN_CACHE_KEY)?.data || null)
  const [loading, setLoading] = useState(!readStored(BULLETIN_CACHE_KEY)?.data)
  const [error, setError] = useState(null)
  const [rateLimited, setRateLimited] = useState(false)
  const [retryAfter, setRetryAfter] = useState(() => Number(localStorage.getItem(BULLETIN_RETRY_KEY)) || 0)

  const reload = useCallback(async () => {
    const blockedUntil = Number(localStorage.getItem(BULLETIN_RETRY_KEY)) || 0
    if (Date.now() < blockedUntil) {
      setRateLimited(true)
      setRetryAfter(blockedUntil)
      setLoading(false)
      return
    }
    setLoading(!data)
    setError(null)
    try {
      const res = await fetchBulletin()
      setData(res.data)
      const entry = { data: res.data, savedAt: Date.now() }
      setCached(entry)
      try { localStorage.setItem(BULLETIN_CACHE_KEY, JSON.stringify(entry)) } catch { /* storage unavailable */ }
      try { localStorage.removeItem(BULLETIN_RETRY_KEY) } catch { /* storage unavailable */ }
      setRateLimited(false)
      setRetryAfter(0)
    } catch (e) {
      if (e?.response?.status === 429) {
        const retryHeader = e.response?.headers?.['retry-after']
        const parsedSeconds = Number(retryHeader)
        const parsedDate = Date.parse(retryHeader)
        const delay = retryHeader && Number.isFinite(parsedSeconds) ? Math.max(0,parsedSeconds * 1000) : Number.isFinite(parsedDate) ? Math.max(0,parsedDate - Date.now()) : BULLETIN_COOLDOWN_MS
        const until = Date.now() + delay
        setRetryAfter(until)
        setRateLimited(true)
        try { localStorage.setItem(BULLETIN_RETRY_KEY, String(until)) } catch { /* storage unavailable */ }
        setError('AI report request limit reached.')
      } else {
        setError(e?.response?.data?.detail || e.message || 'Request failed')
      }
    } finally { setLoading(false) }
  }, [data])

  return { data, loading, error, reload, rateLimited, retryAfter, cachedAt: cached?.savedAt || null }
}

export default function useVolarix() {
  const [horizon,     setHorizon]     = useState(5)
  const [lastTick,    setLastTick]    = useState(Date.now())
  const [prevRegime,  setPrevRegime]  = useState(null)
  const [regimeAlert, setRegimeAlert] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const latest    = useEndpoint(() => fetchLatest(),         [lastTick])
  const predict   = useEndpoint(() => fetchPredict(horizon), [horizon, lastTick])
  const analogues = useEndpoint(() => fetchAnalogues(),      [])
  const shap      = useEndpoint(() => fetchShap(),           [])
  const backtest  = useEndpoint(() => fetchBacktest(),       [])
  const bulletin  = useCachedBulletin()

  useEffect(() => { latest.reload()    }, [lastTick])
  useEffect(() => { predict.reload()   }, [horizon, lastTick])
  useEffect(() => { analogues.reload() }, [])
  useEffect(() => { shap.reload()      }, [])
  useEffect(() => { backtest.reload()  }, [])
  useEffect(() => { bulletin.reload()  }, [])

  useEffect(() => {
    const id = setInterval(() => setLastTick(Date.now()), POLL_INTERVAL)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (!latest.data) return
    const regime = latest.data.regime
    if (prevRegime && prevRegime !== regime) {
      setRegimeAlert({ from: prevRegime, to: regime, ts: Date.now() })
    }
    setPrevRegime(regime)
  }, [latest.data?.regime])

  useEffect(() => {
    if (!latest.loading) setLastUpdated(new Date())
  }, [lastTick, latest.loading])

  const refresh = () => {
  latest.reload()
  predict.reload()
  setLastTick(Date.now())
}

 return {
  latest, predict, analogues, shap, backtest, bulletin,
  horizon, setHorizon, refresh,
  lastUpdated, regimeAlert, setRegimeAlert,
 }
}
