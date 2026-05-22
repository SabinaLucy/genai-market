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
  const bulletin  = useEndpoint(() => fetchBulletin(),       [])

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
