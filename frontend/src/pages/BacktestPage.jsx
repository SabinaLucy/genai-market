import PageHeader from '../components/PageHeader'
import useIsMobile from '../useIsMobile'
import { fmt, pct } from '../utils'

const STRATEGIES = [
  { key: 'spy_stats',   label: 'SPY Buy and Hold',   color: '#22c55e', text: '#4ade80',  cls: 'sc-green'  },
  { key: 'naive_stats', label: 'Naive Regime Switch', color: '#f59e0b', text: '#fbbf24', cls: 'sc-amber'  },
  { key: 'hyst_stats',  label: 'Hysteresis Strategy', color: '#8b5cf6', text: '#c4b5fd', cls: 'sc-purple' },
]
const COLS = [
  { key: 'total_return',   label: 'Return', fmt: pct },
  { key: 'sharpe_ratio',   label: 'Sharpe', fmt: v => fmt(v, 3) },
  { key: 'max_drawdown',   label: 'Max DD',  fmt: pct },
]
const COLS_FULL = [
  { key: 'total_return',   label: 'Total Return', fmt: pct },
  { key: 'ann_return',     label: 'Ann. Return',  fmt: pct },
  { key: 'ann_volatility', label: 'Volatility',   fmt: pct },
  { key: 'sharpe_ratio',   label: 'Sharpe',       fmt: v => fmt(v, 3) },
  { key: 'max_drawdown',   label: 'Max Drawdown', fmt: pct },
]

export default function BacktestPage({ backtest }) {
  const isMobile = useIsMobile()
  const d          = backtest.data
  const sharpes    = STRATEGIES.map(s => d?.[s.key]?.sharpe_ratio ?? -Infinity)
  const bestSharpe = Math.max(...sharpes)
  const px         = isMobile ? 12 : 32

  return (
    <div className="fade-up">
      <PageHeader title="Strategy Backtest" subtitle="SPY vs Naive vs Hysteresis · 2022 to 2026" page="backtest" />
      <div style={{ padding: `${isMobile ? 14 : 28}px ${px}px`, display: 'flex', flexDirection: 'column', gap: isMobile ? 12 : 20 }}>

        {backtest.loading ? (
          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'repeat(3,1fr)', gap: 12 }}>
            {[1,2,3].map(i => <div key={i} className="card" style={{ padding: 20, height: 180 }}><div className="skeleton" style={{ height: '100%' }} /></div>)}
          </div>
        ) : backtest.error ? (
          <div className="card" style={{ padding: 20 }}><p style={{ color: 'var(--text-2)', fontSize: 13 }}>{backtest.error}</p></div>
        ) : d ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'repeat(3,1fr)', gap: isMobile ? 10 : 16 }}>
              {STRATEGIES.map(s => {
                const stats  = d[s.key]
                const isBest = Math.abs((stats?.sharpe_ratio || 0) - bestSharpe) < 0.001
                return (
                  <div key={s.key} className={`stat-card ${s.cls}`} style={{ position: 'relative' }}>
                    {isBest && <span style={{ position: 'absolute', top: 12, right: 14, fontSize: 9, color: s.text, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>BEST ★</span>}
                    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', marginBottom: 10, color: 'var(--text-1)', display: 'flex', alignItems: 'center', gap: 7 }}>
                      <span style={{ width: 7, height: 7, borderRadius: '50%', background: s.color, display: 'inline-block', boxShadow: `0 0 6px ${s.color}80` }} />{s.label}
                    </div>
                    <div style={{ fontSize: 26, fontWeight: 700, fontFamily: 'var(--font-mono)', color: s.text, letterSpacing: '-0.02em', marginBottom: 3 }}>{stats ? fmt(stats.sharpe_ratio, 3) : '—'}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-2)', marginBottom: 12 }}>Sharpe ratio</div>
                    {stats && [
                      ['Total return',  pct(stats.total_return)],
                      ['Annual return', pct(stats.ann_return)],
                      ['Worst drawdown',pct(stats.max_drawdown)],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{k}</span>
                        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: k === 'Worst drawdown' ? '#f87171' : 'var(--text-1)', fontWeight: 500 }}>{v}</span>
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>

            {!isMobile && (
              <div className="card" style={{ padding: 22, overflowX: 'auto' }}>
                <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', marginBottom: 14 }}>
                  Period: {d.test_period_start} to {d.test_period_end} · Tx cost: {(d.transaction_cost * 100).toFixed(2)}% · Rf: {(d.risk_free_rate * 100).toFixed(0)}%
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 500 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', padding: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Strategy</td>
                      {COLS_FULL.map(c => <td key={c.key} style={{ textAlign: 'right', fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', padding: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{c.label}</td>)}
                    </tr>
                  </thead>
                  <tbody>
                    {STRATEGIES.map(s => {
                      const stats = d[s.key]; if (!stats) return null
                      const isBest = Math.abs((stats.sharpe_ratio || 0) - bestSharpe) < 0.001
                      return (
                        <tr key={s.key} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: isBest ? `${s.color}08` : 'transparent' }}>
                          <td style={{ padding: '10px 0', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
                              <span style={{ width: 7, height: 7, borderRadius: '50%', background: s.color, display: 'inline-block' }} />
                              <span style={{ color: s.text }}>{s.label}{isBest ? ' ★' : ''}</span>
                            </span>
                          </td>
                          {COLS_FULL.map(c => <td key={c.key} style={{ textAlign: 'right', fontSize: 12, fontFamily: 'var(--font-mono)', padding: '10px 0', color: c.key === 'max_drawdown' ? '#f87171' : c.key === 'sharpe_ratio' && isBest ? s.text : 'var(--text-1)' }}>{c.fmt(stats[c.key])}</td>)}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 10 }}>★ Best risk-adjusted return</div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  )
}
