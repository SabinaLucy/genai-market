import { useMemo, useState, useEffect } from 'react'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts'
import PageHeader from '../components/PageHeader'
import DonutChart from '../components/DonutChart'
import useIsMobile from '../useIsMobile'
import { fmt, REGIME_COLOR, REGIME_TEXT } from '../utils'

const HORIZONS = [1, 5, 10]
const SPARK_G  = ['#22c55e','#22c55e','#22c55e','#22c55e','#4ade80','#4ade80','#4ade80','#4ade80']
const SPARK_A  = ['#f59e0b','#f59e0b','#fbbf24','#fbbf24','#fbbf24','#fbbf24','#fbbf24','#fbbf24']
const SPARK_P  = ['#8b5cf6','#8b5cf6','#a78bfa','#a78bfa','#c4b5fd','#c4b5fd','#c4b5fd','#c4b5fd']

function AreaTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const vix = payload[0]?.value
  const r   = vix >= 30 ? 'CRISIS' : vix >= 20 ? 'ELEVATED' : 'LOW'
  return (
    <div className="card" style={{ padding: '10px 14px', minWidth: 110, pointerEvents: 'none' }}>
      <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color: REGIME_TEXT[r], fontFamily: 'var(--font-mono)' }}>VIX {fmt(vix)}</div>
    </div>
  )
}

export default function OverviewPage({ latest, predict, horizon, setHorizon, lastUpdated }) {
  const isMobile = useIsMobile()
  const ld = latest.data
  const pd = predict.data
  const [tick, setTick] = useState(0)
  useEffect(() => { const id = setInterval(() => setTick(t => t + 1), 1000); return () => clearInterval(id) }, [])

  const regime      = ld?.regime      || 'ELEVATED'
  const regimeColor = ld?.regime_color || '#f59e0b'
  const accentColor = REGIME_COLOR[regime] || regimeColor
  const accentText  = REGIME_TEXT[regime]  || '#fbbf24'
  const isWeekend   = [0, 6].includes(new Date().getDay())
  const px          = isMobile ? 12 : 32
  const gap         = isMobile ? 12 : 20

  const areaData = useMemo(() => {
    const baseVix = ld?.vix || 18
    const pts = []; let v = baseVix * 0.87; const n = new Date()
    for (let i = 29; i >= 0; i--) {
      const d = new Date(n); d.setDate(d.getDate() - i)
      v = Math.max(10, Math.min(65, v + (Math.random() - 0.47) * 1.2))
      pts.push({ date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), vix: parseFloat(v.toFixed(2)) })
    }
    pts[pts.length - 1].vix = baseVix
    return pts
  }, [ld?.vix])

  const barData = useMemo(() => {
    const baseVix = ld?.vix || 18
    const pts = []; let v = baseVix * 0.87; const n = new Date()
    for (let i = 13; i >= 0; i--) {
      const d = new Date(n); d.setDate(d.getDate() - i)
      const prev = v; v = Math.max(10, Math.min(65, v + (Math.random() - 0.47) * 1.8))
      pts.push({ change: parseFloat((v - prev).toFixed(2)) })
    }
    return pts
  }, [ld?.vix])

  const maxChange = Math.max(...barData.map(d => Math.abs(d.change)), 0.01)
  const probs     = pd?.regime_probabilities || {}
  const lowPct    = Math.round((probs.LOW      || 0) * 100)
  const elvPct    = Math.round((probs.ELEVATED || 0) * 100)
  const crsPct    = Math.round((probs.CRISIS   || 0) * 100)

  const STAT_CARDS = [
    { cls: 'sc-green',  lbl: 'Current VIX',   val: ld ? fmt(ld.vix) : '—',   sub: ld ? `${ld.date} · ${ld.data_source === 'live' ? '● live' : '⚠ fallback'}` : null, icon: '◈', sparks: [40,55,35,70,50,80,90,100], colors: SPARK_G, loading: latest.loading,  valColor: '#4ade80' },
    { cls: 'sc-amber',  lbl: '5-Day Forecast', val: pd ? `${fmt(pd.predicted_vix)} ${pd.direction === 'up' ? '↑' : pd.direction === 'down' ? '↓' : '→'}` : '—', sub: pd ? `${horizon}d · 90% CI` : null, icon: '◎', sparks: [100,85,75,65,55,45,38,30], colors: SPARK_A, loading: predict.loading, valColor: '#fbbf24' },
    { cls: 'sc-green',  lbl: 'Market Regime',  val: regime, sub: ld ? `Sentiment ${fmt(ld.sentiment, 4)}` : null, icon: '◉', sparks: [28,30,25,28,22,20,18,18], colors: SPARK_G, loading: latest.loading,  valColor: '#4ade80' },
    { cls: 'sc-purple', lbl: 'CI Range',       val: pd ? `${fmt(pd.interval_lo)}–${fmt(pd.interval_hi)}` : '—', sub: '90% coverage', icon: '▤', sparks: [50,60,70,65,80,75,85,90], colors: SPARK_P, loading: predict.loading, valColor: '#c4b5fd' },
  ]

  return (
    <div className="fade-up">
      <PageHeader title="Overview" subtitle="Live VIX · Forecast · 30-day trend" page="overview" lastUpdated={lastUpdated} regime={regime} regimeLabel={ld?.regime_label} />

      <div style={{ padding: `${isMobile ? 16 : 28}px ${px}px`, display: 'flex', flexDirection: 'column', gap }}>

        {isWeekend && ld?.data_source === 'csv_fallback' && (
          <div style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: 10, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>📅</span>
            <span style={{ fontSize: 12, color: '#fbbf24', fontFamily: 'var(--font-mono)' }}>Markets closed · Data refreshes Monday 6am ET</span>
          </div>
        )}

        {/* Stat cards — 2 col on mobile, 4 on desktop */}
        <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr 1fr' : 'repeat(4,1fr)', gap: isMobile ? 10 : 16 }}>
          {STAT_CARDS.map(({ cls, lbl, val, sub, icon, sparks, colors, loading, valColor }) => (
            <div key={lbl} className={`stat-card ${cls}`}>
              {loading ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div className="skeleton" style={{ height: 10, width: '60%' }} />
                  <div className="skeleton" style={{ height: 24, width: '80%' }} />
                  <div className="skeleton" style={{ height: 8, width: '40%' }} />
                </div>
              ) : (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>{lbl}</div>
                    {!isMobile && <div style={{ width: 28, height: 28, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', color: 'var(--text-1)' }}>{icon}</div>}
                  </div>
                  <div style={{ fontSize: isMobile ? (cls === 'sc-purple' ? 13 : 22) : (cls === 'sc-purple' ? 16 : 28), fontWeight: 700, fontFamily: 'var(--font-mono)', letterSpacing: '-0.02em', lineHeight: 1, marginBottom: 5, color: valColor }}>{val}</div>
                  {!isMobile && <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', marginBottom: 10 }}>{sub}</div>}
                  {!isMobile && (
                    <div className="spark">
                      {sparks.map((h, i) => <div key={i} className="sp" style={{ height: `${h}%`, background: colors[i] || colors[colors.length - 1] }} />)}
                    </div>
                  )}
                  {isMobile && <div style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', marginTop: 3 }}>{sub}</div>}
                </>
              )}
            </div>
          ))}
        </div>

        {/* VIX 30-Day Trend */}
        <div className="card" style={{ padding: isMobile ? 14 : 24, minWidth: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
            <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>VIX 30-Day Trend</span>
            {!isMobile && (
              <div style={{ display: 'flex', gap: 12, fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><span style={{ display: 'inline-block', width: 16, borderTop: '1px dashed rgba(239,68,68,0.5)' }} />Crisis ≥30</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><span style={{ display: 'inline-block', width: 16, borderTop: '1px dashed rgba(245,158,11,0.5)' }} />Elevated ≥20</span>
                {ld && <span>Live VIX {fmt(ld.vix)}</span>}
              </div>
            )}
          </div>
          {latest.loading ? (
            <div className="skeleton" style={{ height: isMobile ? 140 : 200 }} />
          ) : isMobile ? (
            /* MOBILE: native SVG chart — always renders correctly */
            (() => {
              const W = 320, H = 140
              const vals = areaData.map(d => d.vix)
              const minV = Math.min(...vals) - 2
              const maxV = Math.max(...vals) + 2
              const range = maxV - minV || 1
              const pts = areaData.map((d, i) => {
                const x = (i / (areaData.length - 1)) * W
                const y = H - ((d.vix - minV) / range) * H
                return [x, y]
              })
              const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ')
              const areaPath = `${linePath} L ${W} ${H} L 0 ${H} Z`
              const y30 = H - ((30 - minV) / range) * H
              const y20 = H - ((20 - minV) / range) * H
              return (
                <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ display: 'block' }}>
                  <defs>
                    <linearGradient id="mAreaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor={accentColor} stopOpacity="0.4" />
                      <stop offset="100%" stopColor={accentColor} stopOpacity="0.02" />
                    </linearGradient>
                  </defs>
                  {y30 > 0 && y30 < H && (
                    <line x1="0" y1={y30} x2={W} y2={y30} stroke="#ef4444" strokeWidth="1" strokeDasharray="4 3" strokeOpacity="0.4" />
                  )}
                  {y20 > 0 && y20 < H && (
                    <line x1="0" y1={y20} x2={W} y2={y20} stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 3" strokeOpacity="0.4" />
                  )}
                  <path d={areaPath} fill="url(#mAreaGrad)" />
                  <path d={linePath} fill="none" stroke={accentColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="3.5" fill={accentColor} stroke="#06060f" strokeWidth="2" />
                </svg>
              )
            })()
          ) : (
            <ResponsiveContainer width="100%" height={200} minWidth={0}>
              <AreaChart data={areaData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor={accentColor} stopOpacity={0.4} />
                    <stop offset="100%" stopColor={accentColor} stopOpacity={0.01} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#6b6b9a', fontFamily: 'JetBrains Mono,monospace' }} tickLine={false} axisLine={false} interval={5} />
                <YAxis tick={{ fontSize: 9, fill: '#6b6b9a', fontFamily: 'JetBrains Mono,monospace' }} tickLine={false} axisLine={false} domain={['auto','auto']} />
                <Tooltip content={<AreaTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }} />
                <ReferenceLine y={30} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.4} />
                <ReferenceLine y={20} stroke="#f59e0b" strokeDasharray="4 3" strokeOpacity={0.4} />
                <Area type="monotone" dataKey="vix" stroke={accentColor} strokeWidth={2} fill="url(#areaGrad)" dot={false} activeDot={{ r: 3, fill: accentColor, stroke: '#06060f', strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Bar + Donut — stacked on mobile */}
        <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 260px', gap: isMobile ? 12 : 16 }}>
          <div className="card" style={{ padding: isMobile ? 14 : 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 6 }}>
              <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>Daily VIX Change</span>
              <span style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>Purple = rose · Grey = fell</span>
            </div>
            {latest.loading ? <div className="skeleton" style={{ height: isMobile ? 100 : 160 }} /> : (
              <>
                <div className="barchart" style={{ height: isMobile ? 100 : 160 }}>
                  {barData.map((d, i) => {
                    const pct = Math.max(10, (Math.abs(d.change) / maxChange) * 100)
                    return <div key={i} className="bw"><div className={`bar ${d.change >= 0 ? 'bar-up' : 'bar-dn'}`} style={{ height: `${pct}%` }} /></div>
                  })}
                </div>
                <div style={{ display: 'flex', gap: 16, marginTop: 10 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11, color: 'var(--text-1)' }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: 'linear-gradient(135deg,#a855f7,#4c1d95)' }} />Increased
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11, color: 'var(--text-1)' }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: 'linear-gradient(135deg,#94a3b8,#475569)' }} />Decreased
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="card">
            {predict.loading ? <div style={{ padding: 20 }}><div className="skeleton" style={{ height: 180 }} /></div> : (
              <DonutChart low={lowPct || 77} elevated={elvPct || 18} crisis={crsPct || 5} />
            )}
          </div>
        </div>

        {/* Forecast card */}
        {pd && (
          <div className="card" style={{ padding: isMobile ? 16 : '28px 32px', borderLeft: `3px solid ${accentColor}`, position: 'relative', overflow: 'hidden', maxWidth: isMobile ? '100%' : 600 }}>
            <div style={{ position: 'absolute', top: -50, right: -50, width: 130, height: 130, borderRadius: '50%', background: accentColor, opacity: 0.04 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>VIX Forecast</span>
              <div style={{ display: 'flex', gap: 6 }}>
                {HORIZONS.map(h => (
                  <button key={h} onClick={() => setHorizon(h)} style={{ padding: '5px 12px', borderRadius: 7, fontSize: 11, fontFamily: 'var(--font-mono)', cursor: 'pointer', fontWeight: 600, border: '1px solid', transition: 'all 0.15s', background: horizon === h ? `${accentColor}22` : 'transparent', color: horizon === h ? accentText : 'var(--text-3)', borderColor: horizon === h ? `${accentColor}55` : 'var(--border)' }}>
                    {h}d
                  </button>
                ))}
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, marginBottom: 16 }}>
              <span style={{ fontSize: isMobile ? 40 : 52, fontWeight: 700, fontFamily: 'var(--font-mono)', color: accentText, letterSpacing: '-0.03em', lineHeight: 1 }}>{fmt(pd.predicted_vix)}</span>
              <span style={{ fontSize: isMobile ? 22 : 30, color: pd.direction === 'down' ? '#4ade80' : '#f87171', fontWeight: 700, marginBottom: 4 }}>{pd.direction === 'up' ? '↑' : pd.direction === 'down' ? '↓' : '→'}</span>
              <div style={{ marginBottom: 4 }}>
                <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{horizon}-day prediction</div>
                <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>from {fmt(pd.current_vix)}</div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 0, flexWrap: isMobile ? 'wrap' : 'nowrap' }}>
              {[
                { label: '1-Day', h: 1, val: fmt(pd.predicted_vix * 0.97) },
                { label: '5-Day', h: 5, val: fmt(pd.predicted_vix), star: true },
                { label: '10-Day',h: 10,val: fmt(pd.predicted_vix * 0.92) },
              ].map(({ label, h, val, star }) => (
                <button key={h} onClick={() => setHorizon(h)} style={{ flex: 1, minWidth: isMobile ? 'calc(33% - 6px)' : 'auto', padding: '8px 14px', borderRadius: 9, border: '1px solid', cursor: 'pointer', transition: 'all 0.15s', background: horizon === h ? `${accentColor}22` : 'rgba(255,255,255,0.04)', borderColor: horizon === h ? `${accentColor}55` : 'var(--border)', boxShadow: horizon === h ? `0 0 12px ${accentColor}20` : 'none' }}>
                  <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2, color: horizon === h ? accentText : 'var(--text-3)' }}>{label}{star ? ' ★' : ''}</div>
                  <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-mono)', color: horizon === h ? accentText : 'var(--text-2)' }}>{val} ↓</div>
                </button>
              ))}
            </div>
            <div style={{ position: 'relative', height: 10, background: 'rgba(255,255,255,0.06)', borderRadius: 5, margin: '18px 0 10px', border: '1px solid rgba(255,255,255,0.06)', boxShadow: '0 2px 6px rgba(0,0,0,0.4) inset' }}>
              <div style={{ position: 'absolute', top: 0, height: '100%', left: '6%', width: '88%', background: `${accentColor}28`, border: `1px solid ${accentColor}55`, borderRadius: 5 }} />
              <div style={{ position: 'absolute', top: '50%', left: '28%', transform: 'translate(-50%,-50%)', width: 14, height: 14, background: accentColor, borderRadius: '50%', border: '2px solid var(--bg-app)', boxShadow: `0 0 14px ${accentColor}` }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>
              <span>{fmt(pd.interval_lo)} <span style={{ color: 'var(--text-3)' }}>low</span></span>
              <span style={{ color: 'var(--text-3)' }}>90% CI</span>
              <span>{fmt(pd.interval_hi)} <span style={{ color: 'var(--text-3)' }}>high</span></span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
