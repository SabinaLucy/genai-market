import PageHeader  from '../components/PageHeader'
import RegimeBadge from '../components/RegimeBadge'
import useIsMobile from '../useIsMobile'
import { fmt, REGIME_COLOR, REGIME_TEXT } from '../utils'
import React, { useState } from 'react'
import ReactDOM from 'react-dom'

function InfoTip({ text, title }) {
  const [open, setOpen] = useState(false)
  const [pos,  setPos]  = useState({ top: 0, left: 0 })
  const btnRef          = React.useRef(null)

  const handleClick = () => {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - rect.bottom
      setPos({
        top:  spaceBelow > 200 ? rect.bottom + 10 : rect.top - 220,
        left: Math.max(12, Math.min(rect.left - 100, window.innerWidth - 280)),
      })
    }
    setOpen(o => !o)
  }

  return (
    <>
      <button ref={btnRef} onClick={handleClick} style={{ width: 18, height: 18, borderRadius: 6, background: open ? 'rgba(139,92,246,0.3)' : 'rgba(139,92,246,0.12)', border: `1px solid ${open ? 'rgba(139,92,246,0.7)' : 'rgba(139,92,246,0.4)'}`, color: '#a78bfa', fontSize: 10, fontWeight: 700, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0, lineHeight: 1, padding: 0, marginLeft: 6, boxShadow: open ? '0 0 10px rgba(139,92,246,0.4)' : 'none', transition: 'all 0.2s', fontFamily: 'var(--font-mono)' }}>i</button>
      {open && ReactDOM.createPortal(
        <>
          <div onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 9998 }} />
          <div style={{ position: 'fixed', top: pos.top, left: pos.left, width: 260, background: 'linear-gradient(135deg,#1c1c3a 0%,#12122a 100%)', border: '1px solid rgba(139,92,246,0.45)', borderRadius: 14, overflow: 'hidden', boxShadow: '0 20px 60px rgba(0,0,0,0.9)', zIndex: 9999 }}>
            <div style={{ height: 3, background: 'linear-gradient(90deg,#7c3aed,#06b6d4)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ width: 24, height: 24, borderRadius: 7, background: 'rgba(139,92,246,0.15)', border: '1px solid rgba(139,92,246,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: '#a78bfa', fontWeight: 700, flexShrink: 0 }}>i</div>
              <div style={{ fontSize: 10, fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#a78bfa', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{title || 'What is this?'}</div>
            </div>
            <div style={{ padding: '10px 14px 14px', fontSize: 12, lineHeight: 1.7, color: '#c8c8f0' }}>{text}</div>
          </div>
        </>,
        document.body
      )}
    </>
  )
}

const FEATURE_NAMES = {
  vix_lag1:        "Yesterday's VIX Level",
  vix_lag5:        '5-Day Lagged VIX',
  vix_lag21:       '21-Day Lagged VIX',
  vix_roll_mean5:  '5-Day Rolling VIX Average',
  vix_roll_std21:  '21-Day Volatility Spread',
  fedfunds:        'Federal Funds Rate',
  cpi:             'Inflation Rate (CPI)',
  unrate:          'Unemployment Rate',
  gs10:            '10-Year Treasury Yield',
  indpro:          'Industrial Production Index',
  m2sl:            'Money Supply (M2)',
  sentiment:       'News Sentiment Score',
}

const RC = [
  { cls:'rc-low',    ico:'◈', name:'Stable Market',   big:'LOW',      bColor:'#22c55e', tColor:'#4ade80', lbl:'[STABLE]', regime:'LOW',      probs:[77,18,5]  },
  { cls:'rc-elev',   ico:'◉', name:'Elevated Stress', big:'ELEVATED', bColor:'#f59e0b', tColor:'#fbbf24', lbl:'[WATCH]',  regime:'ELEVATED', probs:[8,88,4]   },
  { cls:'rc-crisis', ico:'⚡', name:'Crisis Alert',    big:'CRISIS',   bColor:'#ef4444', tColor:'#f87171', lbl:'[ALERT]',  regime:'CRISIS',   probs:[2,15,83]  },
]
const PROB_LABELS = ['Stable','Elevated','Crisis']
const PROB_COLORS = ['#22c55e','#f59e0b','#ef4444']

export default function RegimePage({ latest, analogues, shap }) {
  const isMobile = useIsMobile()
  const ld       = latest.data
  const regime   = ld?.regime || 'ELEVATED'
  const color    = REGIME_COLOR[regime] || '#f59e0b'
  const text     = REGIME_TEXT[regime]  || '#fbbf24'
  const px       = isMobile ? 12 : 32
  const drivers  = shap.data?.top_drivers || []
  const maxVal   = Math.max(...drivers.map(d => Math.abs(d.shap_value ?? d.importance ?? 0)), 0.001)

  return (
    <div className="fade-up">
      <PageHeader title="Regime Analysis" subtitle="Classification · Historical analogues · Feature drivers" page="regime" regime={regime} regimeLabel={ld?.regime_label} />
      <div style={{ padding: `${isMobile ? 14 : 28}px ${px}px`, display: 'flex', flexDirection: 'column', gap: isMobile ? 12 : 20 }}>

        {ld && (
          <div className="card" style={{ padding: isMobile ? '16px' : '24px 28px', borderLeft: `3px solid ${color}`, boxShadow: `0 0 32px ${color}18, var(--shadow-card)`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4, display: 'flex', alignItems: 'center' }}>
                 Current Regime
                <InfoTip title="What is Regime?" text="Volarix puts the market into one of three states. STABLE means VIX is below 20 and investors are calm. WATCH means VIX is between 20 and 30 and stress is building. ALERT means VIX is above 30 and the market is in full crisis mode." />
              </div>
              <div style={{ fontSize: isMobile ? 32 : 42, fontWeight: 700, color: text, fontFamily: 'var(--font-mono)', letterSpacing: '-0.02em', lineHeight: 1 }}>{regime}</div>
              <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>VIX {fmt(ld.vix)} · {ld.date}</div>
            </div>
            <RegimeBadge regime={regime} regimeLabel={ld.regime_label} size="lg" />
          </div>
        )}

        {/* Regime cards — 1 col on mobile, 3 on desktop */}
        <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'repeat(3,1fr)', gap: isMobile ? 10 : 16 }}>
          {RC.map(({ cls, ico, name, big, bColor, tColor, lbl, regime: r, probs }) => (
            <div key={big} className={`regime-card ${cls}`}>
              <div style={{ width: 36, height: 36, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, marginBottom: 12, background: `${bColor}18`, border: `1px solid ${bColor}50`, color: tColor, boxShadow: `0 0 14px ${bColor}30` }}>{ico}</div>
              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', marginBottom: 3, color: tColor }}>{name}</div>
              <div style={{ fontSize: 24, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', marginBottom: 5 }}>{big}</div>
              <RegimeBadge regime={r} regimeLabel={lbl} size="sm" />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 12, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.07)' }}>
                {probs.map((p, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 11, color: 'var(--text-1)', width: 64 }}>{PROB_LABELS[i]}</span>
                    <div className="prob-track"><div style={{ width: `${p}%`, height: '100%', borderRadius: 2, background: PROB_COLORS[i] }} /></div>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', width: 28, textAlign: 'right', fontWeight: 600 }}>{p}%</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Analogues + SHAP — stacked on mobile */}
        <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr', gap: isMobile ? 12 : 16 }}>
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-2)', marginBottom: 10, display: 'flex', alignItems: 'center' }}>
                 Similar Historical Periods
              <InfoTip title="What are Analogues?" text="Volarix searches through 25 years of market history to find periods that looked most similar to right now. The match percentage shows how closely that past period resembles current conditions. History does not repeat exactly but it often rhymes." />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {analogues.loading ? [1,2].map(i => <div key={i} className="card" style={{ padding: 16, height: 100 }}><div className="skeleton" style={{ height: '100%' }} /></div>)
              : analogues.error ? <div className="card" style={{ padding: 16 }}><p style={{ color: 'var(--text-2)', fontSize: 13 }}>{analogues.error}</p></div>
              : (analogues.data?.analogues || []).slice(0, 2).map((a, i) => {
                const ac = REGIME_COLOR[a.regime] || '#f59e0b'
                const at = REGIME_TEXT[a.regime]  || '#fbbf24'
                return (
                  <div key={i} style={{ borderRadius: 12, padding: '16px 18px', border: `1px solid ${ac}35`, background: `${ac}08`, position: 'relative', overflow: 'hidden', boxShadow: 'var(--shadow-card)' }}>
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 1, background: `linear-gradient(90deg,transparent,${ac}80,transparent)` }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-mono)', color: at }}>{a.date || `Period ${i+1}`}</div>
                      <RegimeBadge regime={a.regime} regimeLabel={a.regime_label} size="sm" />
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      {a.vix != null && <div><div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>VIX then</div><div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: 600 }}>{fmt(a.vix)}</div></div>}
                      {a.similarity != null && <div><div style={{ fontSize: 9, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Match</div><div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: 600 }}>{(a.similarity * 100).toFixed(1)}%</div></div>}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', color: 'var(--text-2)', marginBottom: 10, display: 'flex', alignItems: 'center' }}>
                 What Is Driving Volatility Right Now
              <InfoTip title="What is SHAP?" text="SHAP shows which factors are pushing volatility up or down right now. The longer the bar, the bigger the influence. Red arrows mean that factor is increasing stress. Purple arrows mean it is reducing it. This is how the model explains its own thinking." />
            </div>
            <div className="card" style={{ padding: isMobile ? 14 : '22px 24px' }}>
              <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 16 }}>Model explanation: how much each factor influences the VIX forecast</div>
              {shap.loading ? [1,2,3,4,5].map(i => <div key={i} className="skeleton" style={{ height: 36, marginBottom: 12 }} />)
              : shap.error ? <p style={{ color: 'var(--text-2)', fontSize: 13 }}>{shap.error}</p>
              : drivers.slice(0, 5).map((d, i) => {
                const raw  = d.shap_value ?? d.importance ?? 0
                const val  = Math.abs(raw); const pos = raw >= 0
                const frac = val / maxVal
                const humanName = FEATURE_NAMES[d.feature] || d.feature
                return (
                  <div key={i} style={{ marginBottom: i < 4 ? 14 : 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 5 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: pos ? '#f87171' : '#c4b5fd', fontWeight: 700, fontSize: 16, lineHeight: 1, flexShrink: 0 }}>{pos ? '↑' : '↓'}</span>
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>{humanName}</div>
                          <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 1 }}>{d.feature}</div>
                        </div>
                      </div>
                      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: 600, flexShrink: 0, marginLeft: 12 }}>{fmt(val, 4)}</span>
                    </div>
                    <div style={{ height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${frac * 100}%`, height: '100%', borderRadius: 3, background: pos ? 'linear-gradient(90deg,#dc2626,#f87171)' : 'linear-gradient(90deg,#6d28d9,#c4b5fd)' }} />
                    </div>
                  </div>
                )
              })}
              {shap.data?.stability_rho != null && (
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 12, marginTop: 12, borderTop: '1px solid var(--border)' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Model stability score</span>
                  <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: 700 }}>{fmt(shap.data.stability_rho, 3)}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
