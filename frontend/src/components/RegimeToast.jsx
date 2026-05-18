import { useEffect, useState } from 'react'
import { REGIME_COLOR, REGIME_TEXT } from '../utils'

export default function RegimeToast({ alert, onDismiss }) {
  const [visible, setVisible] = useState(false)
  const [leaving, setLeaving] = useState(false)

  useEffect(() => {
    if (!alert) return
    setVisible(true); setLeaving(false)
    const t = setTimeout(() => {
      setLeaving(true)
      setTimeout(() => { setVisible(false); onDismiss?.() }, 320)
    }, 5000)
    return () => clearTimeout(t)
  }, [alert?.ts])

  if (!visible || !alert) return null
  const color = REGIME_COLOR[alert.to] || '#f59e0b'
  const text  = REGIME_TEXT[alert.to]  || '#fbbf24'

  return (
    <div className={leaving ? 'toast-out' : 'toast-in'} style={{ position: 'fixed', top: 20, right: 20, zIndex: 100 }}>
      <div className="card" style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 12, minWidth: 260, boxShadow: `0 0 24px ${color}30`, borderColor: `${color}40` }}>
        <span style={{ fontSize: 18 }}>⚡</span>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Regime Change</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: text, fontFamily: 'JetBrains Mono, monospace' }}>{alert.from} → {alert.to}</div>
        </div>
        <button onClick={() => { setLeaving(true); setTimeout(() => { setVisible(false); onDismiss?.() }, 320) }}
          style={{ marginLeft: 'auto', color: 'var(--text-3)', background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, lineHeight: 1 }}>×</button>
      </div>
    </div>
  )
}
