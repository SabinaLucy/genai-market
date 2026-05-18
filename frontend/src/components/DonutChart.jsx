export default function DonutChart({ low = 0, elevated = 0, crisis = 0 }) {
  const r = 58
  const circ = 2 * Math.PI * r
  const lowPct  = low / 100
  const elvPct  = elevated / 100
  const crsPct  = crisis / 100
  const lowDash  = circ * lowPct
  const elvDash  = circ * elvPct
  const crsDash  = circ * crsPct
  const elvOffset = -(crsDash)
  const lowOffset = -(crsDash + elvDash)
  const dominant = low >= elevated && low >= crisis ? { val: low, label: 'Stable', color: '#4ade80' }
    : elevated >= crisis ? { val: elevated, label: 'Elevated', color: '#fbbf24' }
    : { val: crisis, label: 'Crisis', color: '#f87171' }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '22px 20px' }}>
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-2)', marginBottom: 18, alignSelf: 'flex-start' }}>
        Regime Probability
      </div>
      <div style={{ position: 'relative', width: 150, height: 150, marginBottom: 18 }}>
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle cx="75" cy="75" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="20"/>
          {crsPct > 0 && (
            <circle cx="75" cy="75" r={r} fill="none" stroke="#ef4444" strokeWidth="20"
              strokeDasharray={`${crsDash} ${circ - crsDash}`}
              strokeDashoffset="0"
              strokeLinecap="round"
              transform="rotate(-90 75 75)" opacity="0.85"/>
          )}
          {elvPct > 0 && (
            <circle cx="75" cy="75" r={r} fill="none" stroke="#f59e0b" strokeWidth="20"
              strokeDasharray={`${elvDash} ${circ - elvDash}`}
              strokeDashoffset={elvOffset}
              strokeLinecap="round"
              transform="rotate(-90 75 75)" opacity="0.9"/>
          )}
          {lowPct > 0 && (
            <circle cx="75" cy="75" r={r} fill="none" stroke="#22c55e" strokeWidth="22"
              strokeDasharray={`${lowDash} ${circ - lowDash}`}
              strokeDashoffset={lowOffset}
              strokeLinecap="round"
              transform="rotate(-90 75 75)"/>
          )}
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: dominant.color, lineHeight: 1 }}>
            {dominant.val}%
          </div>
          <div style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'JetBrains Mono, monospace', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            {dominant.label}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%' }}>
        {[
          { label: 'Stable', val: low,      color: '#22c55e', text: '#4ade80'  },
          { label: 'Elevated', val: elevated, color: '#f59e0b', text: '#fbbf24' },
          { label: 'Crisis',   val: crisis,   color: '#ef4444', text: '#f87171' },
        ].map(({ label, val, color, text }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-1)' }}>
              <div style={{ width: 9, height: 9, borderRadius: '50%', background: color, flexShrink: 0, boxShadow: `0 0 6px ${color}80` }} />
              {label}
            </div>
            <div style={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace', color: text, fontWeight: 600 }}>
              {val}%
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
