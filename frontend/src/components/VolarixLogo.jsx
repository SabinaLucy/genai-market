export default function VolarixLogo({ regimeColor, collapsed = false }) {
  const accent = regimeColor || '#f07a22'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <svg width="36" height="36" viewBox="210 72 260 370" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ flexShrink: 0 }}>
        <defs>
          <linearGradient id="lg-shield" x1="340" y1="72" x2="340" y2="428" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#1b6b82"/>
            <stop offset="100%" stopColor="#0c2f4a"/>
          </linearGradient>
          <linearGradient id="lg-teal" x1="304" y1="405" x2="304" y2="108" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#1d8fa8"/>
            <stop offset="100%" stopColor="#5ee0f5"/>
          </linearGradient>
          <linearGradient id="lg-grey" x1="355" y1="390" x2="430" y2="148" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#6b8fa8"/>
            <stop offset="100%" stopColor="#c8dde8"/>
          </linearGradient>
          <linearGradient id="lg-wave" x1="235" y1="310" x2="435" y2="200" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#5dd8ee"/>
            <stop offset="100%" stopColor="#a8eef8"/>
          </linearGradient>
        </defs>

        <path
          d="M340 72 L210 118 L210 240 C210 330 270 400 340 428 C410 400 470 330 470 240 L470 118 Z"
          fill="url(#lg-shield)" stroke="#2a9ab8" strokeWidth="1.5" strokeLinejoin="round"
        />
        <path
          d="M340 88 L222 128 L222 240 C222 323 277 388 340 412 C403 388 458 323 458 240 L458 128 Z"
          fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="1"
        />

        <polyline
          points="235,310 268,268 295,292 322,240 350,270 378,220 405,248 435,200"
          stroke="url(#lg-wave)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none"
        />
        <circle cx="235" cy="310" r="4.5" fill="#5dd8ee"/>
        <circle cx="295" cy="292" r="4.5" fill="#5dd8ee"/>
        <circle cx="350" cy="270" r="4.5" fill="#5dd8ee"/>
        <circle cx="405" cy="248" r="4.5" fill="#5dd8ee"/>

        <line x1="304" y1="405" x2="304" y2="158" stroke="url(#lg-teal)" strokeWidth="14" strokeLinecap="round"/>
        <polygon points="304,108 282,158 326,158" fill="#4dd8f0"/>

        <line x1="355" y1="390" x2="428" y2="162" stroke="url(#lg-grey)" strokeWidth="11" strokeLinecap="round"/>
        <polygon points="446,122 418,160 458,168" fill="#c0d8e8"/>

        <line x1="385" y1="385" x2="452" y2="192" stroke={accent} strokeWidth="8" strokeLinecap="round" opacity="0.9"/>
        <polygon points="465,158 442,194 470,204" fill={accent} opacity="0.9"/>
      </svg>

      {!collapsed && (
        <span style={{
          fontFamily:    "'JetBrains Mono', monospace",
          fontWeight:    700,
          fontSize:      15,
          letterSpacing: '0.1em',
          color:         'var(--text-1)',
          userSelect:    'none',
        }}>
          VOLARIX
        </span>
      )}
    </div>
  )
}
