import { useEffect, useState } from 'react'

export default function SplashScreen({ onDone }) {
  const [fading, setFading] = useState(false)

  useEffect(() => {
    const fadeTimer = setTimeout(() => setFading(true), 3500)
    const doneTimer = setTimeout(() => onDone(), 4000)
    return () => { clearTimeout(fadeTimer); clearTimeout(doneTimer) }
  }, [onDone])

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 9999,
      background: 'radial-gradient(ellipse at center, #0e0e1c 0%, #06060f 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 24,
      opacity: fading ? 0 : 1,
      transition: 'opacity 0.5s ease-out',
    }}>
      <style>{`
        @keyframes splashPulse {
          0%, 100% { transform: scale(1); filter: drop-shadow(0 0 20px rgba(34,211,238,0.4)); }
          50%      { transform: scale(1.05); filter: drop-shadow(0 0 40px rgba(34,211,238,0.7)); }
        }
        @keyframes splashFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes splashLine {
          0%   { width: 0; opacity: 0; }
          50%  { width: 100px; opacity: 1; }
          100% { width: 100px; opacity: 0.4; }
        }
      `}</style>

      <div style={{
        animation: 'splashPulse 2s ease-in-out infinite',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <svg width="84" height="84" viewBox="210 72 260 370" fill="none">
          <defs>
            <linearGradient id="sp-s" x1="340" y1="72" x2="340" y2="428" gradientUnits="userSpaceOnUse">
              <stop offset="0%"   stopColor="#1b6b82" />
              <stop offset="100%" stopColor="#0c2f4a" />
            </linearGradient>
            <linearGradient id="sp-t" x1="304" y1="405" x2="304" y2="108" gradientUnits="userSpaceOnUse">
              <stop offset="0%"   stopColor="#1d8fa8" />
              <stop offset="100%" stopColor="#5ee0f5" />
            </linearGradient>
            <linearGradient id="sp-w" x1="235" y1="310" x2="435" y2="200" gradientUnits="userSpaceOnUse">
              <stop offset="0%"   stopColor="#5dd8ee" />
              <stop offset="100%" stopColor="#a8eef8" />
            </linearGradient>
          </defs>
          <path d="M340 72 L210 118 L210 240 C210 330 270 400 340 428 C410 400 470 330 470 240 L470 118 Z" fill="url(#sp-s)" stroke="#2a9ab8" strokeWidth="2" strokeLinejoin="round" />
          <polyline points="235,310 268,268 295,292 322,240 350,270 378,220 405,248 435,200" stroke="url(#sp-w)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <circle cx="295" cy="292" r="6" fill="#5dd8ee" />
          <circle cx="350" cy="270" r="6" fill="#5dd8ee" />
          <circle cx="405" cy="248" r="6" fill="#5dd8ee" />
          <line x1="304" y1="405" x2="304" y2="158" stroke="url(#sp-t)" strokeWidth="16" strokeLinecap="round" />
          <polygon points="304,108 282,158 326,158" fill="#4dd8f0" />
          <line x1="385" y1="385" x2="452" y2="192" stroke="#22c55e" strokeWidth="10" strokeLinecap="round" opacity="0.9" />
          <polygon points="465,158 442,194 470,204" fill="#22c55e" opacity="0.9" />
        </svg>
      </div>

      <div style={{
        animation: 'splashFadeIn 0.6s ease-out 0.3s both',
        textAlign: 'center',
      }}>
        <div style={{
          fontSize: 26,
          fontWeight: 800,
          letterSpacing: '0.25em',
          color: '#f0f0ff',
          fontFamily: 'JetBrains Mono, monospace',
          textShadow: '0 0 24px rgba(34,211,238,0.4)',
          marginBottom: 10,
        }}>
          VOLARIX
        </div>
        <div style={{
          fontSize: 10,
          color: '#6b6b9a',
          fontFamily: 'JetBrains Mono, monospace',
          letterSpacing: '0.18em',
          textTransform: 'uppercase',
        }}>
          Market Stress Intelligence
        </div>
      </div>

      <div style={{
        height: 2,
        background: 'linear-gradient(90deg,transparent,#22d3ee,transparent)',
        animation: 'splashLine 1.5s ease-out 0.8s forwards',
        borderRadius: 2,
      }} />
    </div>
  )
}
