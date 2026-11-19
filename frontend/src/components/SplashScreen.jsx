import { useEffect, useState } from 'react'

export default function SplashScreen({ onDone }) {
  const [fading, setFading] = useState(false)

  useEffect(() => {
    const fadeTimer = setTimeout(() => setFading(true), 1050)
    const doneTimer = setTimeout(() => onDone(), 1450)
    return () => { clearTimeout(fadeTimer); clearTimeout(doneTimer) }
  }, [onDone])

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 9999,
      background: 'radial-gradient(ellipse at center, #211832 0%, #0d0b1a 72%, #080713 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 24,
      opacity: fading ? 0 : 1,
      transition: 'opacity 0.4s ease-out',
    }}>
      <style>{`
        @keyframes splashPulse {
          0%, 100% { transform: scale(1); filter: drop-shadow(0 0 20px rgba(169,125,255,0.38)); }
          50%      { transform: scale(1.05); filter: drop-shadow(0 0 40px rgba(169,125,255,0.60)); }
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
        <svg width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="Volarix V logo">
          <defs>
            <linearGradient id="vol-v" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#d6b9ff"/>
              <stop offset="55%" stopColor="#9d72f6"/>
              <stop offset="100%" stopColor="#6545c8"/>
            </linearGradient>
          </defs>
          <rect x="6" y="6" width="84" height="84" rx="25" fill="#211936" stroke="#8e6bda" strokeWidth="1.5"/>
          <path d="M23 27 L39 27 L48 60 L58 27 L74 27 L57 72 L39 72 Z" fill="url(#vol-v)"/>
          <path d="M58 27 L74 27 L57 72 L48 72 Z" fill="#c5a1ff" opacity=".48"/>
          <circle cx="75" cy="21" r="3" fill="#cdb0ff"/>
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
          textShadow: '0 0 24px rgba(169,125,255,0.38)',
          marginBottom: 10,
        }}>
          VOLARIX
        </div>
        <div style={{
          fontSize: 10,
          color: '#aaa0c4',
          fontFamily: 'JetBrains Mono, monospace',
          letterSpacing: '0.18em',
          textTransform: 'uppercase',
        }}>
          Market Stress Intelligence
        </div>
      </div>

      <div style={{
        height: 2,
        background: 'linear-gradient(90deg,transparent,#a778ff,transparent)',
        animation: 'splashLine 0.8s ease-out 0.15s forwards',
        borderRadius: 2,
      }} />
    </div>
  )
}
