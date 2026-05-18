import PageHeader from '../components/PageHeader'
import useIsMobile from '../useIsMobile'
import { useTheme } from '../ThemeContext'

const SECTIONS = [
  { title: 'What is Volarix?', body: ['Volarix is a financial market stress intelligence system built to forecast the VIX using deep learning, macroeconomic data, and AI-generated analysis. It was developed and designed by Sabina Bimbi, a Masters student in Data Science at Michigan Technological University, to demonstrate end-to-end machine learning applied to a real-world financial problem.', 'Every number you see on this dashboard comes from a live API endpoint, a real market feed, or an AI model making predictions in real time. Nothing is simulated.'] },
  { title: 'How does the forecast work?', body: ['At the core of Volarix is an LSTM neural network that looks at the last 60 trading days of VIX prices combined with six macroeconomic indicators from the Federal Reserve: interest rates, inflation, unemployment, the 10-year Treasury yield, industrial production, and money supply. It also incorporates daily news sentiment scores.', 'The model produces forecasts for three time horizons. Each comes with a confidence interval computed using conformal prediction, guaranteeing the true value falls within the range at least 90% of the time.'] },
  { title: 'What is the regime classification?', body: ['Volarix classifies the current market into one of three regimes using an XGBoost classifier. Stable means VIX is below 20. Elevated means VIX is between 20 and 30. Crisis means VIX is above 30, signalling systemic stress and the need for defensive positioning.'] },
  { title: 'How was the strategy backtest conducted?', body: ['Three strategies were tested against real SPY price data from January 2022 through March 2026. Buy-and-hold as baseline, a naive regime switch, and a hysteresis strategy requiring two consecutive crisis signals before exiting. All apply a 0.05% round-trip transaction cost per switch.'] },
]

const LINKS = [
  { label: '⌥  GitHub',   href: 'https://github.com/SabinaLucy/genai-market',              desc: 'Source code & notebooks' },
  { label: '⎋  API Docs', href: 'https://sabinalucy-volarix.hf.space/docs',                 desc: 'Live FastAPI endpoints'   },
  { label: '◈  HF Space', href: 'https://huggingface.co/spaces/SabinaLucy/volarix',         desc: 'Model hosting & backend'  },
]

export default function AboutPage() {
  const isMobile = useIsMobile()
  const isDark   = useTheme()
  const isLight  = !isDark
  const px       = isMobile ? 16 : 32

  const headingColor  = isLight ? '#5b21b6' : '#c4b5fd'
  const dividerColor  = isLight
    ? 'linear-gradient(90deg,transparent,rgba(91,33,182,0.8),transparent)'
    : 'linear-gradient(90deg,transparent,rgba(167,139,250,0.7),transparent)'
  const techBoxBorder = isLight ? 'rgba(91,33,182,0.3)'  : 'rgba(167,139,250,0.2)'
  const techBoxBg     = isLight ? 'rgba(91,33,182,0.07)' : 'rgba(167,139,250,0.06)'
  const techTextColor = isLight ? '#5b21b6'               : '#c4b5fd'

  return (
    <div className="fade-up">
      <PageHeader title="About" subtitle="Project overview · Architecture · Model details" page="about" />
      <div style={{ display: 'flex', justifyContent: 'center', padding: `${isMobile ? 20 : 48}px ${px}px` }}>
        <div style={{ width: '100%', maxWidth: 740 }}>

          {SECTIONS.map((sec, i) => (
            <div key={i}>
              <div style={{ marginBottom: isMobile ? 20 : 36 }}>
                <h2 style={{ fontSize: isMobile ? 15 : 18, fontWeight: 700, color: headingColor, marginBottom: 10, letterSpacing: '-0.01em' }}>
                  {sec.title}
                </h2>
                {sec.body.map((p, j) => (
                  <p key={j} style={{ fontSize: isMobile ? 14 : 15, color: 'var(--text-1)', lineHeight: 1.85, marginBottom: j < sec.body.length - 1 ? 10 : 0 }}>{p}</p>
                ))}
              </div>
              {i < SECTIONS.length - 1 && (
                <div style={{ height: 1, background: dividerColor, marginBottom: isMobile ? 20 : 36 }} />
              )}
            </div>
          ))}

          <div style={{ height: 1, background: dividerColor, margin: '0 0 20px' }} />

          {/* Tech stack */}
          <div style={{ padding: isMobile ? '14px 16px' : '20px 24px', borderRadius: 14, border: `1px solid ${techBoxBorder}`, background: techBoxBg, marginBottom: 14 }}>
            <p style={{ fontSize: isMobile ? 11 : 13, color: techTextColor, lineHeight: 1.8, fontWeight: isLight ? 500 : 400 }}>
              Built with Python · FastAPI · PyTorch · XGBoost · React 18 · Recharts · Tailwind CSS · Deployed on Hugging Face Spaces and Vercel · Data from Yahoo Finance and FRED
            </p>
          </div>

          {/* Links — same layout on mobile and desktop */}
          <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: isMobile ? 8 : 10, marginBottom: 14 }}>
            {LINKS.map(l => (
              <a key={l.label} href={l.href} target="_blank" rel="noopener noreferrer" style={{
                flex: isMobile ? 'none' : 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: isMobile ? '12px 16px' : '14px 16px',
                borderRadius: 12,
                border: `1px solid ${techBoxBorder}`,
                background: techBoxBg,
                textDecoration: 'none',
                transition: 'all 0.15s',
              }}>
                <div>
                  <div style={{ fontSize: isMobile ? 13 : 12, fontWeight: 600, color: techTextColor, fontFamily: 'var(--font-mono)' }}>{l.label}</div>
                  <div style={{ fontSize: isMobile ? 11 : 10, color: 'var(--text-2)', marginTop: 2 }}>{l.desc}</div>
                </div>
                <span style={{ color: techTextColor, fontSize: 16, opacity: 0.6 }}>→</span>
              </a>
            ))}
          </div>

          {/* Disclaimer */}
          <div style={{ padding: isMobile ? '14px 16px' : '18px 22px', borderRadius: 14, background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)', display: 'flex', gap: 14, alignItems: 'flex-start' }}>
            <div style={{ fontSize: 16, flexShrink: 0, marginTop: 1, opacity: 0.6 }}>⚖️</div>
            <p style={{ fontSize: isMobile ? 11 : 12, color: 'var(--text-2)', lineHeight: 1.75, fontFamily: 'var(--font-mono)' }}>
              <strong style={{ color: 'var(--text-1)', fontWeight: 600 }}>Important Disclaimer.</strong> The forecasts, regime classifications, and analysis presented by Volarix are generated by automated quantitative models and artificial intelligence systems. They are provided for informational and research purposes only and do not constitute investment advice or professional financial guidance. Past model performance does not guarantee future results.
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}
