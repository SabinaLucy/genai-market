import PageHeader from '../components/PageHeader'
import useIsMobile from '../useIsMobile'
import { useTheme } from '../ThemeContext'

const FEATURES = [
  { icon: '📈', title: 'VIX Forecasting', desc: 'Explore short-term volatility predictions and their uncertainty.' },
  { icon: '🧠', title: 'Regime Detection', desc: 'Identify low, elevated, and crisis market conditions.' },
  { icon: '🔎', title: 'Explainable AI', desc: 'Explore the factors influencing model predictions.' },
  { icon: '✨', title: 'Ask Volarix', desc: 'Understand market signals through natural-language conversations.' },
  { icon: '📊', title: 'Backtesting', desc: 'Compare historical strategies, performance, and risk.' },
  { icon: '🌐', title: 'Market Intelligence', desc: 'Connect economic indicators, market data, and news.' },
]
const PIPELINE = [
  { icon: '◈', title: 'Market Data', sub: 'VIX · FRED · News' },
  { icon: '✳', title: 'ML Models', sub: 'LSTM · XGBoost' },
  { icon: '◎', title: 'Explainability', sub: 'SHAP · Conformal prediction' },
  { icon: '✦', title: 'AI Insights', sub: 'Claude · Market analysis' },
]
const STACK = ['Python','PyTorch','XGBoost','SHAP','FastAPI','React','Claude','FRED','Vercel']
const LINKS = [
  { title: 'GitHub', url: 'https://github.com/SabinaLucy/genai-market' },
  { title: 'Portfolio', url: 'https://sabinabimbi.dev' },
  { title: 'API Docs', url: 'https://sabinalucy-volarix.hf.space/docs' },
]
export default function AboutPage() {
  const isMobile = useIsMobile()
  const isDark = useTheme()
  const ink = isDark ? '#f3eaff' : '#27143f'
  const muted = isDark ? '#bfb2d0' : '#695d7c'
  const accent = isDark ? '#c9aaff' : '#7544bd'
  const border = isDark ? 'rgba(178,137,235,.24)' : 'rgba(115,75,180,.19)'
  const surface = isDark ? 'linear-gradient(145deg,#221a34,#161222)' : 'linear-gradient(145deg,#fff,#f5efff)'
  const panel = isDark ? 'linear-gradient(135deg,#20182f,#141222)' : 'linear-gradient(135deg,#fff,#f4edff)'
  const card = { background: surface, border: `1px solid ${border}`, borderRadius: 18, padding: isMobile ? 16 : 21 }
  const label = { fontSize: 11, fontWeight: 700, letterSpacing: '0.13em', color: accent, margin: 0 }
  const title = { color: ink, fontSize: isMobile ? 23 : 28, fontWeight: 750, letterSpacing: '-.035em', lineHeight: 1.2, margin: '8px 0 10px' }
  const body = { color: muted, fontSize: 14, lineHeight: 1.75, margin: 0 }
  return (
    <div className="fade-up" style={{ fontFamily: 'var(--font-sans)' }}>
      <PageHeader title="About Volarix" subtitle="The intelligence behind the platform" page="about" />
      <main style={{ maxWidth: 1050, margin: '0 auto', padding: isMobile ? '16px 16px 48px' : '26px 32px 76px', display: 'grid', gap: isMobile ? 26 : 34 }}>
        <section style={{ position: 'relative', overflow: 'hidden', borderRadius: 25, border: `1px solid ${border}`, padding: isMobile ? '26px 18px' : '44px 48px', textAlign: 'center', background: isDark ? 'radial-gradient(ellipse at 50% 0%,#49316d 0%,#241638 45%,#111522 100%)' : 'radial-gradient(ellipse at 50% 0%,#e7d6ff 0%,#f6efff 53%,#eef4ff 100%)' }}>
          <p style={label}>✦ THE INTELLIGENCE BEHIND VOLARIX</p>
          <svg viewBox="0 0 420 145" style={{ width: '100%', maxWidth: 520, height: 'auto', display: 'block', margin: '12px auto 0' }} aria-label="Volarix V symbol over an upward market signal" role="img">
            <defs><linearGradient id="volarix-about-v" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#b9e7ff"/><stop offset=".5" stopColor="#b89cff"/><stop offset="1" stopColor="#8053e9"/></linearGradient><linearGradient id="volarix-about-area" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#9b75f7" stopOpacity=".3"/><stop offset="1" stopColor="#9b75f7" stopOpacity="0"/></linearGradient></defs>
            {[30,60,90,120].map(y=><line key={y} x1="0" x2="420" y1={y} y2={y} stroke={isDark ? '#7a5b9e' : '#b69bdb'} strokeOpacity=".3" strokeDasharray="3 6"/>)}
            <path d="M0 128 L35 120 L64 129 L94 105 L124 113 L156 88 L185 95 L218 64 L248 74 L284 43 L317 53 L353 24 L383 34 L420 11 L420 145 L0 145Z" fill="url(#volarix-about-area)"/>
            <path d="M0 128 L35 120 L64 129 L94 105 L124 113 L156 88 L185 95 L218 64 L248 74 L284 43 L317 53 L353 24 L383 34 L420 11" fill="none" stroke="#b89cff" strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="353" cy="24" r="5" fill="#dac4ff"/>
            <g transform="translate(170 25)"><path d="M8 17 Q5 11 13 11 H23 Q27 11 30 17 L43 49 L57 17 Q60 11 65 11 H75 Q82 11 77 20 L51 69 Q43 81 35 69 Z" fill="url(#volarix-about-v)"/></g>
          </svg>
          <h1 style={{ color: ink, fontSize: isMobile ? 30 : 43, fontWeight: 800, letterSpacing: '-.055em', lineHeight: 1.12, margin: '12px auto', maxWidth: 730 }}>Market uncertainty.<br/><span style={{ color: accent }}>Made understandable.</span></h1>
          <p style={{ ...body, maxWidth: 600, margin: '0 auto' }}>Volarix brings together predictive machine learning, market intelligence, and explainable AI to help you see beyond the volatility.</p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 9, flexWrap: 'wrap', marginTop: 22 }}>{['Forecast','Explain','Understand'].map(x=><span key={x} style={{ border: `1px solid ${border}`, background: isDark ? '#332447' : '#eee2ff', color: accent, borderRadius: 30, padding: '7px 15px', fontSize: 12, fontWeight: 600 }}>{x}</span>)}</div>
        </section>
        <section style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: isMobile ? 8 : 15 }}>
          {[['870K+','Records analyzed'],['10 days','Forecast horizon'],['3','Market regimes']].map(([value,name])=><div key={name} style={{ ...card, textAlign: 'center', padding: isMobile ? '16px 5px' : 22 }}><div style={{ color: accent, fontWeight: 800, fontSize: isMobile ? 19 : 29, letterSpacing: '-.035em' }}>{value}</div><div style={{ color: muted, fontSize: isMobile ? 10 : 12, marginTop: 5 }}>{name}</div></div>)}
        </section>
        <section>
          <p style={label}>01 / EXPLORE THE PLATFORM</p><h2 style={title}>Intelligence at every layer.</h2><p style={{ ...body, marginBottom: 18 }}>Everything you need to explore market risk, in one connected experience.</p>
          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? 'repeat(2,minmax(0,1fr))' : 'repeat(3,minmax(0,1fr))', gap: 12 }}>
            {FEATURES.map(f=><article key={f.title} style={{ ...card, transition: 'transform .2s, box-shadow .2s' }} onMouseEnter={e=>{e.currentTarget.style.transform='translateY(-3px)';e.currentTarget.style.boxShadow='0 12px 30px rgba(125,78,195,.12)'}} onMouseLeave={e=>{e.currentTarget.style.transform='';e.currentTarget.style.boxShadow=''}}><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}><span style={{ display: 'grid', placeItems: 'center', width: 43, height: 43, borderRadius: 12, background: isDark ? '#332544' : '#efe4ff', fontSize: 22 }}>{f.icon}</span><span style={{ color: accent, fontSize: 18 }}>↗</span></div><h3 style={{ color: ink, fontSize: isMobile ? 13 : 15, fontWeight: 700, margin: '0 0 8px' }}>{f.title}</h3><p style={{ ...body, fontSize: isMobile ? 11 : 13 }}>{f.desc}</p></article>)}
          </div>
        </section>
        <section>
          <p style={label}>02 / HOW IT WORKS</p><h2 style={title}>The intelligence pipeline.</h2><p style={{ ...body, marginBottom: 18 }}>From financial data to machine learning predictions and understandable insights.</p>
          <div style={{ ...card, background: panel, padding: isMobile ? 14 : 24 }}>
            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? 'repeat(2,minmax(0,1fr))' : 'repeat(4,minmax(0,1fr))', gap: 12 }}>
              {PIPELINE.map((step,i)=><div key={step.title} style={{ background: isDark ? '#2b203d' : '#eee5fa', borderRadius: 14, padding: '20px 10px', textAlign: 'center' }}><div style={{ fontSize: 25, color: accent, marginBottom: 10 }}>{step.icon}</div><div style={{ color: ink, fontSize: 14, fontWeight: 700 }}>{step.title}</div><div style={{ color: muted, fontSize: 11, marginTop: 6 }}>{step.sub}</div></div>)}
            </div>
            <div style={{ color: accent, textAlign: 'center', paddingTop: 16, fontSize: 12 }}>↓ &nbsp; Interactive Volarix dashboard</div>
          </div>
        </section>
        <section style={{ ...card, background: panel }}>
          <p style={label}>BUILT TO DEPLOY</p><h2 style={title}>From research to real-world systems.</h2><p style={body}>A complete AI platform spanning data engineering, model development, explainability, backend APIs, and modern frontend design.</p>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 18 }}>{STACK.map(t=><span key={t} style={{ color: accent, background: isDark ? '#312441' : '#efe5fa', border: `1px solid ${border}`, borderRadius: 9, padding: '7px 11px', fontSize: 12 }}>{t}</span>)}</div>
        </section>
        <section>
          <p style={label}>03 / THE CREATOR</p>
          <div style={{ ...card, background: panel, marginTop: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 15 }}>
              <div style={{ flexShrink: 0, width: 60, height: 60, display: 'grid', placeItems: 'center', borderRadius: 17, background: isDark ? '#3a2854' : '#e9d8ff', border: `1px solid ${border}`, color: accent, fontWeight: 800, fontSize: 22 }}>SB</div>
              <div><h2 style={{ color: ink, fontSize: 19, margin: 0 }}>Sabina Bimbi</h2><p style={{ color: accent, fontSize: 12, margin: '5px 0' }}>M.S. Data Science · AI & Machine Learning</p><p style={{ color: muted, fontSize: 11, margin: 0 }}>Michigan Technological University</p></div>
            </div>
            <p style={{ ...body, marginTop: 18 }}>Designed and developed to demonstrate how machine learning research, explainable AI, and full-stack engineering can become a functional financial intelligence platform.</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginTop: 16 }}>{LINKS.map(l=><a key={l.title} href={l.url} target="_blank" rel="noopener noreferrer" style={{ color: accent, fontSize: 13, fontWeight: 650, textDecoration: 'none' }}>{l.title} ↗</a>)}</div>
          </div>
        </section>
        <section style={{ ...card, textAlign: 'center', background: isDark ? 'linear-gradient(135deg,#30204d,#171326)' : 'linear-gradient(135deg,#f0e3ff,#f8f4ff)', padding: isMobile ? 24 : 34 }}>
          <div style={{ fontSize: 26, marginBottom: 9 }}>🛡️</div><h2 style={{ ...title, fontSize: isMobile ? 21 : 25 }}>Built for insight. Not certainty.</h2><p style={{ ...body, maxWidth: 640, margin: '0 auto' }}>Volarix supports understanding and exploration, not guaranteed market predictions. Forecasts and analyses are for informational and research purposes, not financial advice.</p>
        </section>
      </main>
    </div>
  )
}
