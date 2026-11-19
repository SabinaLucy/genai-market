import PageHeader from '../components/PageHeader'
import useIsMobile from '../useIsMobile'

const violet = '#7135b4'
const ink = '#171321'
const body = { color: '#383142', fontSize: 14, lineHeight: 1.8, margin: 0 }
const eyebrow = { color: violet, fontSize: 11, fontWeight: 750, letterSpacing: '.12em', margin: 0 }
const heading = { color: ink, fontWeight: 800, letterSpacing: '-.04em', lineHeight: 1.2, margin: '10px 0 14px' }
const ICON_PATHS = {
  forecast: <><path d="M3 18l6-6 4 3 8-10"/><path d="M17 5h4v4"/></>,
  layers: <><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/></>,
  bars: <><path d="M3 21h18M5 17h3v4H5zM11 11h3v10h-3zM17 5h3v16h-3z"/></>,
  chat: <path d="M4 5h16v11H9l-5 4V5z"/>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/></>,
  cpu: <><rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 9h6v6H9zM9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4"/></>,
  eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></>,
}
function Icon({ name, size = 24 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={violet} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICON_PATHS[name]}</svg>
}
const features = [
  ['forecast','Forecast','Anticipate volatility'],
  ['layers','Detect Regimes','Understand market states'],
  ['bars','Backtest','Evaluate strategies'],
  ['chat','Ask AI','Explore the why'],
]
const steps = [
  ['database','Market Data','VIX · FRED'],
  ['cpu','AI Forecasting','LSTM · XGBoost'],
  ['layers','Risk Regimes','LOW · ELEVATED · CRISIS'],
  ['eye','Explainability','SHAP'],
  ['chat','AI Assistant','Claude'],
]
const methods = [
  ['forecast','VIX Forecasting','Predicts market volatility over a 10-day horizon using machine learning and historical financial indicators.'],
  ['layers','Market Regimes','Identifies LOW, ELEVATED, and CRISIS market stress states to help interpret changing risk conditions.'],
  ['bars','Strategy Backtesting','Compares historical trading strategies against a buy-and-hold benchmark using risk and return metrics.'],
]
export default function AboutPage() {
  const mobile = useIsMobile()
  const section = { padding: mobile ? '30px 18px' : '42px 46px' }
  return <div className="volarix-about" style={{ fontFamily:'var(--font-sans)', color:ink, background:'#faf8ff', minHeight:'100%' }}>
    <style>{`
      .volarix-about * { box-sizing:border-box }
      .volarix-about .va-hero-v { animation:vaFloat 5s ease-in-out infinite;transform-box:fill-box;transform-origin:center }
      .volarix-about .va-chart { stroke-dasharray:650;stroke-dashoffset:650;animation:vaDraw 5s ease-in-out forwards }
      .volarix-about .va-beam { height:2px;flex:0 0 18px;background:linear-gradient(90deg,#8d54cf,#ead8ff,#8d54cf);background-size:200% 100%;animation:vaFlow 2.5s linear infinite }
      .volarix-about .va-model { animation:vaGlow 3s ease-in-out infinite alternate }
      .volarix-about .va-tile { transition:transform .25s ease,box-shadow .25s ease }
      .volarix-about .va-tile:hover { transform:translateY(-4px);box-shadow:0 12px 28px #9b72c922 }
      .volarix-about .va-reveal { animation:vaRise .8s both }
      @keyframes vaFloat { 50% { transform:translateY(-9px) } }
      @keyframes vaDraw { to { stroke-dashoffset:0 } }
      @keyframes vaFlow { to { background-position:-200% 0 } }
      @keyframes vaGlow { to { box-shadow:0 0 18px #a779e644;border-color:#9b77c7 } }
      @keyframes vaRise { from { opacity:0;transform:translateY(12px) } to { opacity:1;transform:translateY(0) } }
      @media (prefers-reduced-motion:reduce) { .volarix-about .va-hero-v,.volarix-about .va-chart,.volarix-about .va-beam,.volarix-about .va-model,.volarix-about .va-reveal { animation:none!important;stroke-dashoffset:0 } .volarix-about .va-tile { transition:none } }
    `}</style>
    <PageHeader title="About Volarix" subtitle="The intelligence behind the platform" page="about" />
    <main style={{maxWidth:1120,margin:'0 auto',padding:mobile?'12px 10px 32px':'24px 24px 60px'}}>
      <div style={{background:'#fff',border:'1px solid #e8def8',borderRadius:22,overflow:'hidden'}}>
        <section className="va-reveal" style={{...section,background:'radial-gradient(ellipse at 75% 20%,#e7d6ff,#faf8ff 68%)'}}>
          <p style={eyebrow}>ABOUT VOLARIX</p>
          <h1 style={{...heading,fontSize:mobile?32:48,maxWidth:710,marginTop:18}}>From market volatility to <span style={{color:'#8143bf'}}>meaningful insight.</span></h1>
          <p style={{...body,maxWidth:660}}>An AI-powered financial intelligence platform that forecasts volatility, detects market regimes, and explains the why behind market movement.</p>
          <svg viewBox="0 0 420 112" style={{width:'100%',maxWidth:690,height:'auto',display:'block',margin:'12px auto'}} role="img" aria-label="Animated rising financial market signal and Volarix V">
            <defs><linearGradient id="vaGradient"><stop stopColor="#c7a5ff"/><stop offset="1" stopColor="#7135b4"/></linearGradient></defs>
            <path d="M0 94L42 87L79 98L115 64L154 77L193 48L225 59L267 25L310 39L354 15L384 28L420 5V112H0Z" fill="#b98aee" opacity=".12"/>
            <path className="va-chart" d="M0 94L42 87L79 98L115 64L154 77L193 48L225 59L267 25L310 39L354 15L384 28L420 5" fill="none" stroke="#8d54cf" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round"/>
            <g className="va-hero-v"><path d="M177 18Q174 12 182 12H192Q196 12 199 18L212 50L226 18Q229 12 234 12H244Q251 12 246 21L220 70Q212 82 204 70Z" fill="url(#vaGradient)"/></g>
          </svg>
          <div style={{display:'grid',gridTemplateColumns:mobile?'repeat(2,minmax(0,1fr))':'repeat(4,minmax(0,1fr))',gap:10}}>
            {features.map(([icon,title,sub])=><div key={title} className="va-tile" style={{background:'#fff',border:'1px solid #e5d8f6',borderRadius:15,padding:mobile?13:18,display:'grid',gap:7}}><Icon name={icon}/><strong style={{fontSize:14}}>{title}</strong><span style={{fontSize:11,color:'#665a72'}}>{sub}</span></div>)}
          </div>
        </section>
        <section className="va-reveal" style={section}>
          <p style={eyebrow}>01 / THE VISION</p>
          <h2 style={{...heading,fontSize:mobile?25:34}}>We don't predict certainty. We make uncertainty <span style={{color:'#7a3ebb'}}>understandable.</span></h2>
          <p style={body}>Financial markets are complex and constantly changing. Volarix brings clarity to that uncertainty by combining machine learning, economic signals, and explainable AI in a single interactive platform.</p>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,minmax(0,1fr))',gap:10,marginTop:25}}>
            {[['870K+','Records analyzed'],['10-day','Forecast horizon'],['3','Risk regimes']].map(([n,s])=><div key={s}><div style={{fontSize:mobile?21:29,color:'#7337af',fontWeight:800}}>{n}</div><div style={{fontSize:11,color:'#655b70'}}>{s}</div></div>)}
          </div>
        </section>
        <section style={{...section,background:'#f1e9fc'}}>
          <p style={eyebrow}>02 / WHY VOLARIX?</p>
          <h2 style={{...heading,fontSize:mobile?24:32}}>Bridging research and real-world insight.</h2>
          <p style={body}>Volarix was developed to explore how advanced machine learning can solve real financial problems. It turns research methods, from forecasting to explainability, into a functional and accessible tool that helps people understand market risk, not just see predictions.</p>
        </section>
        <section style={section}>
          <p style={eyebrow}>03 / THE INTELLIGENCE PIPELINE</p>
          <h2 style={{...heading,fontSize:mobile?24:32}}>From raw data to actionable insight.</h2>
          <p style={body}>A unified process that brings together market data, predictive modeling, explainability, and generative AI.</p>
          <div style={{background:'#f2eafc',borderRadius:18,padding:mobile?12:22,marginTop:18,display:'flex',alignItems:'center',gap:mobile?4:9,flexWrap:mobile?'wrap':'nowrap',justifyContent:'center'}}>
            {steps.map(([icon,name,sub],i)=><div key={name} style={{display:'contents'}}>
              {i>0 && !mobile && <div className="va-beam"/>}
              <div className={'va-tile '+(i===1?'va-model':'')} style={{flex:mobile?'0 0 calc(50% - 8px)':'1 1 0',minWidth:0,background:'#fff',border:'1px solid #e0d0f2',borderRadius:13,padding:mobile?'13px 5px':'19px 7px',display:'grid',justifyItems:'center',gap:7,textAlign:'center'}}>
                <Icon name={icon}/><strong style={{fontSize:mobile?11:12}}>{name}</strong><span style={{fontSize:10,color:'#70627d'}}>{sub}</span>
              </div>
            </div>)}
          </div>
        </section>
        <section style={{...section,background:'#fcfaff'}}>
          <p style={eyebrow}>04 / HOW IT WORKS</p>
          <h2 style={{...heading,fontSize:mobile?24:32}}>The methods behind the insights.</h2>
          <div style={{display:'grid',gridTemplateColumns:mobile?'1fr':'repeat(3,minmax(0,1fr))',gap:12,marginTop:18}}>
            {methods.map(([icon,name,desc])=><article key={name} className="va-tile" style={{background:'#fff',border:'1px solid #e5d9f4',borderRadius:16,padding:19}}><div style={{display:'flex',alignItems:'center',gap:10,marginBottom:12}}><Icon name={icon}/><strong style={{fontSize:15}}>{name}</strong></div><p style={{...body,fontSize:13}}>{desc}</p></article>)}
          </div>
        </section>
        <section style={section}>
          <div style={{background:'linear-gradient(135deg,#f3e9ff,#fbf9ff)',border:'1px solid #d9c2f3',borderRadius:20,padding:mobile?19:28}}>
            <p style={eyebrow}>05 / ABOUT ME — THE CREATOR</p>
            <div style={{display:'flex',alignItems:'center',gap:15,marginTop:20}}>
              <div style={{width:64,height:64,borderRadius:15,display:'grid',placeItems:'center',background:'linear-gradient(135deg,#d8baff,#af7ce5)',color:'#4e227f',fontSize:23,fontWeight:800,flexShrink:0}}>SB</div>
              <div><h2 style={{...heading,fontSize:mobile?21:25,margin:0}}>Sabina Bimbi</h2><div style={{fontSize:12,color:'#41344f',marginTop:4}}>M.S. Data Science · AI & Machine Learning</div><div style={{fontSize:11,color:'#65566f',marginTop:4}}>Michigan Technological University</div></div>
            </div>
            <h3 style={{...heading,fontSize:mobile?23:29,marginTop:22}}>The mind behind Volarix.</h3>
            <div style={{display:'grid',gap:12}}>
              <p style={body}>Volarix was designed and developed by Sabina Bimbi, a master's student in Data Science at Michigan Technological University, specializing in artificial intelligence and machine learning.</p>
              <p style={body}>The project was created to explore how advanced machine learning can solve real-world financial problems. By bringing together predictive modeling, explainable AI, and full-stack development, Volarix demonstrates how research can become a functional, accessible intelligence platform.</p>
              <p style={body}>It reflects a broader interest in developing intelligent systems that do more than generate predictions: systems that help people understand complex information and make more informed decisions.</p>
            </div>
            <div style={{display:'flex',gap:22,marginTop:20,flexWrap:'wrap'}}>
              <a href="https://github.com/SabinaLucy/genai-market" target="_blank" rel="noopener noreferrer" style={{color:violet,fontWeight:700,fontSize:13}}>GitHub ↗</a>
              <a href="https://sabinabimbi.dev" target="_blank" rel="noopener noreferrer" style={{color:violet,fontWeight:700,fontSize:13}}>Portfolio ↗</a>
            </div>
          </div>
        </section>
        <footer style={{...section,textAlign:'center',background:'#f2eafc'}}>
          <h2 style={{...heading,fontSize:mobile?21:27,color:'#7035a9',fontStyle:'italic'}}>Built for insight. Not certainty.</h2>
          <p style={{...body,fontSize:12}}>Forecasts and analyses are for educational and research purposes, not investment advice.</p>
        </footer>
      </div>
    </main>
  </div>
}
