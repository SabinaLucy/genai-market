import { useState, useRef, useEffect } from 'react'
import { Sparkles, MessageCircle, BookOpen, ChartNoAxesCombined, ChartColumnIncreasing, ShieldCheck, ChartLine, TrendingUp, ArrowUpRight, Camera, Image as ImageIcon, FileText, Plus, ChevronDown, Check, ArrowUp, X } from 'lucide-react'
import { fetchAsk, fetchAskUpload, fetchHeadlines } from '../api'
import PageHeader from '../components/PageHeader'
import useIsMobile from '../useIsMobile'
import { REGIME_COLOR, REGIME_TEXT } from '../utils'


const SUGGESTIONS_MOBILE = [
  { icon: '📊', text: 'What is driving volatility?' },
  { icon: '🛡️', text: 'Should I reduce equity exposure?' },
  { icon: '📉', text: 'Compare to 2020 COVID crash' },
  { icon: '💵', text: 'What does ELEVATED mean?' },
  { icon: '⚡', text: 'Is a spike likely this week?' },
]

const SUGGESTIONS_DESK = [
  { icon: '📊', text: 'What is driving volatility right now?' },
  { icon: '🛡️', text: 'Reduce equity exposure?' },
  { icon: '📉', text: 'Compare to 2020 COVID crash' },
  { icon: '💵', text: 'ELEVATED + bonds?' },
  { icon: '⚡', text: 'Is a spike likely in 5 days?' },
]

const FEATURED = [{icon:ChartColumnIncreasing,label:'Market volatility',text:"What's moving the markets?",question:'What is driving volatility?'},{icon:ShieldCheck,label:'Understand risk',text:"Explore today's market regime",question:'Explain the current market regime'},{icon:ChartLine,label:'Historical patterns',text:'Compare with past market events',question:'How does today compare with 2020?'},{icon:TrendingUp,label:'Market outlook',text:'Explore potential scenarios',question:'What is the market outlook?'}]
const STYLE_OPTIONS = [{value:'auto',label:'Adaptive',desc:'Adjusts naturally to you',icon:Sparkles},{value:'simple',label:'Simple',desc:'Easy explanations',icon:MessageCircle},{value:'detailed',label:'Detailed',desc:'More depth and context',icon:BookOpen},{value:'technical',label:'Technical',desc:'Advanced analysis',icon:ChartNoAxesCombined}]
const STYLE_KEY = 'volarix-response-style'
const STYLE_HINTS = {simple:'Use simple English suitable for a beginner.',detailed:'Give a thorough, clear explanation.',technical:'Use precise financial terminology and explain assumptions.'}
function ShieldIcon({ color, size = 30 }) {
  return (
    <svg width={size} height={size} viewBox="210 72 260 370" fill="none">
      <defs>
        <linearGradient id="si-s" x1="340" y1="72" x2="340" y2="428" gradientUnits="userSpaceOnUse"><stop offset="0%" stopColor="#1b6b82"/><stop offset="100%" stopColor="#0c2f4a"/></linearGradient>
        <linearGradient id="si-t" x1="304" y1="405" x2="304" y2="108" gradientUnits="userSpaceOnUse"><stop offset="0%" stopColor="#1d8fa8"/><stop offset="100%" stopColor="#5ee0f5"/></linearGradient>
        <linearGradient id="si-w" x1="235" y1="310" x2="435" y2="200" gradientUnits="userSpaceOnUse"><stop offset="0%" stopColor="#5dd8ee"/><stop offset="100%" stopColor="#a8eef8"/></linearGradient>
      </defs>
      <path d="M340 72 L210 118 L210 240 C210 330 270 400 340 428 C410 400 470 330 470 240 L470 118 Z" fill="url(#si-s)" stroke="#2a9ab8" strokeWidth="1.5" strokeLinejoin="round"/>
      <polyline points="235,310 268,268 295,292 322,240 350,270 378,220 405,248 435,200" stroke="url(#si-w)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
      <circle cx="295" cy="292" r="4.5" fill="#5dd8ee"/><circle cx="350" cy="270" r="4.5" fill="#5dd8ee"/>
      <line x1="304" y1="405" x2="304" y2="158" stroke="url(#si-t)" strokeWidth="14" strokeLinecap="round"/>
      <polygon points="304,108 282,158 326,158" fill="#4dd8f0"/>
      <line x1="385" y1="385" x2="452" y2="192" stroke={color} strokeWidth="8" strokeLinecap="round" opacity="0.9"/>
      <polygon points="465,158 442,194 470,204" fill={color} opacity="0.9"/>
    </svg>
  )
}

function formatResponse(text) {
  if (!text) return null
  const lines = text.split('\n')
  const out = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i].trim()
    if (!line) { i++; continue }
    if (/^#{1,3}\s/.test(line)) {
      out.push(<p key={i} style={{ fontWeight: 700, fontSize: 13, margin: '8px 0 3px' }}>{line.replace(/^#{1,3}\s/, '')}</p>)
      i++; continue
    }
    if (/^\d+\.\s/.test(line)) {
      const items = []
      while (i < lines.length && /^\d+\.\s/.test(lines[i].trim())) {
        items.push(lines[i].replace(/^\d+\.\s/, '').trim()); i++
      }
      out.push(<ol key={`ol${i}`} style={{ paddingLeft: 18, margin: '5px 0', display: 'flex', flexDirection: 'column', gap: 3 }}>
        {items.map((t, j) => <li key={j} style={{ fontSize: 14, lineHeight: 1.755 }}>{inlineFmt(t)}</li>)}
      </ol>)
      continue
    }
    if (/^[-•*]\s/.test(line)) {
      const items = []
      while (i < lines.length && /^[-•*]\s/.test(lines[i].trim())) {
        items.push(lines[i].replace(/^[-•*]\s/, '').trim()); i++
      }
      out.push(<ul key={`ul${i}`} style={{ paddingLeft: 18, margin: '5px 0', display: 'flex', flexDirection: 'column', gap: 3 }}>
        {items.map((t, j) => <li key={j} style={{ fontSize: 14, lineHeight: 1.755 }}>{inlineFmt(t)}</li>)}
      </ul>)
      continue
    }
    out.push(<p key={i} style={{ fontSize: 14, lineHeight: 1.755, margin: '3px 0' }}>{inlineFmt(line)}</p>)
    i++
  }
  return out
}

function inlineFmt(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
    p.startsWith('**') && p.endsWith('**') ? <strong key={i}>{p.slice(2, -2)}</strong> : p
  )
}

const STORAGE_KEY = 'volarix_chat_history'
const MAX_MESSAGES = 50 // keep last 50 messages

export default function AskPage({ bulletinData, latest }) {
  const isMobile = useIsMobile()
  const [messages, setMessages] = useState(() => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) : []
  } catch { return [] }
})
  const [responseStyle, setResponseStyle] = useState(() => { try { return localStorage.getItem(STYLE_KEY) || 'auto' } catch { return 'auto' } })
  const [attachment, setAttachment] = useState(null)
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)
  const [styleOpen,setStyleOpen] = useState(false)
  const [attachOpen,setAttachOpen] = useState(false)
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [headlines, setHeadlines] = useState([])
  const bottomRef               = useRef(null)
  useEffect(() => { try { localStorage.setItem(STYLE_KEY, responseStyle) } catch {} }, [responseStyle])

  const regime = latest?.data?.regime || bulletinData?.regime || 'ELEVATED'
  const color  = REGIME_COLOR[regime] || '#f59e0b'
  const text   = REGIME_TEXT[regime]  || '#fbbf24'
  const SUGGESTIONS = isMobile ? SUGGESTIONS_MOBILE : SUGGESTIONS_DESK

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])


  /* Save to localStorage whenever messages change */
useEffect(() => {
  try {
    const toSave = messages.slice(-MAX_MESSAGES)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave))
  } catch {}
}, [messages])


useEffect(() => {                                  
  fetchHeadlines()
    .then(res => setHeadlines(res.data?.headlines || []))
    .catch(() => setHeadlines([]))
}, [])                                             

const clearHistory = () => {
  setMessages([])
  localStorage.removeItem(STORAGE_KEY)
}

  const handleFile = e => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    const valid = ['image/png','image/jpeg','image/webp','image/gif','application/pdf']
    const limit = file.type === 'application/pdf' ? 10 : 5
    if (!valid.includes(file.type)) { setError('Choose a PNG, JPEG, WebP, GIF or PDF file.'); return }
    if (file.size > limit * 1024 * 1024) { setError(`File must be ${limit} MB or smaller.`); return }
    setAttachment(file); setError(null)
  }

  const send = async (q) => {
    q = (typeof q === 'string' ? q : input).trim()
    if ((!q && !attachment) || loading) return
    setInput(''); setError(null)
    setMessages(m => [...m, { role: 'user', text: attachment ? `${q || 'Please analyze this file.'} 📎 ${attachment.name}` : q }])
    setLoading(true)
    try {
      const headlinesContext = headlines?.length > 0
  ? `\n\nTODAY'S HEADLINES:\n${headlines.map((h, i) => `${i+1}. ${h.title} (${h.source})`).join('\n')}`
  : ''

const context = (bulletinData?.bulletin || 
  `Current market data: VIX ${latest?.data?.vix || 'unavailable'}, 
   Regime: ${latest?.data?.regime || 'unavailable'}, 
   Date: ${latest?.data?.date || new Date().toISOString().split('T')[0]}`) + headlinesContext

const history = messages.slice(-8).filter(m => m.role === 'user' || m.role === 'ai').map(m => ({ role: m.role === 'ai' ? 'assistant' : 'user', content: m.text.slice(0, 3000) }))
      const res = attachment
        ? await fetchAskUpload(attachment, q || 'Please analyze this file.', context, history, responseStyle)
        : await fetchAsk(q, context, history, responseStyle)
      setAttachment(null)
      setMessages(m => [...m, { role: 'ai', text: res.data.answer }])
    } catch (e) {
      setError(e?.response?.status === 429 ? 'AI request limit reached. Please try again later.' : (e?.response?.data?.detail || 'Unable to get a response right now. Please try again.'))
    } finally { setLoading(false) }
  }

  return (
    <>
      <style>{`
        .ask-style-control{position:relative;z-index:10}
        .ask-style-wrap,.ask-attach-wrap{position:relative}
        .ask-style-pill{display:flex;align-items:center;gap:8px;background:rgba(168,126,238,.13);border:1px solid rgba(170,133,240,.4);border-radius:999px;padding:9px 13px;color:var(--text-1);cursor:pointer;font:500 13px var(--font-sans)}
        .ask-style-pill svg,.ask-style-option svg,.ask-attach-menu svg{color:#b497f6}
        .ask-style-menu,.ask-attach-menu{position:absolute;z-index:30;min-width:240px;padding:8px;background:var(--bg-sidebar);border:1px solid rgba(171,133,240,.4);border-radius:15px;box-shadow:0 18px 45px rgba(0,0,0,.28)}
        .ask-style-menu{top:calc(100% + 8px);right:0}
        .ask-attach-menu{bottom:calc(100% + 8px);left:0}
        .ask-menu-heading{display:block;color:var(--text-3);font-size:10px;letter-spacing:.12em;padding:7px 10px}
        .ask-style-option,.ask-attach-menu button{width:100%;display:flex;align-items:center;gap:11px;padding:10px;border:0;border-radius:10px;background:transparent;color:var(--text-1);cursor:pointer;text-align:left;font:13px var(--font-sans)}
        .ask-style-option:hover,.ask-attach-menu button:hover,.ask-style-option.selected{background:rgba(166,125,233,.14)}
        .ask-style-option>span{display:flex;flex-direction:column;flex:1;gap:3px}.ask-style-option small{color:var(--text-2);font-size:11px}
        .ask-attachment-controls{padding:6px 12px!important}.ask-attach-button{display:flex;align-items:center;gap:6px;border-radius:10px!important;padding:8px 11px!important}
        .ask-feature-icon{color:#b497f6}.ask-redesign .ask-feature-card{font-family:var(--font-sans)}

        .ask-attachment-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:7px 18px;font-family:var(--font-sans)}
        .ask-attach-button{border:1px solid rgba(171,132,238,.4);background:rgba(171,132,238,.1);color:var(--text-1);border-radius:9px;padding:7px 11px;font-size:12px;cursor:pointer}
        .ask-attached-name{font-size:12px;color:var(--text-2);overflow-wrap:anywhere}.ask-attached-name button{margin-left:6px;border:0;background:transparent;color:var(--text-1);cursor:pointer}

        .ask-redesign{font-family:var(--font-sans);color:var(--text-1)}
        .ask-redesign button,.ask-redesign textarea,.ask-redesign select,.ask-redesign p,.ask-redesign li{font-family:var(--font-sans)}
        .ask-feature-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;width:100%;max-width:660px;margin:6px auto 0}
        .ask-feature-card{display:flex;flex-direction:column;align-items:flex-start;text-align:left;gap:9px;min-width:0;padding:18px;border:1px solid rgba(175,141,240,.29);border-radius:17px;background:radial-gradient(ellipse at 95% 0%,rgba(165,123,241,.13),transparent 72%),var(--bg-card);color:var(--text-1);cursor:pointer;transition:transform .2s,border-color .2s}
        .ask-feature-card:hover{transform:translateY(-3px);border-color:#b58cf4}
        .ask-feature-top{display:flex;align-items:center;justify-content:space-between;width:100%}
        .ask-feature-icon{display:grid;place-items:center;width:40px;height:40px;border-radius:12px;background:rgba(171,132,238,.15);font-size:21px}
        .ask-feature-arrow{font-size:19px;color:#b995f3}
        .ask-feature-card strong{font-size:14px;font-weight:700}
        .ask-feature-card>span:last-child{font-size:13px;line-height:1.55;color:var(--text-2)}
        .ask-redesign .bubble-ai{background:var(--bg-card);border:1px solid rgba(175,141,240,.24);border-radius:16px}
        .ask-redesign .bubble-user{background:rgba(151,110,219,.19);border:1px solid rgba(175,141,240,.35);border-radius:16px}
        .ask-style-control{display:flex;align-items:center;justify-content:flex-end;gap:9px;padding:9px 18px;color:var(--text-2);font-size:12px}
        .ask-style-control select{background:var(--bg-card);border:1px solid var(--border);border-radius:9px;color:var(--text-1);padding:6px 10px;font-size:13px}
        @media(max-width:420px){.ask-feature-grid{gap:8px}.ask-feature-card{padding:12px}.ask-feature-card strong{font-size:13px}.ask-feature-card>span:last-child{font-size:12px}}

        @keyframes bounce { 0%,100%{transform:translateY(0);opacity:.4} 50%{transform:translateY(-4px);opacity:1} }
        @keyframes msgIn  { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }
        .ask-page-mobile {
          position: fixed;
          top: 0; left: 0; right: 0; 
          height: 100svh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          background: var(--bg-app);
          z-index: 1;
        }
        .ask-messages {
          flex: 1;
          overflow-y: auto;
          overflow-x: hidden;
          -webkit-overflow-scrolling: touch;
        }
        .ask-input-bar {
          flex-shrink: 0;
          width: 100%;
          background: var(--bg-sidebar);
          border-top: 1px solid var(--border);
          padding: 10px 12px 120px;
          box-sizing: border-box;
        }
        .ask-input-row {
          display: flex;
          align-items: flex-end;
          gap: 8px;
          background: var(--bg-card);
          border-radius: 14px;
          padding: 5px 5px 5px 14px;
          box-sizing: border-box;
          width: 100%;
          max-width: 100%;
          overflow: hidden;
        }
        .ask-textarea {
          flex: 1;
          min-width: 0;
          outline: none;
          background: transparent;
          color: var(--text-1);
          font-size: 14px;
          font-family: var(--font-sans);
          border: none;
          resize: none;
          line-height: 1.5;
          height: 72px;
          overflow-y: auto;
          padding: 0;
          display: block;
          word-break: break-word;
        }
        .ask-textarea::placeholder { color: var(--text-3); }
        .ask-send-btn {
          width: 34px; height: 34px; min-width: 34px;
          border-radius: 10px;
          flex-shrink: 0;
          font-size: 18px; font-weight: 700;
          display: flex; align-items: center; justify-content: center;
          transition: all 0.2s;
          cursor: pointer;
        }
      `}</style>

      {isMobile ? (
        /* ── MOBILE: fully fixed layout, nothing scrolls except messages ── */
        <div className="ask-page-mobile ask-redesign">

          {/* Header — fixed at top */}
          <PageHeader
            title="Ask Volarix"
            subtitle="Your AI assistant for understanding market volatility"
            page="ask"
            regime={regime}
            regimeLabel={latest?.data?.regime_label}
          />

          {messages.length > 0 && (
  <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '6px 14px 0', background: 'var(--bg-app)' }}>
    <button onClick={clearHistory} style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-sans)', background: 'transparent', border: '1px solid var(--border)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer' }}>
      Clear history
    </button>
  </div>
)}

          <div className="ask-style-control"><div className="ask-style-wrap"><button type="button" className="ask-style-pill" onClick={()=>setStyleOpen(v=>!v)} aria-expanded={styleOpen}><Sparkles size={15}/><span>{STYLE_OPTIONS.find(x=>x.value===responseStyle)?.label||'Adaptive'}</span><ChevronDown size={14}/></button>{styleOpen&&<div className="ask-style-menu"><span className="ask-menu-heading">RESPONSE STYLE</span>{STYLE_OPTIONS.map(option=>{const Icon=option.icon;return <button type="button" key={option.value} className={`ask-style-option ${responseStyle===option.value?'selected':''}`} onClick={()=>{setResponseStyle(option.value);setStyleOpen(false)}}><Icon size={18}/><span><strong>{option.label}</strong><small>{option.desc}</small></span>{responseStyle===option.value&&<Check size={16}/>}</button>})}</div>}</div></div>
          {/* Messages — only this scrolls */}
          <div className="ask-messages" style={{ padding: '16px 14px 8px' }}>
            {messages.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, paddingTop: 16 }}>
                <div style={{ width: 60, height: 60, borderRadius: 16, background: `${color}15`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ShieldIcon color={color} size={40} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-1)', marginBottom: 4 }}>What would you like to explore?</p>
                </div>
<div className="ask-feature-grid">{FEATURED.map((item,i)=>{const Icon=item.icon;return <button key={i} className="ask-feature-card" onClick={()=>send(item.question)} disabled={loading}><span className="ask-feature-top"><span className="ask-feature-icon"><Icon size={21} strokeWidth={1.8}/></span><ArrowUpRight size={17} className="ask-feature-arrow"/></span><strong>{item.label}</strong><span>{item.text}</span></button>})}</div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, flexDirection: m.role === 'user' ? 'row-reverse' : 'row', marginBottom: 14, animation: 'msgIn 0.2s ease both' }}>
                <div style={{ width: 26, height: 26, borderRadius: '50%', flexShrink: 0, marginTop: 2, background: m.role === 'user' ? 'rgba(99,102,241,0.15)' : `${color}15`, border: `1px solid ${m.role === 'user' ? 'rgba(99,102,241,0.3)' : `${color}40`}`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  {m.role === 'user' ? <span style={{ fontSize: 10, fontWeight: 700, color: '#c4b5fd', fontFamily: 'var(--font-sans)' }}>U</span> : <ShieldIcon color={color} size={18} />}
                </div>
                <div className={m.role === 'user' ? 'bubble-user' : 'bubble-ai'} style={{ padding: '10px 13px', maxWidth: '85%', color: 'var(--text-1)', display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {m.role === 'ai' ? formatResponse(m.text) : <p style={{ fontSize: 14, lineHeight: 1.75, margin: 0 }}>{m.text}</p>}
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: 'flex', gap: 10, marginBottom: 14 }}>
                <div style={{ width: 26, height: 26, borderRadius: '50%', background: `${color}15`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  <ShieldIcon color={color} size={18} />
                </div>
                <div className="bubble-ai" style={{ padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 5 }}>
                  {[0,1,2].map(i => <span key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: color, display: 'inline-block', animation: `bounce 1.2s ease-in-out ${i*0.2}s infinite` }} />)}
                </div>
              </div>
            )}

            {error && (
              <div style={{ fontSize: 12, color: '#f87171', padding: '9px 12px', borderRadius: 10, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', marginBottom: 14 }}>⚠️ {error}</div>
            )}
            <div ref={bottomRef} style={{ height: 4 }} />
          </div>

          <div className="ask-attachment-controls"><input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp,image/gif,application/pdf" style={{display:'none'}} onChange={handleFile}/><input ref={cameraInputRef} type="file" accept="image/*" capture="environment" style={{display:'none'}} onChange={handleFile}/><div className="ask-attach-wrap"><button type="button" className="ask-attach-button" onClick={()=>setAttachOpen(v=>!v)} disabled={loading}><Plus size={17}/> Attach</button>{attachOpen&&<div className="ask-attach-menu"><button onClick={()=>{setAttachOpen(false);fileInputRef.current?.click()}}><ImageIcon size={19}/><span>Upload image or chart</span></button><button onClick={()=>{setAttachOpen(false);cameraInputRef.current?.click()}}><Camera size={19}/><span>Take a photo</span></button><button onClick={()=>{setAttachOpen(false);fileInputRef.current?.click()}}><FileText size={19}/><span>Upload PDF</span></button></div>}</div>{attachment&&<span className="ask-attached-name">📎 {attachment.name}<button type="button" onClick={()=>setAttachment(null)} aria-label="Remove attachment"><X size={14}/></button></span>}</div>
          {/* Input — fixed at bottom, never moves */}
          <div className="ask-input-bar">
            <div className="ask-input-row" style={{ border: `1.5px solid ${input ? `${color}70` : 'var(--border)'}` }}>
              <textarea
                className="ask-textarea"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask about markets, risk, forecast…"
              />
              <button
                className="ask-send-btn"
                onClick={() => send(input)}
                disabled={(!input.trim() && !attachment) || loading}
                style={{ background: input.trim() ? `${color}25` : 'transparent', color: input.trim() ? color : 'var(--text-3)', border: `1.5px solid ${input.trim() ? `${color}60` : 'var(--border)'}` }}
              >↑</button>
            </div>
          </div>

        </div>
      ) : (
        /* ── DESKTOP  */
        <div className="fade-up ask-redesign" style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <PageHeader title="Ask Volarix" subtitle="Your AI assistant for understanding market volatility" page="ask" regime={regime} regimeLabel={latest?.data?.regime_label}>
  {messages.length > 0 && (
    <button onClick={clearHistory} style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-sans)', background: 'transparent', border: '1px solid var(--border)', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
      Clear history
    </button>
  )}
</PageHeader>

          <div className="ask-style-control"><div className="ask-style-wrap"><button type="button" className="ask-style-pill" onClick={()=>setStyleOpen(v=>!v)} aria-expanded={styleOpen}><Sparkles size={15}/><span>{STYLE_OPTIONS.find(x=>x.value===responseStyle)?.label||'Adaptive'}</span><ChevronDown size={14}/></button>{styleOpen&&<div className="ask-style-menu"><span className="ask-menu-heading">RESPONSE STYLE</span>{STYLE_OPTIONS.map(option=>{const Icon=option.icon;return <button type="button" key={option.value} className={`ask-style-option ${responseStyle===option.value?'selected':''}`} onClick={()=>{setResponseStyle(option.value);setStyleOpen(false)}}><Icon size={18}/><span><strong>{option.label}</strong><small>{option.desc}</small></span>{responseStyle===option.value&&<Check size={16}/>}</button>})}</div>}</div></div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '28px 32px 16px', display: 'flex', flexDirection: 'column', gap: 16 }}>
            {messages.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, paddingTop: 28 }}>
                <div style={{ width: 76, height: 76, borderRadius: 20, background: `${color}15`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `0 0 28px ${color}20` }}>
                  <ShieldIcon color={color} size={52} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 19, fontWeight: 700, color: 'var(--text-1)', marginBottom: 5 }}>What would you like to explore?</p>
                </div>
<div className="ask-feature-grid">{FEATURED.map((item,i)=>{const Icon=item.icon;return <button key={i} className="ask-feature-card" onClick={()=>send(item.question)} disabled={loading}><span className="ask-feature-top"><span className="ask-feature-icon"><Icon size={21} strokeWidth={1.8}/></span><ArrowUpRight size={17} className="ask-feature-arrow"/></span><strong>{item.label}</strong><span>{item.text}</span></button>})}</div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} style={{ display: 'flex', gap: 12, flexDirection: m.role === 'user' ? 'row-reverse' : 'row' }}>
                <div style={{ width: 30, height: 30, borderRadius: '50%', flexShrink: 0, marginTop: 2, background: m.role === 'user' ? 'rgba(99,102,241,0.15)' : `${color}15`, border: `1px solid ${m.role === 'user' ? 'rgba(99,102,241,0.3)' : `${color}40`}`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  {m.role === 'user' ? <span style={{ fontSize: 11, fontWeight: 700, color: '#c4b5fd', fontFamily: 'var(--font-sans)' }}>U</span> : <ShieldIcon color={color} size={20} />}
                </div>
                <div className={m.role === 'user' ? 'bubble-user' : 'bubble-ai'} style={{ padding: '12px 16px', maxWidth: '78%', color: 'var(--text-1)', display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {m.role === 'ai' ? formatResponse(m.text) : <p style={{ fontSize: 14, lineHeight: 1.75, margin: 0 }}>{m.text}</p>}
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: 'flex', gap: 12 }}>
                <div style={{ width: 30, height: 30, borderRadius: '50%', background: `${color}15`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  <ShieldIcon color={color} size={20} />
                </div>
                <div className="bubble-ai" style={{ padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 5 }}>
                  {[0,1,2].map(i => <span key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: color, display: 'inline-block', animation: `bounce 1.2s ease-in-out ${i*0.2}s infinite` }} />)}
                </div>
              </div>
            )}

            {error && <div style={{ fontSize: 12, color: '#f87171', padding: '10px 13px', borderRadius: 10, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>⚠️ {error}</div>}
            <div ref={bottomRef} />
          </div>

<div className="ask-attachment-controls"><input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp,image/gif,application/pdf" style={{display:'none'}} onChange={handleFile}/><input ref={cameraInputRef} type="file" accept="image/*" capture="environment" style={{display:'none'}} onChange={handleFile}/><div className="ask-attach-wrap"><button type="button" className="ask-attach-button" onClick={()=>setAttachOpen(v=>!v)} disabled={loading}><Plus size={17}/> Attach</button>{attachOpen&&<div className="ask-attach-menu"><button onClick={()=>{setAttachOpen(false);fileInputRef.current?.click()}}><ImageIcon size={19}/><span>Upload image or chart</span></button><button onClick={()=>{setAttachOpen(false);cameraInputRef.current?.click()}}><Camera size={19}/><span>Take a photo</span></button><button onClick={()=>{setAttachOpen(false);fileInputRef.current?.click()}}><FileText size={19}/><span>Upload PDF</span></button></div>}</div>{attachment&&<span className="ask-attached-name">📎 {attachment.name}<button type="button" onClick={()=>setAttachment(null)} aria-label="Remove attachment"><X size={14}/></button></span>}</div>
          <div style={{ padding: '12px 32px 20px', borderTop: '1px solid var(--border)', background: 'var(--bg-sidebar)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, background: 'var(--bg-card)', border: `1.5px solid ${input ? `${color}70` : 'var(--border)'}`, borderRadius: 14, padding: '10px 10px 10px 16px' }}>
              <textarea value={input} onChange={e => setInput(e.target.value)} placeholder="Ask about volatility, regime, risk…" style={{ flex: 1, minWidth: 0, outline: 'none', background: 'transparent', color: 'var(--text-1)', fontSize: 14, fontFamily: 'var(--font-sans)', border: 'none', resize: 'none', lineHeight: 1.5, height: 52, overflowY: 'auto', padding: 0 }} />
              <button onClick={() => send(input)} disabled={(!input.trim() && !attachment) || loading} style={{ width: 36, height: 36, minWidth: 36, borderRadius: 10, background: input.trim() ? `${color}25` : 'transparent', color: input.trim() ? color : 'var(--text-3)', border: `1.5px solid ${input.trim() ? `${color}60` : 'var(--border)'}`, cursor: input.trim() ? 'pointer' : 'default', fontSize: 18, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}>↑</button>
            </div>
            <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 6, textAlign: 'center', fontFamily: 'var(--font-sans)' }}>Enter for new line · tap ↑ to send · 5 requests/hour</p>
          </div>
        </div>
      )}
    </>
  )
}