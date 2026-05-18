import { useState, useRef, useEffect } from 'react'
import { fetchAsk } from '../api'
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
        {items.map((t, j) => <li key={j} style={{ fontSize: 13, lineHeight: 1.65 }}>{inlineFmt(t)}</li>)}
      </ol>)
      continue
    }
    if (/^[-•*]\s/.test(line)) {
      const items = []
      while (i < lines.length && /^[-•*]\s/.test(lines[i].trim())) {
        items.push(lines[i].replace(/^[-•*]\s/, '').trim()); i++
      }
      out.push(<ul key={`ul${i}`} style={{ paddingLeft: 18, margin: '5px 0', display: 'flex', flexDirection: 'column', gap: 3 }}>
        {items.map((t, j) => <li key={j} style={{ fontSize: 13, lineHeight: 1.65 }}>{inlineFmt(t)}</li>)}
      </ul>)
      continue
    }
    out.push(<p key={i} style={{ fontSize: 13, lineHeight: 1.75, margin: '3px 0' }}>{inlineFmt(line)}</p>)
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
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const bottomRef               = useRef(null)

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

const clearHistory = () => {
  setMessages([])
  localStorage.removeItem(STORAGE_KEY)
}

  const send = async (q) => {
    q = (typeof q === 'string' ? q : input).trim()
    if (!q || loading) return
    setInput(''); setError(null)
    setMessages(m => [...m, { role: 'user', text: q }])
    setLoading(true)
    try {
      const context = bulletinData?.bulletin || 
  `Current market data: VIX ${latest?.data?.vix || 'unavailable'}, 
   Regime: ${latest?.data?.regime || 'unavailable'}, 
   Date: ${latest?.data?.date || new Date().toISOString().split('T')[0]}`

const res = await fetchAsk(q, context)
      setMessages(m => [...m, { role: 'ai', text: res.data.answer }])
    } catch (e) {
      setError(e?.response?.data?.detail || 'Rate limit reached (5/hour) — try again later.')
    } finally { setLoading(false) }
  }

  return (
    <>
      <style>{`
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
        <div className="ask-page-mobile">

          {/* Header — fixed at top */}
          <PageHeader
            title="Ask Volarix"
            subtitle="Chat with the model · 5/hour"
            page="ask"
            regime={regime}
            regimeLabel={latest?.data?.regime_label}
          />

          {messages.length > 0 && (
  <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '6px 14px 0', background: 'var(--bg-app)' }}>
    <button onClick={clearHistory} style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', background: 'transparent', border: '1px solid var(--border)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer' }}>
      Clear history
    </button>
  </div>
)}

          {/* Messages — only this scrolls */}
          <div className="ask-messages" style={{ padding: '16px 14px 8px' }}>
            {messages.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, paddingTop: 16 }}>
                <div style={{ width: 60, height: 60, borderRadius: 16, background: `${color}15`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ShieldIcon color={color} size={40} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-1)', marginBottom: 4 }}>Ask anything about the market</p>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%' }}>
  {/* Row 1: full width */}
  <button onClick={() => send(SUGGESTIONS[0].text)} style={{ fontSize: 13, padding: '11px 14px', borderRadius: 12, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'var(--bg-card)', cursor: 'pointer', textAlign: 'left', display: 'flex', alignItems: 'center', gap: 10, width: '100%' }}>
    <span style={{ fontSize: 15 }}>{SUGGESTIONS[0].icon}</span><span>{SUGGESTIONS[0].text}</span>
  </button>
  {/* Row 2: pair */}
  <div style={{ display: 'flex', gap: 8 }}>
    {[SUGGESTIONS[1], SUGGESTIONS[2]].map((s, i) => (
      <button key={i} onClick={() => send(s.text)} style={{ flex: 1, fontSize: 12, padding: '10px 10px', borderRadius: 12, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'var(--bg-card)', cursor: 'pointer', textAlign: 'left', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 14 }}>{s.icon}</span><span>{s.text}</span>
      </button>
    ))}
  </div>
  {/* Row 3: pair */}
  <div style={{ display: 'flex', gap: 8 }}>
    {[SUGGESTIONS[3], SUGGESTIONS[4]].map((s, i) => (
      <button key={i} onClick={() => send(s.text)} style={{ flex: 1, fontSize: 12, padding: '10px 10px', borderRadius: 12, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'var(--bg-card)', cursor: 'pointer', textAlign: 'left', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 14 }}>{s.icon}</span><span>{s.text}</span>
      </button>
    ))}
  </div>
</div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, flexDirection: m.role === 'user' ? 'row-reverse' : 'row', marginBottom: 14, animation: 'msgIn 0.2s ease both' }}>
                <div style={{ width: 26, height: 26, borderRadius: '50%', flexShrink: 0, marginTop: 2, background: m.role === 'user' ? 'rgba(99,102,241,0.15)' : `${color}15`, border: `1px solid ${m.role === 'user' ? 'rgba(99,102,241,0.3)' : `${color}40`}`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  {m.role === 'user' ? <span style={{ fontSize: 10, fontWeight: 700, color: '#c4b5fd', fontFamily: 'var(--font-mono)' }}>U</span> : <ShieldIcon color={color} size={18} />}
                </div>
                <div className={m.role === 'user' ? 'bubble-user' : 'bubble-ai'} style={{ padding: '10px 13px', maxWidth: '85%', color: 'var(--text-1)', display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {m.role === 'ai' ? formatResponse(m.text) : <p style={{ fontSize: 13, lineHeight: 1.6, margin: 0 }}>{m.text}</p>}
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
                disabled={!input.trim() || loading}
                style={{ background: input.trim() ? `${color}25` : 'transparent', color: input.trim() ? color : 'var(--text-3)', border: `1.5px solid ${input.trim() ? `${color}60` : 'var(--border)'}` }}
              >↑</button>
            </div>
          </div>

        </div>
      ) : (
        /* ── DESKTOP  */
        <div className="fade-up" style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <PageHeader title="Ask Volarix" subtitle="Chat with the model · 5 requests/hour" page="ask" regime={regime} regimeLabel={latest?.data?.regime_label}>
  {messages.length > 0 && (
    <button onClick={clearHistory} style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', background: 'transparent', border: '1px solid var(--border)', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
      Clear history
    </button>
  )}
</PageHeader>

          <div style={{ flex: 1, overflowY: 'auto', padding: '28px 32px 16px', display: 'flex', flexDirection: 'column', gap: 16 }}>
            {messages.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, paddingTop: 28 }}>
                <div style={{ width: 76, height: 76, borderRadius: 20, background: `${color}15`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `0 0 28px ${color}20` }}>
                  <ShieldIcon color={color} size={52} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 19, fontWeight: 700, color: 'var(--text-1)', marginBottom: 5 }}>Ask anything about the market</p>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%', maxWidth: 480, alignItems: 'center' }}>
  <button onClick={() => send(SUGGESTIONS[0].text)} style={{ fontSize: 13, padding: '10px 16px', borderRadius: 10, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'var(--bg-card)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10, width: '70%' }}>
    <span style={{ fontSize: 16 }}>{SUGGESTIONS[0].icon}</span><span>{SUGGESTIONS[0].text}</span>
  </button>
  <div style={{ display: 'flex', gap: 8, width: '100%' }}>
    {[SUGGESTIONS[1], SUGGESTIONS[2]].map((s, i) => (
      <button key={i} onClick={() => send(s.text)} style={{ flex: 1, fontSize: 13, padding: '9px 14px', borderRadius: 10, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'var(--bg-card)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span>{s.icon}</span><span>{s.text}</span>
      </button>
    ))}
  </div>
  <div style={{ display: 'flex', gap: 8, width: '100%' }}>
    {[SUGGESTIONS[3], SUGGESTIONS[4]].map((s, i) => (
      <button key={i} onClick={() => send(s.text)} style={{ flex: 1, fontSize: 13, padding: '9px 14px', borderRadius: 10, border: '1px solid var(--border-light)', color: 'var(--text-1)', background: 'var(--bg-card)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span>{s.icon}</span><span>{s.text}</span>
      </button>
    ))}
  </div>
</div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} style={{ display: 'flex', gap: 12, flexDirection: m.role === 'user' ? 'row-reverse' : 'row' }}>
                <div style={{ width: 30, height: 30, borderRadius: '50%', flexShrink: 0, marginTop: 2, background: m.role === 'user' ? 'rgba(99,102,241,0.15)' : `${color}15`, border: `1px solid ${m.role === 'user' ? 'rgba(99,102,241,0.3)' : `${color}40`}`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  {m.role === 'user' ? <span style={{ fontSize: 11, fontWeight: 700, color: '#c4b5fd', fontFamily: 'var(--font-mono)' }}>U</span> : <ShieldIcon color={color} size={20} />}
                </div>
                <div className={m.role === 'user' ? 'bubble-user' : 'bubble-ai'} style={{ padding: '12px 16px', maxWidth: '78%', color: 'var(--text-1)', display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {m.role === 'ai' ? formatResponse(m.text) : <p style={{ fontSize: 13, lineHeight: 1.7, margin: 0 }}>{m.text}</p>}
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

          <div style={{ padding: '12px 32px 20px', borderTop: '1px solid var(--border)', background: 'var(--bg-sidebar)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, background: 'var(--bg-card)', border: `1.5px solid ${input ? `${color}70` : 'var(--border)'}`, borderRadius: 14, padding: '10px 10px 10px 16px' }}>
              <textarea value={input} onChange={e => setInput(e.target.value)} placeholder="Ask about volatility, regime, risk…" style={{ flex: 1, minWidth: 0, outline: 'none', background: 'transparent', color: 'var(--text-1)', fontSize: 14, fontFamily: 'var(--font-sans)', border: 'none', resize: 'none', lineHeight: 1.5, height: 52, overflowY: 'auto', padding: 0 }} />
              <button onClick={() => send(input)} disabled={!input.trim() || loading} style={{ width: 36, height: 36, minWidth: 36, borderRadius: 10, background: input.trim() ? `${color}25` : 'transparent', color: input.trim() ? color : 'var(--text-3)', border: `1.5px solid ${input.trim() ? `${color}60` : 'var(--border)'}`, cursor: input.trim() ? 'pointer' : 'default', fontSize: 18, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}>↑</button>
            </div>
            <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 6, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>Enter for new line · tap ↑ to send · 5 requests/hour</p>
          </div>
        </div>
      )}
    </>
  )
}