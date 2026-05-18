import { useEffect, useState } from 'react'
import useVolarix    from './useVolarix'
import useIsMobile   from './useIsMobile'
import { ThemeContext } from './ThemeContext'
import { REGIME_COLOR } from './utils'

import Sidebar       from './components/Sidebar'
import MobileNav     from './components/MobileNav'
import SplashScreen  from './components/SplashScreen'
import RegimeToast   from './components/RegimeToast'
import OverviewPage  from './pages/OverviewPage'
import RegimePage    from './pages/RegimePage'
import BulletinPage  from './pages/BulletinPage'
import AskPage       from './pages/AskPage'
import BacktestPage  from './pages/BacktestPage'
import AboutPage     from './pages/AboutPage'

export default function App() {
  const isMobile = useIsMobile()
  const [page,   setPage]   = useState('overview')
  const [isDark, setIsDark] = useState(true)
  const [showSplash, setShowSplash] = useState(isMobile)

  const {
    latest, predict, analogues, shap, backtest, bulletin,
    horizon, setHorizon,
    lastUpdated, regimeAlert, setRegimeAlert,
  } = useVolarix()

  const regime      = latest.data?.regime      || 'ELEVATED'
  const regimeColor = latest.data?.regime_color || REGIME_COLOR.ELEVATED

  /* Theme: update DOM and CSS vars in the same synchronous call */
  useEffect(() => {
    const root = document.documentElement
    if (isDark) {
      root.classList.remove('light')
    } else {
      root.classList.add('light')
    }
  }, [isDark])

  /* Scroll to top on page change */
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [page])

  useEffect(() => {
    const color  = regimeColor
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 32
    const ctx    = canvas.getContext('2d')
    ctx.fillStyle = `${color}22`
    ctx.beginPath(); ctx.roundRect?.(0,0,32,32,6) || ctx.rect(0,0,32,32); ctx.fill()
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(4,22); ctx.lineTo(9,14); ctx.lineTo(13,18)
    ctx.lineTo(17,10); ctx.lineTo(21,16); ctx.lineTo(25,8); ctx.lineTo(28,12)
    ctx.stroke()
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(28,12,2.5,0,Math.PI*2); ctx.fill()
    const link = document.querySelector("link[rel~='icon']") || document.createElement('link')
    link.rel = 'icon'; link.href = canvas.toDataURL(); document.head.appendChild(link)
  }, [regimeColor])

  useEffect(() => {
    document.title = `Volarix${latest.data?.vix ? ` · VIX ${latest.data.vix}` : ''} · ${regime}`
  }, [latest.data?.vix, regime])

  const pages = {
    overview: <OverviewPage latest={latest} predict={predict} horizon={horizon} setHorizon={setHorizon} lastUpdated={lastUpdated} />,
    regime:   <RegimePage   latest={latest} analogues={analogues} shap={shap} />,
    bulletin: <BulletinPage bulletin={bulletin} latest={latest} />,
    ask:      <AskPage      bulletinData={bulletin.data} latest={latest} />,
    backtest: <BacktestPage backtest={backtest} />,
    about:    <AboutPage />,
  }

  const toggleTheme = () => setIsDark(d => !d)

  return (
    <ThemeContext.Provider value={isDark}>
      {showSplash && <SplashScreen onDone={() => setShowSplash(false)} />}
      <RegimeToast alert={regimeAlert} onDismiss={() => setRegimeAlert(null)} />

      <div style={{
        position: 'fixed', top: 0, right: 0, left: isMobile ? 0 : 240, height: 300,
        background: `radial-gradient(ellipse 50% 40% at 70% -5%, ${regimeColor}10 0%, transparent 70%)`,
        pointerEvents: 'none', zIndex: 0, transition: 'background 1.2s ease',
      }} />

      <Sidebar active={page} onNav={setPage} latestData={latest.data} isDark={isDark} onThemeToggle={toggleTheme} />
      <MobileNav active={page} onNav={setPage} isDark={isDark} onThemeToggle={toggleTheme} />

      <div className="main-content" style={{ position: 'relative', zIndex: 1, paddingBottom: page === 'ask' ? 0 : undefined }}>
        {pages[page] || pages.overview}
      </div>
    </ThemeContext.Provider>
  )
}
