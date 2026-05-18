export default function ThemeToggle({ isDark, onToggle }) {
  return (
    <button onClick={onToggle} aria-label="Toggle theme" style={{
      width: 44, height: 24, borderRadius: 999,
      background: isDark ? '#1e1e2e' : '#e0e0f0',
      border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.12)'}`,
      position: 'relative', cursor: 'pointer',
      transition: 'background 0.25s, border-color 0.25s', flexShrink: 0,
    }}>
      <span style={{
        position: 'absolute', top: 3,
        left: isDark ? 3 : 22,
        width: 16, height: 16, borderRadius: '50%',
        background: isDark ? '#6366f1' : '#f59e0b',
        transition: 'left 0.25s',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 9,
        boxShadow: isDark ? '0 0 8px rgba(99,102,241,0.5)' : '0 0 8px rgba(245,158,11,0.5)',
      }}>
        {isDark ? '🌙' : '☀️'}
      </span>
    </button>
  )
}
