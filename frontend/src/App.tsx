import { useEffect, useState } from 'react'
import './App.css'
import { fetchHealth, fetchScenarioCatalog } from './api'
import { ArchitecturePage } from './components/ArchitecturePage'
import { AblationPage } from './pages/AblationPage'
import { HistoryPage } from './pages/HistoryPage'
import { StudioPage } from './pages/StudioPage'
import type {
  ExampleItem,
  HealthResponse,
  ScenarioSubdivision,
} from './types'

type AppPage = 'studio' | 'ablation' | 'history' | 'architecture'
type Theme = 'light' | 'dark'

const THEME_KEY = 'av_ethics.theme'

function getInitialTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(THEME_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch { /* ignore */ }
  return 'light'
}

export default function App() {
  const [page, setPage] = useState<AppPage>('studio')
  const [theme, setTheme] = useState<Theme>(getInitialTheme)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [examples, setExamples] = useState<ExampleItem[]>([])
  const [subdivisions, setSubdivisions] = useState<ScenarioSubdivision[]>([])
  const [defaultJsonInput, setDefaultJsonInput] = useState('{\n  "loading": true\n}')
  const [defaultTextInput, setDefaultTextInput] = useState('')
  const [defaultExampleId, setDefaultExampleId] = useState<string | null>(null)
  const [defaultSubdivisionId, setDefaultSubdivisionId] = useState('')
  const [loadingInitial, setLoadingInitial] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function boot() {
      try {
        const [healthPayload, catalogPayload] = await Promise.all([fetchHealth(), fetchScenarioCatalog()])
        if (cancelled) return
        setHealth(healthPayload)
        setExamples(catalogPayload.examples)
        setSubdivisions(catalogPayload.subdivisions)

        const defaultJson = catalogPayload.examples.find((e) => e.mode === 'json')
        const defaultText = catalogPayload.examples.find((e) => e.mode === 'text')
        if (defaultJson && typeof defaultJson.value !== 'string') {
          setDefaultJsonInput(JSON.stringify(defaultJson.value, null, 2))
          setDefaultExampleId(defaultJson.id)
          if (defaultJson.subdivision_id) setDefaultSubdivisionId(defaultJson.subdivision_id)
        }
        if (defaultText && typeof defaultText.value === 'string') {
          setDefaultTextInput(defaultText.value)
        }
        if (!defaultJson && catalogPayload.subdivisions[0]) {
          setDefaultSubdivisionId(catalogPayload.subdivisions[0].id)
        }
      } finally {
        if (!cancelled) setLoadingInitial(false)
      }
    }
    void boot()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try { window.localStorage.setItem(THEME_KEY, theme) } catch { /* ignore */ }
  }, [theme])

  const statusClass = loadingInitial ? 'booting' : health?.status === 'ok' ? 'online' : 'offline'

  return (
    <div className="root">
      {/* ── NAVBAR ── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <span className="brand-icon">◈</span>
          <span className="brand-name">AV·ETHICS</span>
          <span className="brand-sub">Ethical Decision Pipeline</span>
        </div>

        <div className="navbar-center">
          <div className="nav-tabs">
            {([ ['studio', '⬡', 'Studio'], ['ablation', '⊟', 'Ablation'], ['history', '☰', 'History'], ['architecture', '◫', 'Architecture'] ] as [AppPage, string, string][]).map(([id, icon, label]) => (
              <button
                key={id}
                type="button"
                className={`nav-tab ${page === id ? 'active' : ''}`}
                onClick={() => setPage(id)}
              >
                <span className="nav-tab-icon">{icon}</span> {label}
              </button>
            ))}
          </div>
        </div>

        <div className="navbar-actions">
          {health ? (
            <div className="navbar-runtime">
              <div className="rt-chip">
                <span className="rt-dot" data-state={health.rag.runtime_available ? 'ok' : 'warn'} />
                <span className="rt-label">RAG</span>
                <span className="rt-val">{health.rag.runtime_available ? 'Ready' : 'Fallback'}</span>
              </div>
              <div className="rt-chip">
                <span className="rt-dot" data-state={health.reasoning.runtime_available ? 'ok' : 'warn'} />
                <span className="rt-label">LLM</span>
                <span className="rt-val">{health.reasoning.runtime_available ? (health.reasoning.model_name ?? 'Ready') : 'Degraded'}</span>
              </div>
            </div>
          ) : (
            <span className="rt-loading">Connecting…</span>
          )}

          <div className={`status-badge ${statusClass}`}>
            <span className="status-pulse" />
            {loadingInitial ? 'Booting' : health?.status ?? 'Offline'}
          </div>

          <button
            type="button"
            className={`theme-toggle ${theme === 'dark' ? 'is-dark' : ''}`}
            onClick={() => setTheme((t) => t === 'light' ? 'dark' : 'light')}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <span aria-hidden="true">{theme === 'dark' ? '☀' : '🌙'}</span>
          </button>
        </div>
      </nav>

      {/* ── PAGES ── */}
      {page === 'studio' && (
        <StudioPage
          examples={examples}
          subdivisions={subdivisions}
          initialJsonInput={defaultJsonInput}
          initialTextInput={defaultTextInput}
          initialExampleId={defaultExampleId}
          initialSubdivisionId={defaultSubdivisionId}
        />
      )}
      {page === 'ablation' && (
        <AblationPage
          examples={examples}
          subdivisions={subdivisions}
          initialJsonInput={defaultJsonInput}
          initialExampleId={defaultExampleId}
        />
      )}
      {page === 'history' && <HistoryPage />}
      {page === 'architecture' && <ArchitecturePage />}
    </div>
  )
}
