import { startTransition, useEffect, useState } from 'react'
import './App.css'
import {
  fetchScenarioCatalog,
  fetchHealth,
  fetchScenarioRunById,
  fetchScenarioRunHistory,
  runScenario,
  runScenarioBank,
  runSubdivision,
} from './api'
import { ArtifactTabs } from './components/ArtifactTabs'
import { StageInspector } from './components/StageInspector'
import { StageTimeline } from './components/StageTimeline'
import { ArchitecturePage } from './components/ArchitecturePage'
import { RunHistoryPanel } from './components/RunHistoryPanel'
import { SubdivisionMetricsPanel } from './components/SubdivisionMetricsPanel'
import type {
  ArtifactTabKey,
  EvaluationRunResponse,
  ExampleItem,
  HealthResponse,
  InputEditorMode,
  ReplayStage,
  RunEnvelope,
  ScenarioRunHistoryResponse,
  ScenarioRunRecord,
  ScenarioSubdivision,
} from './types'

type AppPage = 'studio' | 'architecture'
type Theme = 'light' | 'dark'

const DEFAULT_ARTIFACT_TAB: ArtifactTabKey = 'parser_result'
const THEME_STORAGE_KEY = 'av_ethics.theme'

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  try {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (storedTheme === 'light' || storedTheme === 'dark') {
      return storedTheme
    }
  } catch {
    // Ignore storage access issues and keep a safe default.
  }
  return 'light'
}

function prettyJson(value: Record<string, unknown>) {
  return JSON.stringify(value, null, 2)
}

function normalizeReplayDelay(durationMs: number, isFinalStage: boolean) {
  if (isFinalStage) return 280
  return Math.max(420, Math.min(1200, Math.round(durationMs * 0.85 + 240)))
}

function getCurrentStages(runEnvelope: RunEnvelope | null): ReplayStage[] {
  if (!runEnvelope) return []
  return runEnvelope.payload.replay
}

function getCurrentStage(runEnvelope: RunEnvelope | null, stageIndex: number) {
  const stages = getCurrentStages(runEnvelope)
  if (!stages.length) return null
  return stages[Math.min(stageIndex, stages.length - 1)]
}

function getPreviousStage(runEnvelope: RunEnvelope | null, stageIndex: number) {
  const stages = getCurrentStages(runEnvelope)
  if (stageIndex <= 0 || !stages.length) return null
  return stages[Math.min(stageIndex - 1, stages.length - 1)]
}

function getFrameworkRationale(runEnvelope: RunEnvelope | null): string | null {
  if (!runEnvelope || runEnvelope.kind !== 'success') return null
  const rationale = runEnvelope.payload.artifacts.reasoning_result?.rationale
  if (typeof rationale !== 'string') return null
  const trimmed = rationale.trim()
  return trimmed.length > 0 ? trimmed : null
}

export default function App() {
  const [page, setPage] = useState<AppPage>('studio')
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme())
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [examples, setExamples] = useState<ExampleItem[]>([])
  const [subdivisions, setSubdivisions] = useState<ScenarioSubdivision[]>([])
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(null)
  const [selectedSubdivisionId, setSelectedSubdivisionId] = useState<string>('')
  const [mode, setMode] = useState<InputEditorMode>('json')
  const [jsonInput, setJsonInput] = useState('{\n  "loading": true\n}')
  const [textInput, setTextInput] = useState('')
  const [runEnvelope, setRunEnvelope] = useState<RunEnvelope | null>(null)
  const [runHistory, setRunHistory] = useState<ScenarioRunHistoryResponse | null>(null)
  const [evaluationResult, setEvaluationResult] = useState<EvaluationRunResponse | null>(null)
  const [activeStageIndex, setActiveStageIndex] = useState(0)
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactTabKey>(DEFAULT_ARTIFACT_TAB)
  const [selectedHistoryRunId, setSelectedHistoryRunId] = useState<string | null>(null)
  const [loadingInitial, setLoadingInitial] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isBatchSubmitting, setIsBatchSubmitting] = useState(false)
  const [isBankSubmitting, setIsBankSubmitting] = useState(false)
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)
  const [isStoredRunLoading, setIsStoredRunLoading] = useState(false)
  const [isReplaying, setIsReplaying] = useState(false)
  const [clientError, setClientError] = useState<string | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [batchError, setBatchError] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)

  async function refreshRunHistory() {
    setIsHistoryLoading(true)
    setHistoryError(null)
    try {
      const historyPayload = await fetchScenarioRunHistory()
      startTransition(() => {
        setRunHistory(historyPayload)
        if (
          selectedHistoryRunId &&
          !historyPayload.runs.some((storedRun) => storedRun.id === selectedHistoryRunId)
        ) {
          setSelectedHistoryRunId(null)
        }
      })
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Failed to load stored run history.')
    } finally {
      setIsHistoryLoading(false)
    }
  }

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [healthPayload, catalogPayload] = await Promise.all([fetchHealth(), fetchScenarioCatalog()])
        let historyPayload: ScenarioRunHistoryResponse | null = null
        try {
          historyPayload = await fetchScenarioRunHistory()
        } catch (historyLoadError) {
          if (!cancelled) {
            setHistoryError(
              historyLoadError instanceof Error
                ? historyLoadError.message
                : 'Failed to load stored run history.',
            )
          }
        }
        if (cancelled) return
        setHealth(healthPayload)
        setExamples(catalogPayload.examples)
        setSubdivisions(catalogPayload.subdivisions)
        if (historyPayload) {
          setRunHistory(historyPayload)
        }
        const defaultJsonExample = catalogPayload.examples.find((item) => item.mode === 'json')
        const defaultTextExample = catalogPayload.examples.find((item) => item.mode === 'text')
        if (defaultJsonExample && typeof defaultJsonExample.value !== 'string') {
          setJsonInput(prettyJson(defaultJsonExample.value))
          setSelectedExampleId(defaultJsonExample.id)
          if (defaultJsonExample.subdivision_id) {
            setSelectedSubdivisionId(defaultJsonExample.subdivision_id)
          }
        }
        if (defaultTextExample && typeof defaultTextExample.value === 'string') {
          setTextInput(defaultTextExample.value)
        }
        if (!defaultJsonExample && catalogPayload.subdivisions[0]) {
          setSelectedSubdivisionId(catalogPayload.subdivisions[0].id)
        }
      } catch (error) {
        if (!cancelled) {
          setRequestError(error instanceof Error ? error.message : 'Failed to load the frontend shell.')
        }
      } finally {
        if (!cancelled) setLoadingInitial(false)
      }
    }
    void load()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // Ignore storage write issues; theme still applies for this session.
    }
  }, [theme])

  useEffect(() => {
    const stages = getCurrentStages(runEnvelope)
    if (!stages.length) return
    let cancelled = false
    setIsReplaying(true)
    async function replay() {
      for (let index = 0; index < stages.length; index += 1) {
        if (cancelled) return
        startTransition(() => { setActiveStageIndex(index) })
        const delay = normalizeReplayDelay(index === 0 ? 0 : stages[index].duration_ms, index === stages.length - 1)
        await new Promise((resolve) => window.setTimeout(resolve, delay))
      }
      if (!cancelled) setIsReplaying(false)
    }
    void replay()
    return () => { cancelled = true; setIsReplaying(false) }
  }, [runEnvelope])

  const currentStage = getCurrentStage(runEnvelope, activeStageIndex)
  const previousStage = getPreviousStage(runEnvelope, activeStageIndex)
  const hasSuccessResult = runEnvelope?.kind === 'success'
  const currentRunRecord: ScenarioRunRecord | null = runEnvelope?.payload.run ?? null
  const selectedExample = examples.find((example) => example.id === selectedExampleId) ?? null
  const selectedSubdivision = subdivisions.find((subdivision) => subdivision.id === selectedSubdivisionId) ?? null
  const frameworkRationale = getFrameworkRationale(runEnvelope)
  const isBusy = isSubmitting || isBatchSubmitting || isBankSubmitting || isStoredRunLoading

  async function handleRun() {
    setClientError(null)
    setRequestError(null)
    setBatchError(null)
    setHistoryError(null)
    setIsSubmitting(true)
    try {
      let payload: string | Record<string, unknown>
      if (mode === 'json') {
        const parsed = JSON.parse(jsonInput) as unknown
        if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
          setClientError('Structured mode expects a top-level JSON object.')
          setIsSubmitting(false)
          return
        }
        payload = parsed as Record<string, unknown>
      } else {
        const trimmed = textInput.trim()
        if (trimmed.length === 0) {
          setClientError('Enter a natural-language scenario before running.')
          setIsSubmitting(false)
          return
        }
        payload = trimmed
      }
      const response = await runScenario(payload, mode)
      let historyPayload: ScenarioRunHistoryResponse | null = null
      try {
        historyPayload = await fetchScenarioRunHistory()
        setHistoryError(null)
      } catch (historyLoadError) {
        setHistoryError(
          historyLoadError instanceof Error ? historyLoadError.message : 'Failed to refresh stored run history.',
        )
      }
      startTransition(() => {
        setRunEnvelope(response)
        if (historyPayload) {
          setRunHistory(historyPayload)
        }
        setActiveStageIndex(0)
        setSelectedArtifact(DEFAULT_ARTIFACT_TAB)
      })
    } catch (error) {
      if (error instanceof SyntaxError) {
        setClientError('Enter valid JSON before running the scenario.')
      } else {
        setRequestError(error instanceof Error ? error.message : 'Backend request failed.')
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleRunSubdivision() {
    if (!selectedSubdivisionId) {
      setBatchError('Select a subdivision before running the batch.')
      return
    }

    setBatchError(null)
    setRequestError(null)
    setIsBatchSubmitting(true)
    try {
      const response = await runSubdivision(selectedSubdivisionId)
      startTransition(() => {
        setEvaluationResult(response)
      })
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : 'Subdivision batch request failed.')
    } finally {
      setIsBatchSubmitting(false)
    }
  }

  async function handleRunScenarioBank() {
    setBatchError(null)
    setRequestError(null)
    setIsBankSubmitting(true)
    try {
      const response = await runScenarioBank()
      startTransition(() => {
        setEvaluationResult(response)
      })
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : 'Scenario bank evaluation request failed.')
    } finally {
      setIsBankSubmitting(false)
    }
  }

  async function handleLoadStoredRun(runId: string) {
    setHistoryError(null)
    setRequestError(null)
    setSelectedHistoryRunId(runId)
    setIsStoredRunLoading(true)
    try {
      const storedRun = await fetchScenarioRunById(runId)
      startTransition(() => {
        setRunEnvelope(storedRun)
        setActiveStageIndex(0)
        setSelectedArtifact(DEFAULT_ARTIFACT_TAB)
      })
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Failed to load the stored run.')
    } finally {
      setIsStoredRunLoading(false)
    }
  }

  function handleReset() {
    setRunEnvelope(null)
    setEvaluationResult(null)
    setActiveStageIndex(0)
    setClientError(null)
    setRequestError(null)
    setBatchError(null)
    setHistoryError(null)
    setSelectedArtifact(DEFAULT_ARTIFACT_TAB)
  }

  function handleLoadExample(example: ExampleItem) {
    setSelectedExampleId(example.id)
    setMode(example.mode)
    setRunEnvelope(null)
    setClientError(null)
    setRequestError(null)
    if (example.subdivision_id) {
      setSelectedSubdivisionId(example.subdivision_id)
    }
    if (example.mode === 'json' && typeof example.value !== 'string') {
      setJsonInput(prettyJson(example.value))
      return
    }
    if (example.mode === 'text' && typeof example.value === 'string') {
      setTextInput(example.value)
    }
  }

  function handleFormatJson() {
    try {
      setJsonInput(prettyJson(JSON.parse(jsonInput) as Record<string, unknown>))
      setClientError(null)
    } catch {
      setClientError('Formatting failed - JSON is not valid yet.')
    }
  }

  function handleToggleTheme() {
    setTheme((currentTheme) => (currentTheme === 'light' ? 'dark' : 'light'))
  }

  const statusClass = loadingInitial ? 'booting' : health?.status === 'ok' ? 'online' : 'offline'

  return (
    <div className="root">
      {/* ── TOP NAVBAR ── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <span className="brand-icon">◈</span>
          <span className="brand-name">AV·ETHICS</span>
          <span className="brand-sub">Pipeline Studio</span>
        </div>

        <div className="navbar-actions">
          <div className="navbar-runtime">
            {health ? (
              <>
                <div className="rt-chip">
                  <span className="rt-dot" data-state={health.rag.runtime_available ? 'ok' : 'warn'} />
                  <span className="rt-label">RAG</span>
                  <span className="rt-val">{health.rag.runtime_available ? 'Ready' : 'Fallback'}</span>
                </div>
                <div className="rt-chip">
                  <span className="rt-dot" data-state={health.reasoning.runtime_available ? 'ok' : 'warn'} />
                  <span className="rt-label">Reasoning</span>
                  <span className="rt-val">
                    {health.reasoning.runtime_available ? (health.reasoning.model_name ?? 'Ready') : 'Degraded'}
                  </span>
                </div>
                <div className="rt-chip">
                  <span className="rt-dot" data-state="ok" />
                  <span className="rt-label">KB</span>
                  <span className="rt-val">{health.knowledge_base_path ?? '—'}</span>
                </div>
              </>
            ) : (
              <span className="rt-loading">Connecting to runtime…</span>
            )}

            <div className="nav-tabs">
              <button
                type="button"
                className={`nav-tab ${page === 'studio' ? 'active' : ''}`}
                onClick={() => setPage('studio')}
              >
                <span className="nav-tab-icon">⬡</span> Studio
              </button>
              <button
                type="button"
                className={`nav-tab ${page === 'architecture' ? 'active' : ''}`}
                onClick={() => setPage('architecture')}
              >
                <span className="nav-tab-icon">◫</span> Architecture
              </button>
            </div>

            <div className={`status-badge ${statusClass}`}>
              <span className="status-pulse" />
              {loadingInitial ? 'Booting' : health?.status ?? 'Offline'}
            </div>
          </div>

          <button
            type="button"
            className={`theme-toggle ${theme === 'dark' ? 'is-dark' : ''}`}
            onClick={handleToggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <span aria-hidden="true">🌙</span>
          </button>
        </div>
      </nav>

      {/* ── ARCHITECTURE PAGE ── */}
      {page === 'architecture' && <ArchitecturePage />}

      {/* ── STUDIO WORKSPACE ── */}
      {page === 'studio' && (
      <div className="workspace">

        {/* ── LEFT: INPUT PANEL ── */}
        <aside className="input-col">
          <div className="panel-header">
            <span className="panel-tag">01</span>
            <span className="panel-title">Scenario Input</span>
          </div>

          {/* Mode toggle */}
          <div className="seg-control">
            <button
              type="button"
              className={`seg-btn ${mode === 'json' ? 'active' : ''}`}
              onClick={() => setMode('json')}
            >
              <span className="seg-icon">{ }</span> Structured JSON
            </button>
            <button
              type="button"
              className={`seg-btn ${mode === 'text' ? 'active' : ''}`}
              onClick={() => setMode('text')}
            >
              <span className="seg-icon">¶</span> Natural Language
            </button>
          </div>

          {/* Examples dropdown */}
          {examples.length > 0 && (
            <div className="examples-block">
              <label htmlFor="example-select" className="block-label">Load Example</label>
              <div className="select-wrap">
                <select
                  id="example-select"
                  className="example-select"
                  value={selectedExampleId ?? ''}
                  onChange={(e) => {
                    const found = examples.find((ex) => ex.id === e.target.value)
                    if (found) handleLoadExample(found)
                  }}
                >
                  <option value="" disabled>Select a scenario...</option>
                  {examples.map((example) => (
                    <option key={example.id} value={example.id}>
                      [{example.mode.toUpperCase()}] {example.label}
                      {example.subdivision_label ? ` - ${example.subdivision_label}` : ''}
                    </option>
                  ))}
                </select>
                <span className="select-chevron">▾</span>
              </div>

              {selectedExample && (
                <div className="catalog-meta">
                  <span className="catalog-chip">{selectedExample.mode.toUpperCase()}</span>
                  {selectedExample.subdivision_label && (
                    <span className="catalog-chip accent">{selectedExample.subdivision_label}</span>
                  )}
                  {selectedExample.expected_framework && (
                    <span className="catalog-chip">Expected {selectedExample.expected_framework}</span>
                  )}
                </div>
              )}

              {subdivisions.length > 0 && (
                <div className="subdivision-block">
                  <label htmlFor="subdivision-select" className="block-label">Run Subdivision</label>
                  <p className="subdivision-help">
                    Execute every scenario in one subdivision, wait for the batch, then compare framework percentages.
                  </p>
                  <div className="select-wrap">
                    <select
                      id="subdivision-select"
                      className="example-select"
                      value={selectedSubdivisionId}
                      onChange={(e) => {
                        setSelectedSubdivisionId(e.target.value)
                        setEvaluationResult(null)
                        setBatchError(null)
                      }}
                    >
                      <option value="" disabled>Select a subdivision...</option>
                      {subdivisions.map((subdivision) => (
                        <option key={subdivision.id} value={subdivision.id}>
                          {subdivision.label} ({subdivision.scenario_count})
                        </option>
                      ))}
                    </select>
                    <span className="select-chevron">▾</span>
                  </div>

                  {selectedSubdivision && (
                    <div className="catalog-meta">
                      <span className="catalog-chip accent">{selectedSubdivision.label}</span>
                      <span className="catalog-chip">{selectedSubdivision.scenario_count} scenarios</span>
                      {selectedSubdivision.expected_framework && (
                        <span className="catalog-chip">Expected {selectedSubdivision.expected_framework}</span>
                      )}
                      {selectedSubdivision.expectation && (
                        <span className="catalog-chip">{selectedSubdivision.expectation.decision_principle}</span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Editor */}
          <div className="editor-wrap">
            <div className="editor-bar">
              <span className="editor-lang">{mode === 'json' ? 'application/json' : 'text/plain'}</span>
              <div className="editor-actions">
                {mode === 'json' && (
                  <button type="button" className="ghost-btn" onClick={handleFormatJson}>Format</button>
                )}
                <button type="button" className="ghost-btn" onClick={handleReset}>Reset</button>
              </div>
            </div>
            {mode === 'json' ? (
              <textarea
                aria-label="Structured JSON editor"
                className="code-editor"
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
              />
            ) : (
              <textarea
                aria-label="Natural language scenario editor"
                className="code-editor natural"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
              />
            )}
          </div>

          {/* Errors */}
          {clientError && <div className="inline-error"><span>x</span> {clientError}</div>}
          {requestError && <div className="inline-error"><span>x</span> {requestError}</div>}
          {batchError && <div className="inline-error"><span>x</span> {batchError}</div>}

          {/* Run buttons */}
          <div className="run-actions">
            <button
              type="button"
              aria-label="Run Scenario"
              className={`run-btn ${isSubmitting ? 'loading' : ''}`}
              onClick={handleRun}
              disabled={isBusy}
            >
              {isSubmitting
                ? <><span className="run-spinner" /> Processing...</>
                : <><span className="run-icon">▶</span> Execute Pipeline</>
              }
            </button>

            <button
              type="button"
              aria-label="Run Subdivision"
              className={`run-btn secondary ${isBatchSubmitting ? 'loading' : ''}`}
              onClick={handleRunSubdivision}
              disabled={isBusy || !selectedSubdivisionId}
            >
              {isBatchSubmitting
                ? <><span className="run-spinner" /> Processing...</>
                : <><span className="run-icon">▦</span> Run Whole Subdivision</>
              }
            </button>

            <button
              type="button"
              aria-label="Run Full Scenario Bank"
              className={`run-btn secondary ${isBankSubmitting ? 'loading' : ''}`}
              onClick={handleRunScenarioBank}
              disabled={isBusy}
            >
              {isBankSubmitting
                ? <><span className="run-spinner" /> Processing...</>
                : <><span className="run-icon">▦</span> Run Full Scenario Bank</>
              }
            </button>
          </div>
        </aside>

        {/* ── RIGHT: OUTPUT COLUMN ── */}
        <div className="output-col">

          {/* Summary row */}
          {hasSuccessResult && (
            <div className="metrics-row">
              <div className="metric-card accent">
                <span className="metric-label">Deterministic Best</span>
                <strong className="metric-value">{runEnvelope.payload.summary.deterministic_best_action}</strong>
              </div>
              <div className="metric-card">
                <span className="metric-label">Dominant Framework</span>
                <strong className="metric-value">{runEnvelope.payload.summary.dominant_framework ?? '—'}</strong>
              </div>
              <div className="metric-card">
                <span className="metric-label">Framework Rationale</span>
                <p className={`metric-rationale ${frameworkRationale ? '' : 'is-empty'}`}>
                  {frameworkRationale ?? 'No rationale returned by the reasoning stage.'}
                </p>
              </div>
              <div className="metric-card">
                <span className="metric-label">Input Mode</span>
                <strong className="metric-value">{runEnvelope.payload.summary.input_mode}</strong>
              </div>
            </div>
          )}

          {/* Runtime warnings */}
          {hasSuccessResult && !runEnvelope.payload.summary.rag_runtime_available && (
            <div className="warn-bar">⚠ RAG degraded: {runEnvelope.payload.summary.rag_runtime_error}</div>
          )}
          {hasSuccessResult && !runEnvelope.payload.summary.reasoning_runtime_available && (
            <div className="warn-bar">⚠ Reasoning degraded: {runEnvelope.payload.summary.reasoning_runtime_error}</div>
          )}

          <div className="history-panel">
            <div className="panel-header">
              <span className="panel-tag">02A</span>
              <span className="panel-title">Run History</span>
              <span className="panel-status">
                {isHistoryLoading
                  ? <><span className="spin-dot" /> Refreshing</>
                  : runHistory
                  ? `${runHistory.total_runs} stored run${runHistory.total_runs === 1 ? '' : 's'}`
                  : 'Idle - awaiting stored runs'}
              </span>
            </div>
            <RunHistoryPanel
              history={runHistory}
              isLoading={isHistoryLoading}
              isHydrating={isStoredRunLoading}
              error={historyError}
              selectedRunId={selectedHistoryRunId}
              currentRun={currentRunRecord}
              onRefresh={() => { void refreshRunHistory() }}
              onSelectRun={setSelectedHistoryRunId}
              onLoadRun={(runId) => { void handleLoadStoredRun(runId) }}
            />
          </div>

          <div className="subdivision-panel">
            <div className="panel-header">
              <span className="panel-tag">02B</span>
              <span className="panel-title">Evaluation Metrics</span>
              <span className="panel-status">
                {isBatchSubmitting || isBankSubmitting
                  ? <><span className="spin-dot" /> Processing</>
                  : evaluationResult
                  ? `${evaluationResult.scope === 'full_bank' ? 'Full bank' : evaluationResult.subdivision_label ?? 'Subdivision'} · ${evaluationResult.summary.accuracy_pct}% accuracy`
                  : 'Idle - awaiting batch run'}
              </span>
            </div>
            <SubdivisionMetricsPanel
              result={evaluationResult}
              isLoading={isBatchSubmitting || isBankSubmitting}
              error={batchError}
            />
          </div>

          {/* Pipeline replay panel */}
          <div className="pipeline-panel">
            <div className="panel-header">
              <span className="panel-tag">02</span>
              <span className="panel-title">Pipeline Replay</span>
              <span className="panel-status">
                {isSubmitting
                  ? <><span className="spin-dot" /> Processing</>
                  : currentStage
                  ? `Stage · ${currentStage.label}`
                  : 'Idle — awaiting run'}
              </span>
            </div>

            {isSubmitting && (
              <div className="proc-banner" data-testid="processing-banner">
                <div className="proc-bar">
                  <div className="proc-fill" />
                </div>
                <p>Pipeline executing — replay inspector will animate on response.</p>
              </div>
            )}

            {runEnvelope?.kind === 'error' && (
              <div className="error-block">
                <strong className="error-code">{runEnvelope.payload.error.code}</strong>
                <p>{runEnvelope.payload.error.message}</p>
              </div>
            )}

            <div className="replay-body">
              <StageTimeline
                stages={getCurrentStages(runEnvelope)}
                activeIndex={activeStageIndex}
                isReplaying={isReplaying}
                onSelectStage={(index) => setActiveStageIndex(index)}
              />
              <StageInspector
                currentStage={currentStage}
                previousStage={previousStage}
                isReplaying={isReplaying}
              />
            </div>
          </div>

          {/* Artifacts */}
          {hasSuccessResult && (
            <div className="artifacts-panel">
              <div className="panel-header">
                <span className="panel-tag">03</span>
                <span className="panel-title">Artifacts</span>
              </div>
              <ArtifactTabs
                selectedTab={selectedArtifact}
                onSelectTab={setSelectedArtifact}
                artifacts={runEnvelope.payload.artifacts}
              />
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  )
}
