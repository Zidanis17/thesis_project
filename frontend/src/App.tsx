import { startTransition, useEffect, useState } from 'react'
import './App.css'
import { fetchExamples, fetchHealth, runScenario } from './api'
import { ArtifactTabs } from './components/ArtifactTabs'
import { StageInspector } from './components/StageInspector'
import { StageTimeline } from './components/StageTimeline'
import type {
  ArtifactTabKey,
  ExampleItem,
  HealthResponse,
  InputEditorMode,
  ReplayStage,
  RunEnvelope,
} from './types'

const DEFAULT_ARTIFACT_TAB: ArtifactTabKey = 'parser_result'

function prettyJson(value: Record<string, unknown>) {
  return JSON.stringify(value, null, 2)
}

function normalizeReplayDelay(durationMs: number, isFinalStage: boolean) {
  if (isFinalStage) {
    return 280
  }
  return Math.max(420, Math.min(1200, Math.round(durationMs * 0.85 + 240)))
}

function getCurrentStages(runEnvelope: RunEnvelope | null): ReplayStage[] {
  if (!runEnvelope) {
    return []
  }
  return runEnvelope.payload.replay
}

function getCurrentStage(runEnvelope: RunEnvelope | null, stageIndex: number) {
  const stages = getCurrentStages(runEnvelope)
  if (!stages.length) {
    return null
  }
  return stages[Math.min(stageIndex, stages.length - 1)]
}

function getPreviousStage(runEnvelope: RunEnvelope | null, stageIndex: number) {
  const stages = getCurrentStages(runEnvelope)
  if (stageIndex <= 0 || !stages.length) {
    return null
  }
  return stages[Math.min(stageIndex - 1, stages.length - 1)]
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [examples, setExamples] = useState<ExampleItem[]>([])
  const [mode, setMode] = useState<InputEditorMode>('json')
  const [jsonInput, setJsonInput] = useState('{\n  "loading": true\n}')
  const [textInput, setTextInput] = useState('')
  const [runEnvelope, setRunEnvelope] = useState<RunEnvelope | null>(null)
  const [activeStageIndex, setActiveStageIndex] = useState(0)
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactTabKey>(DEFAULT_ARTIFACT_TAB)
  const [loadingInitial, setLoadingInitial] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isReplaying, setIsReplaying] = useState(false)
  const [clientError, setClientError] = useState<string | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [healthPayload, examplesPayload] = await Promise.all([fetchHealth(), fetchExamples()])
        if (cancelled) {
          return
        }

        setHealth(healthPayload)
        setExamples(examplesPayload)

        const defaultJsonExample = examplesPayload.find((item) => item.mode === 'json')
        const defaultTextExample = examplesPayload.find((item) => item.mode === 'text')

        if (defaultJsonExample && typeof defaultJsonExample.value !== 'string') {
          setJsonInput(prettyJson(defaultJsonExample.value))
        }
        if (defaultTextExample && typeof defaultTextExample.value === 'string') {
          setTextInput(defaultTextExample.value)
        }
      } catch (error) {
        if (!cancelled) {
          setRequestError(error instanceof Error ? error.message : 'Failed to load the frontend shell.')
        }
      } finally {
        if (!cancelled) {
          setLoadingInitial(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const stages = getCurrentStages(runEnvelope)
    if (!stages.length) {
      return
    }

    let cancelled = false
    setIsReplaying(true)

    async function replay() {
      for (let index = 0; index < stages.length; index += 1) {
        if (cancelled) {
          return
        }
        startTransition(() => {
          setActiveStageIndex(index)
        })

        const delay = normalizeReplayDelay(index === 0 ? 0 : stages[index].duration_ms, index === stages.length - 1)
        await new Promise((resolve) => window.setTimeout(resolve, delay))
      }

      if (!cancelled) {
        setIsReplaying(false)
      }
    }

    void replay()
    return () => {
      cancelled = true
      setIsReplaying(false)
    }
  }, [runEnvelope])

  const currentStage = getCurrentStage(runEnvelope, activeStageIndex)
  const previousStage = getPreviousStage(runEnvelope, activeStageIndex)
  const hasSuccessResult = runEnvelope?.kind === 'success'

  async function handleRun() {
    setClientError(null)
    setRequestError(null)
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
          setClientError('Enter a natural-language scenario before running the pipeline.')
          setIsSubmitting(false)
          return
        }
        payload = trimmed
      }

      const response = await runScenario(payload, mode)
      startTransition(() => {
        setRunEnvelope(response)
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

  function handleReset() {
    setRunEnvelope(null)
    setActiveStageIndex(0)
    setClientError(null)
    setRequestError(null)
    setSelectedArtifact(DEFAULT_ARTIFACT_TAB)
  }

  function handleLoadExample(example: ExampleItem) {
    setMode(example.mode)
    setRunEnvelope(null)
    setClientError(null)
    setRequestError(null)
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
      setClientError('Formatting failed because the JSON is not valid yet.')
    }
  }

  return (
    <div className="app-shell">
      <div className="grid-backdrop" />
      <header className="hero">
        <section className="hero-copy">
          <p className="eyebrow">AV Ethics Pipeline Studio</p>
          <h1>See what each stage receives, changes, and returns.</h1>
          <p className="hero-text">
            Run a scenario, then follow the pipeline as a clean sequence of transformations. Each replay step
            shows the incoming snapshot, the outgoing JSON, and a plain-language explanation of what changed.
          </p>
          <div className="hero-points">
            <span className="hero-pill">Structured JSON or free text</span>
            <span className="hero-pill">Animated stage replay</span>
            <span className="hero-pill">Before/after diff storytelling</span>
          </div>
        </section>

        <section className="runtime-panel">
          <div className="section-heading">
            <span>Runtime Overview</span>
            <span className="runtime-status">{loadingInitial ? 'Booting' : health?.status ?? 'Offline'}</span>
          </div>
          {health ? (
            <>
              <div className="runtime-grid">
                <div className="runtime-card">
                  <span className="runtime-label">Knowledge Base</span>
                  <span className="runtime-value">{health.knowledge_base_path ?? 'Unavailable'}</span>
                </div>
                <div className="runtime-card">
                  <span className="runtime-label">RAG</span>
                  <span className={`runtime-value ${health.rag.runtime_available ? 'is-ready' : 'is-warning'}`}>
                    {health.rag.runtime_available ? 'Ready' : 'Fallback mode'}
                  </span>
                </div>
                <div className="runtime-card">
                  <span className="runtime-label">Reasoning Model</span>
                  <span className={`runtime-value ${health.reasoning.runtime_available ? 'is-ready' : 'is-warning'}`}>
                    {health.reasoning.runtime_available
                      ? health.reasoning.model_name ?? 'Ready'
                      : health.reasoning.runtime_error ?? 'Unavailable'}
                  </span>
                </div>
              </div>

              {health.warnings.length ? (
                <div className="runtime-warnings">
                  {health.warnings.map((warning) => (
                    <div key={warning} className="notice notice-warning runtime-warning">
                      {warning}
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <p className="empty-state">Loading backend runtime information.</p>
          )}
        </section>
      </header>

      <main className="workspace">
        <section className="input-panel panel">
          <div className="section-heading">
            <span>Scenario Input</span>
            <span className="artifact-caption">Clean submission workspace</span>
          </div>

          <div className="panel-intro">
            <p className="panel-title">Prepare a scenario and replay the pipeline response below.</p>
            <p className="panel-text">
              Switch between JSON and natural language, load a sample, then inspect how each backend stage
              transforms the snapshot.
            </p>
          </div>

          <div className="mode-switcher">
            <button
              type="button"
              className={`mode-button ${mode === 'json' ? 'is-selected' : ''}`}
              onClick={() => setMode('json')}
            >
              Structured JSON
            </button>
            <button
              type="button"
              className={`mode-button ${mode === 'text' ? 'is-selected' : ''}`}
              onClick={() => setMode('text')}
            >
              Natural Language
            </button>
          </div>

          <div className="example-bank">
            <div className="subsection-heading">
              <span>Examples</span>
              <span>{examples.length ? 'Load a baseline and iterate' : 'No examples loaded yet'}</span>
            </div>
            <div className="example-strip">
              {examples.map((example) => (
                <button
                  key={example.id}
                  type="button"
                  className="example-pill"
                  onClick={() => handleLoadExample(example)}
                >
                  {example.label}
                </button>
              ))}
            </div>
          </div>

          <div className="editor-toolbar">
            <button type="button" className="toolbar-button" onClick={handleRun} disabled={isSubmitting}>
              {isSubmitting ? 'Running...' : 'Run Scenario'}
            </button>
            {mode === 'json' ? (
              <button type="button" className="toolbar-button ghost" onClick={handleFormatJson}>
                Format JSON
              </button>
            ) : null}
            <button type="button" className="toolbar-button ghost" onClick={handleReset}>
              Reset View
            </button>
          </div>

          {mode === 'json' ? (
            <textarea
              aria-label="Structured JSON editor"
              className="scenario-editor"
              value={jsonInput}
              onChange={(event) => setJsonInput(event.target.value)}
            />
          ) : (
            <textarea
              aria-label="Natural language scenario editor"
              className="scenario-editor"
              value={textInput}
              onChange={(event) => setTextInput(event.target.value)}
            />
          )}

          {clientError ? <div className="notice notice-error">{clientError}</div> : null}
          {requestError ? <div className="notice notice-error">{requestError}</div> : null}
        </section>

        <section className="output-column">
          {hasSuccessResult ? (
            <section className="summary-band">
              <article className="summary-card emphasis">
                <span className="summary-label">Recommended Action</span>
                <strong>{runEnvelope.payload.summary.recommended_action ?? 'Deterministic only'}</strong>
              </article>
              <article className="summary-card">
                <span className="summary-label">Deterministic Best Action</span>
                <strong>{runEnvelope.payload.summary.deterministic_best_action}</strong>
              </article>
              <article className="summary-card">
                <span className="summary-label">Dominant Framework</span>
                <strong>{runEnvelope.payload.summary.dominant_framework ?? 'Unavailable'}</strong>
              </article>
              <article className="summary-card">
                <span className="summary-label">Input Mode</span>
                <strong>{runEnvelope.payload.summary.input_mode}</strong>
              </article>
            </section>
          ) : null}

          {hasSuccessResult && !runEnvelope.payload.summary.rag_runtime_available ? (
            <div className="notice notice-warning">RAG degraded: {runEnvelope.payload.summary.rag_runtime_error}</div>
          ) : null}
          {hasSuccessResult && !runEnvelope.payload.summary.reasoning_runtime_available ? (
            <div className="notice notice-warning">
              Reasoning degraded: {runEnvelope.payload.summary.reasoning_runtime_error}
            </div>
          ) : null}

          <section className="output-panel panel">
            <div className="section-heading">
              <span>Pipeline Replay</span>
              <span className="artifact-caption">
                {isSubmitting ? 'Backend processing in progress' : currentStage ? `Viewing ${currentStage.label}` : 'Idle'}
              </span>
            </div>

            <div className="panel-intro compact">
              <p className="panel-title">Track the transformation from one stage to the next.</p>
              <p className="panel-text">
                Each step highlights the JSON paths that changed and explains the stage in plain language.
              </p>
            </div>

            {isSubmitting ? (
              <div className="processing-banner" data-testid="processing-banner">
                <div className="processing-pulse" />
                <div>
                  <strong>Backend processing</strong>
                  <p>The replay inspector will animate each stage as soon as the response arrives.</p>
                </div>
              </div>
            ) : null}

            {runEnvelope?.kind === 'error' ? (
              <div className="notice notice-error">
                <strong>{runEnvelope.payload.error.code}</strong>
                <p>{runEnvelope.payload.error.message}</p>
              </div>
            ) : null}

            <StageTimeline
              stages={getCurrentStages(runEnvelope)}
              activeIndex={activeStageIndex}
              isReplaying={isReplaying}
              onSelectStage={(index) => setActiveStageIndex(index)}
            />
            <StageInspector currentStage={currentStage} previousStage={previousStage} isReplaying={isReplaying} />
          </section>

          {hasSuccessResult ? (
            <ArtifactTabs
              selectedTab={selectedArtifact}
              onSelectTab={setSelectedArtifact}
              artifacts={runEnvelope.payload.artifacts}
            />
          ) : null}
        </section>
      </main>
    </div>
  )
}
