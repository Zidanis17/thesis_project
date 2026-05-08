import { startTransition, useEffect, useState } from 'react'
import { runScenario } from '../api'
import { ArtifactTabs } from '../components/ArtifactTabs'
import { StageInspector } from '../components/StageInspector'
import { StageTimeline } from '../components/StageTimeline'
import type {
  ArtifactTabKey,
  EvaluationVariant,
  ExampleItem,
  InputEditorMode,
  ReplayStage,
  RunEnvelope,
  ScenarioSubdivision,
} from '../types'

const DEFAULT_ARTIFACT_TAB: ArtifactTabKey = 'parser_result'

function prettyJson(value: Record<string, unknown>) {
  return JSON.stringify(value, null, 2)
}

function normalizeReplayDelay(durationMs: number, isFinalStage: boolean) {
  if (isFinalStage) return 280
  return Math.max(420, Math.min(1200, Math.round(durationMs * 0.85 + 240)))
}

function getFrameworkRationale(runEnvelope: RunEnvelope | null): string | null {
  if (!runEnvelope || runEnvelope.kind !== 'success') return null
  const rationale = runEnvelope.payload.artifacts.reasoning_result?.rationale
  if (typeof rationale !== 'string') return null
  const trimmed = rationale.trim()
  return trimmed.length > 0 ? trimmed : null
}

interface StudioPageProps {
  examples: ExampleItem[]
  subdivisions: ScenarioSubdivision[]
  initialJsonInput: string
  initialTextInput: string
  initialExampleId: string | null
  initialSubdivisionId: string
}

export function StudioPage({
  examples,
  initialJsonInput,
  initialTextInput,
  initialExampleId,
}: StudioPageProps) {
  const [mode, setMode] = useState<InputEditorMode>('json')
  const [jsonInput, setJsonInput] = useState(initialJsonInput)
  const [textInput, setTextInput] = useState(initialTextInput)
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(initialExampleId)
  const [evaluationVariant, setEvaluationVariant] = useState<EvaluationVariant>('full_system')
  const [runEnvelope, setRunEnvelope] = useState<RunEnvelope | null>(null)
  const [activeStageIndex, setActiveStageIndex] = useState(0)
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactTabKey>(DEFAULT_ARTIFACT_TAB)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isReplaying, setIsReplaying] = useState(false)
  const [clientError, setClientError] = useState<string | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)

  // Keep inputs synced when parent provides new values on initial load
  useEffect(() => { setJsonInput(initialJsonInput) }, [initialJsonInput])
  useEffect(() => { setTextInput(initialTextInput) }, [initialTextInput])
  useEffect(() => { setSelectedExampleId(initialExampleId) }, [initialExampleId])

  const stages = runEnvelope?.payload.replay ?? []
  const currentStage: ReplayStage | null = stages[Math.min(activeStageIndex, stages.length - 1)] ?? null
  const previousStage: ReplayStage | null = activeStageIndex > 0 ? stages[Math.min(activeStageIndex - 1, stages.length - 1)] ?? null : null
  const hasSuccessResult = runEnvelope?.kind === 'success'
  const selectedExample = examples.find((ex) => ex.id === selectedExampleId) ?? null
  const frameworkRationale = getFrameworkRationale(runEnvelope)

  useEffect(() => {
    if (!stages.length) return
    let cancelled = false
    setIsReplaying(true)
    async function replay() {
      for (let i = 0; i < stages.length; i++) {
        if (cancelled) return
        startTransition(() => setActiveStageIndex(i))
        const delay = normalizeReplayDelay(i === 0 ? 0 : stages[i].duration_ms, i === stages.length - 1)
        await new Promise((r) => window.setTimeout(r, delay))
      }
      if (!cancelled) setIsReplaying(false)
    }
    void replay()
    return () => { cancelled = true; setIsReplaying(false) }
  }, [runEnvelope])

  function handleLoadExample(example: ExampleItem) {
    setSelectedExampleId(example.id)
    setMode(example.mode)
    setRunEnvelope(null)
    setClientError(null)
    setRequestError(null)
    if (example.mode === 'json' && typeof example.value !== 'string') {
      setJsonInput(prettyJson(example.value))
    } else if (example.mode === 'text' && typeof example.value === 'string') {
      setTextInput(example.value)
    }
  }

  function handleFormatJson() {
    try {
      setJsonInput(prettyJson(JSON.parse(jsonInput) as Record<string, unknown>))
      setClientError(null)
    } catch {
      setClientError('Formatting failed — JSON is not valid yet.')
    }
  }

  function handleReset() {
    setRunEnvelope(null)
    setActiveStageIndex(0)
    setClientError(null)
    setRequestError(null)
    setSelectedArtifact(DEFAULT_ARTIFACT_TAB)
  }

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
          return
        }
        payload = parsed as Record<string, unknown>
      } else {
        const trimmed = textInput.trim()
        if (!trimmed) { setClientError('Enter a natural-language scenario before running.'); return }
        payload = trimmed
      }
      const response = await runScenario(payload, mode, evaluationVariant)
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

  return (
    <div className="page-workspace">
      {/* ── LEFT: INPUT ── */}
      <aside className="input-col">
        <div className="panel-header">
          <span className="panel-tag">01</span>
          <span className="panel-title">Scenario Input</span>
        </div>

        <div className="seg-control">
          <button type="button" className={`seg-btn ${mode === 'json' ? 'active' : ''}`} onClick={() => setMode('json')}>
            <span className="seg-icon">{ }</span> Structured JSON
          </button>
          <button type="button" className={`seg-btn ${mode === 'text' ? 'active' : ''}`} onClick={() => setMode('text')}>
            <span className="seg-icon">¶</span> Natural Language
          </button>
        </div>

        {examples.length > 0 && (
          <div className="examples-block">
            <label htmlFor="studio-example-select" className="block-label">Load Example</label>
            <div className="select-wrap">
              <select
                id="studio-example-select"
                className="example-select"
                value={selectedExampleId ?? ''}
                onChange={(e) => {
                  const found = examples.find((ex) => ex.id === e.target.value)
                  if (found) handleLoadExample(found)
                }}
              >
                <option value="" disabled>Select a scenario…</option>
                {examples.map((ex) => (
                  <option key={ex.id} value={ex.id}>
                    [{ex.mode.toUpperCase()}] {ex.label}{ex.subdivision_label ? ` · ${ex.subdivision_label}` : ''}
                  </option>
                ))}
              </select>
              <span className="select-chevron">▾</span>
            </div>
            {selectedExample && (
              <div className="catalog-meta">
                <span className="catalog-chip">{selectedExample.mode.toUpperCase()}</span>
                {selectedExample.subdivision_label && <span className="catalog-chip accent">{selectedExample.subdivision_label}</span>}
                {selectedExample.expected_framework && <span className="catalog-chip">Expected {selectedExample.expected_framework}</span>}
              </div>
            )}

            <label htmlFor="studio-variant-select" className="block-label evaluation-variant-label">Pipeline Variant</label>
            <div className="select-wrap">
              <select
                id="studio-variant-select"
                className="example-select"
                value={evaluationVariant}
                onChange={(e) => setEvaluationVariant(e.target.value as EvaluationVariant)}
              >
                <option value="full_system">Full System</option>
                <option value="no_rag">No RAG</option>
                <option value="no_math">No Math Layer</option>
                <option value="no_rag_no_math">No RAG + No Math</option>
              </select>
              <span className="select-chevron">▾</span>
            </div>
          </div>
        )}

        <div className="editor-wrap">
          <div className="editor-bar">
            <span className="editor-lang">{mode === 'json' ? 'application/json' : 'text/plain'}</span>
            <div className="editor-actions">
              {mode === 'json' && <button type="button" className="ghost-btn" onClick={handleFormatJson}>Format</button>}
              <button type="button" className="ghost-btn" onClick={handleReset}>Reset</button>
            </div>
          </div>
          {mode === 'json' ? (
            <textarea aria-label="Structured JSON editor" className="code-editor" value={jsonInput} onChange={(e) => setJsonInput(e.target.value)} />
          ) : (
            <textarea aria-label="Natural language scenario editor" className="code-editor natural" value={textInput} onChange={(e) => setTextInput(e.target.value)} />
          )}
        </div>

        {clientError && <div className="inline-error"><span>✕</span> {clientError}</div>}
        {requestError && <div className="inline-error"><span>✕</span> {requestError}</div>}

        <div className="run-actions">
          <button
            type="button"
            className={`run-btn ${isSubmitting ? 'loading' : ''}`}
            onClick={() => { void handleRun() }}
            disabled={isSubmitting}
          >
            {isSubmitting ? <><span className="run-spinner" /> Processing…</> : <><span className="run-icon">▶</span> Execute Pipeline</>}
          </button>
        </div>
      </aside>

      {/* ── RIGHT: OUTPUT ── */}
      <div className="output-col">
        {hasSuccessResult && (
          <div className="metrics-row">
            <div className="metric-card accent">
              <span className="metric-label">Deterministic Best</span>
              <strong className="metric-value">{runEnvelope.payload.summary.deterministic_best_action ?? 'Skipped'}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Dominant Framework</span>
              <strong className="metric-value">{runEnvelope.payload.summary.dominant_framework ?? '—'}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Input Mode</span>
              <strong className="metric-value">{runEnvelope.payload.summary.input_mode}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Variant</span>
              <strong className="metric-value">{runEnvelope.payload.summary.variant ?? 'full_system'}</strong>
            </div>
          </div>
        )}

        {hasSuccessResult && frameworkRationale && (
          <div className="rationale-banner">
            <span className="rationale-label">Rationale</span>
            <p>{frameworkRationale}</p>
          </div>
        )}

        {hasSuccessResult && !runEnvelope.payload.summary.rag_runtime_available && (
          <div className="warn-bar">⚠ RAG degraded: {runEnvelope.payload.summary.rag_runtime_error}</div>
        )}
        {hasSuccessResult && runEnvelope.payload.summary.math_runtime_available === false && (
          <div className="warn-bar">Math layer skipped for {runEnvelope.payload.summary.variant ?? 'ablation'}.</div>
        )}
        {hasSuccessResult && !runEnvelope.payload.summary.reasoning_runtime_available && (
          <div className="warn-bar">⚠ Reasoning degraded: {runEnvelope.payload.summary.reasoning_runtime_error}</div>
        )}

        {/* Pipeline Replay */}
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
            <div className="proc-banner">
              <div className="proc-bar"><div className="proc-fill" /></div>
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
              stages={stages}
              activeIndex={activeStageIndex}
              isReplaying={isReplaying}
              onSelectStage={(i) => setActiveStageIndex(i)}
            />
            <StageInspector currentStage={currentStage} previousStage={previousStage} isReplaying={isReplaying} />
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

        {!hasSuccessResult && !isSubmitting && (
          <div className="empty-page-state">
            <p>Configure a scenario on the left and click <strong>Execute Pipeline</strong> to run it.</p>
          </div>
        )}
      </div>
    </div>
  )
}
