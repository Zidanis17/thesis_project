import { startTransition, useEffect, useState } from 'react'
import {
  fetchAblatableFields,
  runFieldAblationCompare,
  runScenarioBank,
  runSubdivision,
} from '../api'
import { SubdivisionMetricsPanel } from '../components/SubdivisionMetricsPanel'
import type {
  AblatableField,
  EvaluationRunResponse,
  EvaluationVariant,
  ExampleItem,
  FieldAblationCompareResponse,
  FieldAblationResult,
  InputEditorMode,
  ScenarioSubdivision,
} from '../types'

function prettyJson(value: Record<string, unknown>) {
  return JSON.stringify(value, null, 2)
}

function formatMs(ms: number) {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`
}

function pctBar(value: number) {
  return Math.max(0, Math.min(100, value))
}

type AblationTab = 'pipeline' | 'field'

const FIELD_GROUPS: Record<string, string> = {
  ego_vehicle: 'Ego Vehicle',
  obstacles: 'Obstacles',
  sensor: 'Sensor Confidence',
  meta: 'Scenario Flags',
}

/* ── Preset ablation bundles ── */
const ABLATION_PRESETS: Array<{ label: string; fields: string[] }> = [
  { label: 'No Weight Data', fields: ['ego_vehicle.mass_kg', 'obstacles.mass_kg'] },
  { label: 'No Kinematics', fields: ['ego_vehicle.speed_kmh', 'ego_vehicle.acceleration_ms2', 'ego_vehicle.braking_distance_m'] },
  { label: 'No Sensor Data', fields: ['sensor_confidence.lidar', 'sensor_confidence.camera', 'sensor_confidence.radar', 'sensor_confidence.overall_scene_confidence'] },
  { label: 'No Responsibility Data', fields: ['obstacles.responsible_for_risk', 'ego_vehicle.passenger_at_risk'] },
  { label: 'No Timing Data', fields: ['obstacles.time_to_impact_s'] },
  { label: 'Minimal Input', fields: ['ego_vehicle.mass_kg', 'obstacles.mass_kg', 'ego_vehicle.speed_kmh', 'ego_vehicle.acceleration_ms2', 'ego_vehicle.braking_distance_m', 'sensor_confidence.lidar', 'sensor_confidence.camera', 'sensor_confidence.radar', 'sensor_confidence.overall_scene_confidence'] },
]

/* ── Field Ablation result row ── */
function FieldResultRow({ result, isBaseline }: { result: FieldAblationResult; isBaseline: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const statusOk = result.status === 'success'

  return (
    <div className={`ablation-result-row ${isBaseline ? 'is-baseline' : ''} ${!statusOk ? 'is-error' : ''}`}>
      <div className="ablation-result-head" onClick={() => setExpanded((v) => !v)} style={{ cursor: 'pointer' }}>
        <div className="ablation-result-label">
          {isBaseline && <span className="ablation-badge baseline">BASELINE</span>}
          {!isBaseline && result.fields_removed.length > 0 && (
            <span className="ablation-badge removed">−{result.fields_removed.length} fields</span>
          )}
          <strong>{result.ablation_label}</strong>
        </div>
        <div className="ablation-result-meta">
          {statusOk ? (
            <>
              <span className="ablation-fw">{result.dominant_framework ?? '—'}</span>
              <span className="ablation-action">{result.deterministic_best_action ?? '—'}</span>
              {typeof result.confidence === 'number' && (
                <span className="ablation-conf">{(result.confidence * 100).toFixed(0)}% conf</span>
              )}
              <span className="ablation-dur">{formatMs(result.duration_ms)}</span>
            </>
          ) : (
            <span className="ablation-error-code">{result.error_code ?? 'error'}</span>
          )}
          <span className="ablation-toggle">{expanded ? '▲' : '▼'}</span>
        </div>
      </div>

      {expanded && (
        <div className="ablation-result-body">
          {!isBaseline && result.fields_removed.length > 0 && (
            <div className="ablation-removed-fields">
              <span className="ablation-removed-label">Fields removed:</span>
              {result.fields_removed.map((f) => <span key={f} className="ablation-field-chip">{f}</span>)}
            </div>
          )}

          {statusOk && result.rationale && (
            <div className="ablation-rationale">
              <span className="ablation-rationale-label">LLM Rationale</span>
              <p>{result.rationale}</p>
            </div>
          )}

          {!statusOk && result.error_message && (
            <div className="ablation-rationale is-error-text">
              <span className="ablation-rationale-label">Error</span>
              <p>{result.error_message}</p>
            </div>
          )}

          {statusOk && Object.keys(result.weights).length > 0 && (
            <div className="ablation-weights">
              {Object.entries(result.weights).map(([k, v]) => (
                <div key={k} className="ablation-weight-row">
                  <span className="ablation-weight-label">{k}</span>
                  <div className="ablation-weight-track">
                    <div className="ablation-weight-fill" style={{ width: `${pctBar((v as number) * 100)}%` }} />
                  </div>
                  <span className="ablation-weight-val">{((v as number) * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Field Ablation compare table ── */
function FieldAblationCompareTable({ data }: { data: FieldAblationCompareResponse }) {
  const all = [data.baseline, ...data.ablations]
  return (
    <div className="ablation-compare-wrap">
      <div className="ablation-compare-header">
        <span className="ablation-compare-count">{all.length} run{all.length !== 1 ? 's' : ''} · 1 baseline + {data.ablations.length} ablation{data.ablations.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="ablation-compare-list">
        {all.map((result, i) => (
          <FieldResultRow key={i} result={result} isBaseline={i === 0} />
        ))}
      </div>
    </div>
  )
}

interface AblationPageProps {
  examples: ExampleItem[]
  subdivisions: ScenarioSubdivision[]
  initialJsonInput: string
  initialExampleId: string | null
}

export function AblationPage({ examples, subdivisions, initialJsonInput, initialExampleId }: AblationPageProps) {
  const [activeTab, setActiveTab] = useState<AblationTab>('pipeline')
  const [mode] = useState<InputEditorMode>('json')
  const [jsonInput, setJsonInput] = useState(initialJsonInput)
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(initialExampleId)
  const [selectedSubdivisionId, setSelectedSubdivisionId] = useState(subdivisions[0]?.id ?? '')
  const [evaluationVariant, setEvaluationVariant] = useState<EvaluationVariant>('full_system')

  /* pipeline ablation state */
  const [evaluationResult, setEvaluationResult] = useState<EvaluationRunResponse | null>(null)
  const [isBatchSubmitting, setIsBatchSubmitting] = useState(false)
  const [isBankSubmitting, setIsBankSubmitting] = useState(false)
  const [batchError, setBatchError] = useState<string | null>(null)

  /* field ablation state */
  const [ablatableFields, setAblatableFields] = useState<AblatableField[]>([])
  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set())
  const [namedGroups, setNamedGroups] = useState<Array<{ label: string; fields: string[] }>>([])
  const [fieldAblationResult, setFieldAblationResult] = useState<FieldAblationCompareResponse | null>(null)
  const [isFieldAblating, setIsFieldAblating] = useState(false)
  const [fieldAblationError, setFieldAblationError] = useState<string | null>(null)

  useEffect(() => {
    fetchAblatableFields().then(setAblatableFields).catch(() => {})
  }, [])

  useEffect(() => { setJsonInput(initialJsonInput) }, [initialJsonInput])
  useEffect(() => { setSelectedExampleId(initialExampleId) }, [initialExampleId])

  const selectedSubdivision = subdivisions.find((s) => s.id === selectedSubdivisionId) ?? null
  const fieldsByGroup = ablatableFields.reduce<Record<string, AblatableField[]>>((acc, f) => {
    if (!acc[f.group]) acc[f.group] = []
    acc[f.group].push(f)
    return acc
  }, {})

  function handleLoadExample(example: ExampleItem) {
    setSelectedExampleId(example.id)
    if (example.mode === 'json' && typeof example.value !== 'string') {
      setJsonInput(prettyJson(example.value))
    }
    if (example.subdivision_id) setSelectedSubdivisionId(example.subdivision_id)
  }

  function toggleField(path: string) {
    setSelectedFields((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  function applyPreset(preset: { label: string; fields: string[] }) {
    setSelectedFields(new Set(preset.fields))
  }

  function addGroup() {
    if (selectedFields.size === 0) return
    setNamedGroups((prev) => [...prev, { label: `Group ${prev.length + 1} (${selectedFields.size} fields)`, fields: Array.from(selectedFields) }])
    setSelectedFields(new Set())
  }

  function removeGroup(index: number) {
    setNamedGroups((prev) => prev.filter((_, i) => i !== index))
  }

  async function handleRunFieldAblation() {
    setFieldAblationError(null)
    setIsFieldAblating(true)
    try {
      const parsed = JSON.parse(jsonInput) as Record<string, unknown>
      const groups = namedGroups.length > 0
        ? namedGroups.map((g) => g.fields)
        : selectedFields.size > 0
        ? [Array.from(selectedFields)]
        : ABLATION_PRESETS.map((p) => p.fields)

      const result = await runFieldAblationCompare(parsed, mode, groups)
      startTransition(() => { setFieldAblationResult(result) })
    } catch (error) {
      setFieldAblationError(error instanceof Error ? error.message : 'Field ablation request failed.')
    } finally {
      setIsFieldAblating(false)
    }
  }

  async function handleRunSubdivision() {
    if (!selectedSubdivisionId) { setBatchError('Select a subdivision first.'); return }
    setBatchError(null)
    setIsBatchSubmitting(true)
    try {
      const result = await runSubdivision(selectedSubdivisionId, evaluationVariant)
      startTransition(() => { setEvaluationResult(result) })
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : 'Subdivision request failed.')
    } finally {
      setIsBatchSubmitting(false)
    }
  }

  async function handleRunBank() {
    setBatchError(null)
    setIsBankSubmitting(true)
    try {
      const result = await runScenarioBank(evaluationVariant)
      startTransition(() => { setEvaluationResult(result) })
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : 'Scenario bank request failed.')
    } finally {
      setIsBankSubmitting(false)
    }
  }

  const isBusy = isBatchSubmitting || isBankSubmitting || isFieldAblating

  return (
    <div className="page-workspace">
      {/* ── LEFT: ABLATION CONTROLS ── */}
      <aside className="input-col">
        <div className="panel-header">
          <span className="panel-tag">ABL</span>
          <span className="panel-title">Ablation Setup</span>
        </div>

        {/* Tab toggle */}
        <div className="seg-control">
          <button type="button" className={`seg-btn ${activeTab === 'pipeline' ? 'active' : ''}`} onClick={() => setActiveTab('pipeline')}>
            <span className="seg-icon">◈</span> Pipeline
          </button>
          <button type="button" className={`seg-btn ${activeTab === 'field' ? 'active' : ''}`} onClick={() => setActiveTab('field')}>
            <span className="seg-icon">⊟</span> Field
          </button>
        </div>

        {/* Scenario input (shared) */}
        <div className="examples-block">
          <label htmlFor="abl-example-select" className="block-label">Scenario</label>
          <div className="select-wrap">
            <select
              id="abl-example-select"
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
                  {ex.label}{ex.subdivision_label ? ` · ${ex.subdivision_label}` : ''}
                </option>
              ))}
            </select>
            <span className="select-chevron">▾</span>
          </div>
        </div>

        <div className="editor-wrap" style={{ maxHeight: activeTab === 'field' ? '180px' : '200px' }}>
          <div className="editor-bar">
            <span className="editor-lang">application/json</span>
          </div>
          <textarea
            aria-label="Scenario JSON editor"
            className="code-editor"
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
          />
        </div>

        {/* ── PIPELINE TAB ── */}
        {activeTab === 'pipeline' && (
          <div className="ablation-controls-section">
            <div className="block-label" style={{ padding: '10px 16px 4px' }}>Subdivision</div>
            <div className="examples-block" style={{ borderTop: 'none', paddingTop: 0 }}>
              <div className="select-wrap">
                <select
                  className="example-select"
                  value={selectedSubdivisionId}
                  onChange={(e) => { setSelectedSubdivisionId(e.target.value); setEvaluationResult(null) }}
                >
                  <option value="" disabled>Select a subdivision…</option>
                  {subdivisions.map((s) => (
                    <option key={s.id} value={s.id}>{s.label} ({s.scenario_count})</option>
                  ))}
                </select>
                <span className="select-chevron">▾</span>
              </div>
              {selectedSubdivision && (
                <div className="catalog-meta" style={{ marginTop: 8 }}>
                  <span className="catalog-chip accent">{selectedSubdivision.label}</span>
                  <span className="catalog-chip">{selectedSubdivision.scenario_count} scenarios</span>
                  {selectedSubdivision.expected_framework && <span className="catalog-chip">Expected {selectedSubdivision.expected_framework}</span>}
                </div>
              )}

              <label className="block-label evaluation-variant-label">Pipeline Variant</label>
              <div className="select-wrap">
                <select
                  className="example-select"
                  value={evaluationVariant}
                  onChange={(e) => { setEvaluationVariant(e.target.value as EvaluationVariant); setEvaluationResult(null) }}
                >
                  <option value="full_system">Full System</option>
                  <option value="no_rag">No RAG</option>
                  <option value="no_math">No Math Layer</option>
                  <option value="no_rag_no_math">No RAG + No Math</option>
                </select>
                <span className="select-chevron">▾</span>
              </div>
            </div>

            {batchError && <div className="inline-error"><span>✕</span> {batchError}</div>}

            <div className="run-actions">
              <button
                type="button"
                className={`run-btn secondary ${isBatchSubmitting ? 'loading' : ''}`}
                onClick={() => { void handleRunSubdivision() }}
                disabled={isBusy || !selectedSubdivisionId}
              >
                {isBatchSubmitting ? <><span className="run-spinner" /> Processing…</> : <><span className="run-icon">▦</span> Run Subdivision</>}
              </button>
              <button
                type="button"
                className={`run-btn secondary ${isBankSubmitting ? 'loading' : ''}`}
                onClick={() => { void handleRunBank() }}
                disabled={isBusy}
              >
                {isBankSubmitting ? <><span className="run-spinner" /> Processing…</> : <><span className="run-icon">▦</span> Run Full Bank</>}
              </button>
            </div>
          </div>
        )}

        {/* ── FIELD ABLATION TAB ── */}
        {activeTab === 'field' && (
          <div className="ablation-controls-section">
            <div className="ablation-presets-row">
              <span className="block-label" style={{ padding: '10px 16px 6px', display: 'block' }}>Quick Presets</span>
              <div className="ablation-preset-chips">
                {ABLATION_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    className="ablation-preset-chip"
                    onClick={() => applyPreset(preset)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="ablation-field-groups">
              {Object.entries(fieldsByGroup).map(([group, fields]) => (
                <div key={group} className="ablation-field-group">
                  <span className="ablation-group-label">{FIELD_GROUPS[group] ?? group}</span>
                  {fields.map((f) => (
                    <label key={f.path} className="ablation-field-check">
                      <input
                        type="checkbox"
                        checked={selectedFields.has(f.path)}
                        onChange={() => toggleField(f.path)}
                      />
                      <span>{f.label}</span>
                      <span className="ablation-field-path">{f.path}</span>
                    </label>
                  ))}
                </div>
              ))}
            </div>

            {selectedFields.size > 0 && (
              <div className="ablation-group-builder">
                <div className="ablation-selected-count">{selectedFields.size} field{selectedFields.size !== 1 ? 's' : ''} selected</div>
                <button type="button" className="ghost-btn" onClick={addGroup}>+ Add as Named Group</button>
              </div>
            )}

            {namedGroups.length > 0 && (
              <div className="ablation-named-groups">
                <span className="block-label" style={{ padding: '8px 16px 4px', display: 'block' }}>Ablation Groups</span>
                {namedGroups.map((g, i) => (
                  <div key={i} className="ablation-named-group-row">
                    <span className="ablation-named-group-label">{g.label}</span>
                    <span className="ablation-named-group-fields">{g.fields.length} fields</span>
                    <button type="button" className="ghost-btn" onClick={() => removeGroup(i)}>✕</button>
                  </div>
                ))}
              </div>
            )}

            {fieldAblationError && <div className="inline-error"><span>✕</span> {fieldAblationError}</div>}

            <div className="run-actions">
              <button
                type="button"
                className={`run-btn ${isFieldAblating ? 'loading' : ''}`}
                onClick={() => { void handleRunFieldAblation() }}
                disabled={isBusy}
              >
                {isFieldAblating
                  ? <><span className="run-spinner" /> Running Ablations…</>
                  : <><span className="run-icon">⊟</span> Run Field Ablation</>}
              </button>
              <p className="ablation-run-hint">
                {namedGroups.length > 0
                  ? `Will compare baseline against ${namedGroups.length} named group${namedGroups.length !== 1 ? 's' : ''}.`
                  : selectedFields.size > 0
                  ? `Will compare baseline vs. removing ${selectedFields.size} selected field${selectedFields.size !== 1 ? 's' : ''}.`
                  : `No groups defined — will run all ${ABLATION_PRESETS.length} presets.`}
              </p>
            </div>
          </div>
        )}
      </aside>

      {/* ── RIGHT: RESULTS ── */}
      <div className="output-col">
        {activeTab === 'pipeline' && (
          <div className="subdivision-panel" style={{ flex: 1 }}>
            <div className="panel-header">
              <span className="panel-tag">02</span>
              <span className="panel-title">Pipeline Ablation Results</span>
              <span className="panel-status">
                {isBatchSubmitting || isBankSubmitting
                  ? <><span className="spin-dot" /> Processing</>
                  : evaluationResult
                  ? `${evaluationResult.scope === 'full_bank' ? 'Full bank' : evaluationResult.subdivision_label ?? 'Subdivision'} · ${evaluationResult.summary.accuracy_pct}% accuracy`
                  : 'Idle — run a subdivision or full bank'}
              </span>
            </div>
            <SubdivisionMetricsPanel
              result={evaluationResult}
              isLoading={isBatchSubmitting || isBankSubmitting}
              error={batchError}
            />
          </div>
        )}

        {activeTab === 'field' && (
          <div className="subdivision-panel" style={{ flex: 1 }}>
            <div className="panel-header">
              <span className="panel-tag">02</span>
              <span className="panel-title">Field Ablation Results</span>
              <span className="panel-status">
                {isFieldAblating
                  ? <><span className="spin-dot" /> Running</>
                  : fieldAblationResult
                  ? `${fieldAblationResult.ablations.length + 1} runs compared`
                  : 'Idle — configure fields and run'}
              </span>
            </div>

            {isFieldAblating && (
              <div className="subdivision-panel-body">
                <div className="proc-banner">
                  <div className="proc-bar"><div className="proc-fill" /></div>
                  <p>Running baseline + field ablation variants through the full pipeline…</p>
                </div>
              </div>
            )}

            {!isFieldAblating && fieldAblationError && (
              <div className="subdivision-panel-body">
                <div className="error-block">
                  <strong className="error-code">field_ablation_error</strong>
                  <p>{fieldAblationError}</p>
                </div>
              </div>
            )}

            {!isFieldAblating && !fieldAblationResult && !fieldAblationError && (
              <div className="subdivision-panel-body">
                <div className="empty-state">
                  Select fields to remove on the left, then click <strong>Run Field Ablation</strong> to compare how missing input data impacts the reasoning output.
                  <br /><br />
                  You can use presets (e.g. "No Weight Data") or select individual fields. Multiple named groups let you compare several ablation hypotheses at once.
                </div>
              </div>
            )}

            {!isFieldAblating && fieldAblationResult && (
              <div className="subdivision-panel-body">
                <FieldAblationCompareTable data={fieldAblationResult} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
