import { useEffect, useState } from 'react'
import {
  fetchEvaluationHistory,
  fetchScenarioRunHistory,
} from '../api'
import type {
  EvaluationRunHistoryResponse,
  EvaluationRunRecord,
  ScenarioRunHistoryResponse,
  ScenarioRunRecord,
} from '../types'

type HistoryTab = 'runs' | 'evaluations'

/* ─── Export helpers ─── */
function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

function csvEscape(value: unknown): string {
  const text = Array.isArray(value) ? value.join('|') : value == null ? '' : String(value)
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

function toCsv<T extends object>(rows: T[], keys: (keyof T)[]): string {
  return [
    keys.join(','),
    ...rows.map((row) => keys.map((k) => csvEscape(row[k])).join(',')),
  ].join('\n')
}

function exportRunsCsv(runs: ScenarioRunRecord[]) {
  const keys: (keyof ScenarioRunRecord)[] = [
    'id', 'created_at', 'status', 'input_mode_hint', 'resolved_input_mode',
    'submitted_kind', 'model_name', 'deterministic_best_action', 'dominant_framework',
    'rag_runtime_available', 'reasoning_runtime_available', 'error_code', 'error_message',
    'replay_stage_count',
  ]
  downloadText('scenario-runs.csv', toCsv(runs, keys), 'text/csv;charset=utf-8')
}

function exportRunsJson(runs: ScenarioRunRecord[]) {
  downloadText('scenario-runs.json', JSON.stringify(runs, null, 2), 'application/json;charset=utf-8')
}

function exportEvaluationsCsv(runs: EvaluationRunRecord[]) {
  const keys: (keyof EvaluationRunRecord)[] = [
    'id', 'created_at', 'scope', 'subdivision_id', 'variant', 'model_name',
    'total_scenarios', 'completed_runs', 'failed_runs', 'overall_accuracy_pct',
  ]
  downloadText('evaluation-runs.csv', toCsv(runs, keys), 'text/csv;charset=utf-8')
}

function exportEvaluationsJson(runs: EvaluationRunRecord[]) {
  downloadText('evaluation-runs.json', JSON.stringify(runs, null, 2), 'application/json;charset=utf-8')
}

/* ─── Formatting ─── */
function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
  } catch {
    return iso
  }
}

function StatusChip({ ok, label }: { ok: boolean; label?: string }) {
  return (
    <span className={`metric-chip ${ok ? 'status-chip-ok' : 'status-chip-error'}`}>
      {label ?? (ok ? 'success' : 'error')}
    </span>
  )
}

/* ─── Runs Table ─── */
function RunsTable({ runs }: { runs: ScenarioRunRecord[] }) {
  if (!runs.length) return <div className="empty-state">No scenario runs stored yet.</div>
  return (
    <div className="subdivision-table-wrap">
      <table className="subdivision-table history-runs-table">
        <thead>
          <tr>
            <th scope="col">Run ID</th>
            <th scope="col">Time</th>
            <th scope="col">Status</th>
            <th scope="col">Mode</th>
            <th scope="col">Framework</th>
            <th scope="col">Best Action</th>
            <th scope="col">RAG</th>
            <th scope="col">Reasoning</th>
            <th scope="col">Stages</th>
            <th scope="col">Error</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>
                <span className="history-run-id">{run.id.slice(0, 8)}…</span>
              </td>
              <td>{formatDate(run.created_at)}</td>
              <td><StatusChip ok={run.status === 'success'} /></td>
              <td><span className="metric-chip">{run.resolved_input_mode ?? run.input_mode_hint}</span></td>
              <td>{run.dominant_framework ?? '—'}</td>
              <td>{run.deterministic_best_action ?? '—'}</td>
              <td><StatusChip ok={run.rag_runtime_available} label={run.rag_runtime_available ? 'ok' : 'off'} /></td>
              <td><StatusChip ok={run.reasoning_runtime_available} label={run.reasoning_runtime_available ? 'ok' : 'off'} /></td>
              <td>{run.replay_stage_count}</td>
              <td>
                {run.error_code
                  ? <span className="metric-chip status-chip-error">{run.error_code}</span>
                  : <span className="history-ok">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Evaluations Table ─── */
function EvaluationsTable({ runs }: { runs: EvaluationRunRecord[] }) {
  if (!runs.length) return <div className="empty-state">No evaluation runs stored yet.</div>
  return (
    <div className="subdivision-table-wrap">
      <table className="subdivision-table history-eval-table">
        <thead>
          <tr>
            <th scope="col">Eval ID</th>
            <th scope="col">Time</th>
            <th scope="col">Scope</th>
            <th scope="col">Subdivision</th>
            <th scope="col">Variant</th>
            <th scope="col">Model</th>
            <th scope="col">Scenarios</th>
            <th scope="col">Completed</th>
            <th scope="col">Failed</th>
            <th scope="col">Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td><span className="history-run-id">{run.id.slice(0, 8)}…</span></td>
              <td>{formatDate(run.created_at)}</td>
              <td><span className="metric-chip">{run.scope}</span></td>
              <td>{run.subdivision_id ?? '(all)'}</td>
              <td><span className="metric-chip">{run.variant ?? '—'}</span></td>
              <td>{run.model_name ?? '—'}</td>
              <td>{run.total_scenarios}</td>
              <td>{run.completed_runs}</td>
              <td>
                {run.failed_runs > 0
                  ? <span className="metric-chip status-chip-error">{run.failed_runs}</span>
                  : <span className="history-ok">0</span>}
              </td>
              <td>
                <span className={`metric-chip ${run.overall_accuracy_pct >= 70 ? 'status-chip-ok' : 'status-chip-error'}`}>
                  {run.overall_accuracy_pct.toFixed(1)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function HistoryPage() {
  const [tab, setTab] = useState<HistoryTab>('runs')
  const [runsData, setRunsData] = useState<ScenarioRunHistoryResponse | null>(null)
  const [evalsData, setEvalsData] = useState<EvaluationRunHistoryResponse | null>(null)
  const [runsLoading, setRunsLoading] = useState(false)
  const [evalsLoading, setEvalsLoading] = useState(false)
  const [runsError, setRunsError] = useState<string | null>(null)
  const [evalsError, setEvalsError] = useState<string | null>(null)

  async function loadRuns() {
    setRunsLoading(true)
    setRunsError(null)
    try {
      const data = await fetchScenarioRunHistory(100)
      setRunsData(data)
    } catch (err) {
      setRunsError(err instanceof Error ? err.message : 'Failed to load run history.')
    } finally {
      setRunsLoading(false)
    }
  }

  async function loadEvals() {
    setEvalsLoading(true)
    setEvalsError(null)
    try {
      const data = await fetchEvaluationHistory(100)
      setEvalsData(data)
    } catch (err) {
      setEvalsError(err instanceof Error ? err.message : 'Failed to load evaluation history.')
    } finally {
      setEvalsLoading(false)
    }
  }

  useEffect(() => { void loadRuns() }, [])
  useEffect(() => { if (tab === 'evaluations' && !evalsData) void loadEvals() }, [tab])

  const runs = runsData?.runs ?? []
  const evals = evalsData?.runs ?? []

  return (
    <div className="history-page">
      {/* Header */}
      <div className="history-page-header">
        <div>
          <h1 className="history-page-title">Run History</h1>
          <p className="history-page-subtitle">All stored scenario runs and evaluation results — export as CSV or JSON for analysis.</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="history-tab-bar">
        <button
          type="button"
          className={`history-tab ${tab === 'runs' ? 'active' : ''}`}
          onClick={() => setTab('runs')}
        >
          Scenario Runs
          {runsData && <span className="history-tab-count">{runsData.total_runs}</span>}
        </button>
        <button
          type="button"
          className={`history-tab ${tab === 'evaluations' ? 'active' : ''}`}
          onClick={() => setTab('evaluations')}
        >
          Evaluation Runs
          {evalsData && <span className="history-tab-count">{evalsData.total_runs}</span>}
        </button>
      </div>

      {/* ── Scenario Runs tab ── */}
      {tab === 'runs' && (
        <div className="history-tab-body">
          <div className="history-toolbar">
            {runsData && (
              <div className="history-stats">
                <span className="metric-chip">{runsData.total_runs} total</span>
                <span className="metric-chip status-chip-ok">{runsData.success_runs} success</span>
                <span className={`metric-chip ${runsData.failed_runs > 0 ? 'status-chip-error' : ''}`}>{runsData.failed_runs} failed</span>
              </div>
            )}
            <div className="history-export-group">
              <span className="history-export-label">Export:</span>
              <button
                type="button"
                className="ghost-btn export-btn"
                disabled={runs.length === 0}
                onClick={() => exportRunsCsv(runs)}
              >
                CSV
              </button>
              <button
                type="button"
                className="ghost-btn export-btn"
                disabled={runs.length === 0}
                onClick={() => exportRunsJson(runs)}
              >
                JSON
              </button>
              <button type="button" className="ghost-btn" onClick={() => { void loadRuns() }}>
                ↻ Refresh
              </button>
            </div>
          </div>

          {runsLoading && (
            <div className="proc-banner" style={{ margin: '16px 24px' }}>
              <div className="proc-bar"><div className="proc-fill" /></div>
              <p>Loading run history…</p>
            </div>
          )}
          {runsError && (
            <div className="error-block" style={{ margin: '16px 24px' }}>
              <strong className="error-code">history_load_error</strong>
              <p>{runsError}</p>
            </div>
          )}
          {!runsLoading && !runsError && <RunsTable runs={runs} />}
        </div>
      )}

      {/* ── Evaluations tab ── */}
      {tab === 'evaluations' && (
        <div className="history-tab-body">
          <div className="history-toolbar">
            {evalsData && (
              <div className="history-stats">
                <span className="metric-chip">{evalsData.total_runs} total</span>
                <span className="metric-chip">{evalsData.full_bank_runs} full-bank</span>
                <span className="metric-chip">{evalsData.subdivision_runs} subdivision</span>
              </div>
            )}
            <div className="history-export-group">
              <span className="history-export-label">Export:</span>
              <button
                type="button"
                className="ghost-btn export-btn"
                disabled={evals.length === 0}
                onClick={() => exportEvaluationsCsv(evals)}
              >
                CSV
              </button>
              <button
                type="button"
                className="ghost-btn export-btn"
                disabled={evals.length === 0}
                onClick={() => exportEvaluationsJson(evals)}
              >
                JSON
              </button>
              <button type="button" className="ghost-btn" onClick={() => { void loadEvals() }}>
                ↻ Refresh
              </button>
            </div>
          </div>

          {evalsLoading && (
            <div className="proc-banner" style={{ margin: '16px 24px' }}>
              <div className="proc-bar"><div className="proc-fill" /></div>
              <p>Loading evaluation history…</p>
            </div>
          )}
          {evalsError && (
            <div className="error-block" style={{ margin: '16px 24px' }}>
              <strong className="error-code">evaluation_load_error</strong>
              <p>{evalsError}</p>
            </div>
          )}
          {!evalsLoading && !evalsError && <EvaluationsTable runs={evals} />}
        </div>
      )}
    </div>
  )
}
