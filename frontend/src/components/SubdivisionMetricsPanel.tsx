import type { CSSProperties } from 'react'
import type {
  ConfusionMatrix,
  EvaluationRunResponse,
  EvaluationScenarioResult,
  FrameworkDistributionMetric,
  PerFrameworkAccuracyMetric,
} from '../types'

interface SubdivisionMetricsPanelProps {
  result: EvaluationRunResponse | null
  isLoading: boolean
  error: string | null
}

const FRAMEWORK_COLORS: Record<string, string> = {
  'EF-01': '#f0a500',
  'EF-02': '#00c8ff',
  'EF-03': '#00e676',
  'EF-04': '#ff7f50',
  'EF-05': '#4dd0e1',
  'EF-06': '#8bc34a',
  unresolved: '#4a5e80',
}

const SCENARIO_CSV_COLUMNS = [
  'scenario_id',
  'scenario_label',
  'subdivision_id',
  'expected_framework',
  'dominant_framework',
  'correct_prediction',
  'confidence',
  'deterministic_best_action',
  'status',
  'duration_ms',
  'rag_runtime_available',
  'reasoning_runtime_available',
  'retrieved_framework_ids',
  'expected_framework_retrieved',
  'reasoning_contract_valid',
  'risk_matrix_preserved',
  'weights_sum_to_one',
  'error_code',
  'error_message',
] as const

function getFrameworkColor(frameworkId: string | null): string {
  if (!frameworkId) return FRAMEWORK_COLORS.unresolved
  return FRAMEWORK_COLORS[frameworkId] ?? FRAMEWORK_COLORS.unresolved
}

function buildBarStyle(metric: FrameworkDistributionMetric): CSSProperties {
  const accent = getFrameworkColor(metric.framework_id)
  return {
    width: `${Math.max(metric.percentage, metric.count > 0 ? 3 : 0)}%`,
    background: `linear-gradient(90deg, ${accent}, ${accent}cc)`,
    boxShadow: `0 0 12px ${accent}33`,
  }
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${durationMs} ms`
  return `${(durationMs / 1000).toFixed(1)} s`
}

function formatPct(value: number | null | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  return `${value.toFixed(1)}%`
}

function formatNumber(value: number | null | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  return value.toFixed(3).replace(/\.?0+$/, '')
}

function boolLabel(value: boolean | null | undefined): string {
  if (value === true) return 'yes'
  if (value === false) return 'no'
  return 'N/A'
}

function statusChipClass(value: boolean | null | undefined): string {
  if (value === true) return 'status-chip-ok'
  if (value === false) return 'status-chip-error'
  return ''
}

function csvEscape(value: unknown): string {
  const text = Array.isArray(value) ? value.join('|') : value == null ? '' : String(value)
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

function toCsv(rows: object[], columns: readonly string[]): string {
  return [
    columns.join(','),
    ...rows.map((row) => {
      const record = row as Record<string, unknown>
      return columns.map((column) => csvEscape(record[column])).join(',')
    }),
  ].join('\n')
}

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}

function resultBaseName(result: EvaluationRunResponse): string {
  const scope = result.scope === 'full_bank' ? 'full-bank' : result.subdivision_id ?? 'subdivision'
  return `evaluation-${scope}-${result.evaluation_id?.slice(0, 8) ?? 'latest'}`
}

function exportEvaluationJson(result: EvaluationRunResponse) {
  downloadText(
    `${resultBaseName(result)}.json`,
    JSON.stringify(result, null, 2),
    'application/json;charset=utf-8',
  )
}

function exportScenarioCsv(result: EvaluationRunResponse) {
  const rows = result.scenario_results.map((scenario) => ({
    ...scenario,
    retrieved_framework_ids: scenario.retrieved_framework_ids.join('|'),
  }))
  downloadText(
    `${resultBaseName(result)}-scenario-results.csv`,
    toCsv(rows, SCENARIO_CSV_COLUMNS),
    'text/csv;charset=utf-8',
  )
}

function exportConfusionMatrixCsv(result: EvaluationRunResponse) {
  const labels = result.confusion_matrix.labels
  const rows = result.confusion_matrix.rows.map((row) => ({
    expected: row.expected,
    ...row.predictions,
  }))
  downloadText(
    `${resultBaseName(result)}-confusion-matrix.csv`,
    toCsv(rows, ['expected', ...labels]),
    'text/csv;charset=utf-8',
  )
}

function exportPerFrameworkCsv(result: EvaluationRunResponse) {
  const columns: Array<keyof PerFrameworkAccuracyMetric> = [
    'framework_id',
    'expected_count',
    'correct_count',
    'incorrect_count',
    'accuracy_pct',
  ]
  downloadText(
    `${resultBaseName(result)}-per-framework-accuracy.csv`,
    toCsv(result.per_framework_accuracy, columns),
    'text/csv;charset=utf-8',
  )
}

function renderFrameworkDistribution(metrics: FrameworkDistributionMetric[]) {
  return (
    <div className="framework-graph" data-testid="framework-graph">
      {metrics.map((metric) => (
        <div key={`${metric.framework_id ?? 'unresolved'}-${metric.framework_label}`} className="framework-graph-row">
          <div className="framework-graph-copy">
            <span className="framework-graph-id">{metric.framework_id ?? 'N/A'}</span>
            <span className="framework-graph-label">{metric.framework_label}</span>
          </div>
          <div className="framework-graph-track">
            <div className="framework-graph-fill" style={buildBarStyle(metric)} />
          </div>
          <div className="framework-graph-stat">
            {metric.percentage.toFixed(1)}% ({metric.count})
          </div>
        </div>
      ))}
    </div>
  )
}

function renderPerFrameworkAccuracy(metrics: PerFrameworkAccuracyMetric[]) {
  return (
    <div className="subdivision-table-wrap">
      <table className="subdivision-table">
        <thead>
          <tr>
            <th scope="col">Framework</th>
            <th scope="col">Expected</th>
            <th scope="col">Correct</th>
            <th scope="col">Incorrect</th>
            <th scope="col">Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric.framework_id}>
              <td>{metric.framework_id}</td>
              <td>{metric.expected_count}</td>
              <td>{metric.correct_count}</td>
              <td>{metric.incorrect_count}</td>
              <td>{formatPct(metric.accuracy_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function renderConfusionMatrix(matrix: ConfusionMatrix) {
  return (
    <div className="subdivision-table-wrap">
      <table className="subdivision-table confusion-table">
        <thead>
          <tr>
            <th scope="col">Expected</th>
            {matrix.labels.map((label) => (
              <th key={label} scope="col">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.expected}>
              <td>{row.expected}</td>
              {matrix.labels.map((label) => (
                <td key={`${row.expected}-${label}`}>{row.predictions[label] ?? 0}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function renderScenarioRows(rows: EvaluationScenarioResult[]) {
  return (
    <div className="subdivision-table-wrap">
      <table className="subdivision-table scenario-results-table">
        <thead>
          <tr>
            <th scope="col">Scenario</th>
            <th scope="col">Expected</th>
            <th scope="col">Predicted</th>
            <th scope="col">Correct</th>
            <th scope="col">Confidence</th>
            <th scope="col">Deterministic Best</th>
            <th scope="col">RAG Hit</th>
            <th scope="col">Contract</th>
            <th scope="col">Risk Matrix</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((scenario) => (
            <tr key={scenario.scenario_id}>
              <td>
                <div className="subdivision-scenario-cell">
                  <strong>{scenario.scenario_label}</strong>
                  <span>{scenario.scenario_id}</span>
                </div>
              </td>
              <td>{scenario.expected_framework ?? 'N/A'}</td>
              <td>{scenario.dominant_framework ?? 'Unresolved'}</td>
              <td>
                <span className={`metric-chip ${statusChipClass(scenario.correct_prediction)}`}>
                  {scenario.correct_prediction ? 'correct' : 'incorrect'}
                </span>
              </td>
              <td>{formatNumber(scenario.confidence)}</td>
              <td>{scenario.deterministic_best_action ?? 'N/A'}</td>
              <td>
                <span className={`metric-chip ${statusChipClass(scenario.expected_framework_retrieved)}`}>
                  {boolLabel(scenario.expected_framework_retrieved)}
                </span>
              </td>
              <td>
                <span className={`metric-chip ${statusChipClass(scenario.reasoning_contract_valid)}`}>
                  {boolLabel(scenario.reasoning_contract_valid)}
                </span>
              </td>
              <td>
                <span className={`metric-chip ${statusChipClass(scenario.risk_matrix_preserved)}`}>
                  {boolLabel(scenario.risk_matrix_preserved)}
                </span>
              </td>
              <td>
                <span className={`metric-chip ${scenario.status === 'error' ? 'status-chip-error' : 'status-chip-ok'}`}>
                  {scenario.status === 'error' ? scenario.error_code ?? 'error' : 'success'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function SubdivisionMetricsPanel({ result, isLoading, error }: SubdivisionMetricsPanelProps) {
  if (isLoading) {
    return (
      <div className="subdivision-panel-body">
        <div className="proc-banner" data-testid="subdivision-processing-banner">
          <div className="proc-bar">
            <div className="proc-fill" />
          </div>
          <p>Running the evaluation batch and aggregating accuracy, retrieval, and contract checks.</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="subdivision-panel-body">
        <div className="error-block">
          <strong className="error-code">evaluation_run_error</strong>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="subdivision-panel-body">
        <div className="empty-state">
          Run a subdivision or the full scenario bank to generate evaluation metrics for the thesis tables.
        </div>
      </div>
    )
  }

  const expectation = result.subdivision?.expectation
  const scopeLabel = result.scope === 'full_bank' ? 'Full Scenario Bank' : result.subdivision?.label ?? 'Subdivision'
  const expectedFramework = expectation?.expected_dominant_framework ?? result.summary.expected_framework ?? 'N/A'
  const perFrameworkAccuracy = result.per_framework_accuracy ?? []
  const confusionMatrix = result.confusion_matrix ?? { labels: [], rows: [] }

  return (
    <div className="subdivision-panel-body">
      <div className="subdivision-summary-row evaluation-summary-row">
        <div className="metric-card accent">
          <span className="metric-label">Scope / Subdivision</span>
          <strong className="metric-value">{scopeLabel}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Scenarios</span>
          <strong className="metric-value">{result.summary.scenario_count}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Accuracy</span>
          <strong className="metric-value">{formatPct(result.summary.accuracy_pct)}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Completed / Failed</span>
          <strong className="metric-value">{result.summary.completed_runs} / {result.summary.failed_runs}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Batch Duration</span>
          <strong className="metric-value">{formatDuration(result.summary.total_duration_ms)}</strong>
        </div>
      </div>

      <div className="subdivision-meta-strip">
        <span className="metric-chip">{formatPct(result.summary.rag_retrieval_hit_rate_pct)} RAG hit rate</span>
        <span className="metric-chip">{formatPct(result.summary.reasoning_contract_validity_rate_pct)} contract valid</span>
        <span className="metric-chip">{formatPct(result.summary.risk_matrix_preservation_rate_pct)} risk preserved</span>
        <span className="metric-chip">{formatPct(result.summary.weights_validity_rate_pct)} weights valid</span>
        <span className="metric-chip">{formatPct(result.summary.reasoning_runtime_ready_pct)} reasoning ready</span>
        <span className="metric-chip">{formatPct(result.summary.rag_runtime_ready_pct)} RAG ready</span>
        <span className="metric-chip">Expected: {expectedFramework}</span>
        <span className="metric-chip">{result.variant}</span>
      </div>

      <div className="evaluation-export-row">
        <button type="button" className="ghost-btn" onClick={() => exportEvaluationJson(result)}>
          Export Evaluation JSON
        </button>
        <button type="button" className="ghost-btn" onClick={() => exportScenarioCsv(result)}>
          Export Scenario Results CSV
        </button>
        <button type="button" className="ghost-btn" onClick={() => exportConfusionMatrixCsv(result)}>
          Export Confusion Matrix CSV
        </button>
        <button type="button" className="ghost-btn" onClick={() => exportPerFrameworkCsv(result)}>
          Export Per-Framework Accuracy CSV
        </button>
      </div>

      {expectation && (
        <div className="expectation-panel" data-testid="subdivision-expectation">
          <div className="section-heading">
            <span>Expected Outcome</span>
            <span>{expectation.decision_principle}</span>
          </div>

          <div className="expectation-grid">
            <div className="expectation-card">
              <span className="expectation-label">Expected Dominant Framework</span>
              <strong>{expectation.expected_dominant_framework}</strong>
              <p>{expectation.decision_principle}</p>
            </div>
            <div className="expectation-card">
              <span className="expectation-label">Core Property</span>
              <p>{expectation.core_property}</p>
            </div>
            <div className="expectation-card">
              <span className="expectation-label">Evaluation Rule</span>
              <p>{expectation.critical_evaluation_rule}</p>
            </div>
          </div>
        </div>
      )}

      <div className="section-heading">
        <span>Framework Distribution</span>
        <span>{result.summary.top_framework_label ?? 'Unresolved'} top share {formatPct(result.summary.top_framework_percentage)}</span>
      </div>
      {renderFrameworkDistribution(result.framework_distribution)}

      <div className="section-heading">
        <span>Per-Framework Accuracy</span>
        <span>{result.summary.correct_predictions} correct / {result.summary.incorrect_predictions} incorrect</span>
      </div>
      {renderPerFrameworkAccuracy(perFrameworkAccuracy)}

      <div className="section-heading">
        <span>Confusion Matrix</span>
        <span>expected x predicted</span>
      </div>
      {renderConfusionMatrix(confusionMatrix)}

      <div className="section-heading">
        <span>Scenario Results</span>
        <span>{result.summary.failed_runs} failed</span>
      </div>
      {renderScenarioRows(result.scenario_results)}
    </div>
  )
}
