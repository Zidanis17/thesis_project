import type { CSSProperties } from 'react'
import type { SubdivisionFrameworkMetric, SubdivisionRunResponse } from '../types'

interface SubdivisionMetricsPanelProps {
  result: SubdivisionRunResponse | null
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

function getFrameworkColor(frameworkId: string | null): string {
  if (!frameworkId) return FRAMEWORK_COLORS.unresolved
  return FRAMEWORK_COLORS[frameworkId] ?? FRAMEWORK_COLORS.unresolved
}

function buildBarStyle(metric: SubdivisionFrameworkMetric): CSSProperties {
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

export function SubdivisionMetricsPanel({ result, isLoading, error }: SubdivisionMetricsPanelProps) {
  if (isLoading) {
    return (
      <div className="subdivision-panel-body">
        <div className="proc-banner" data-testid="subdivision-processing-banner">
          <div className="proc-bar">
            <div className="proc-fill" />
          </div>
          <p>Running every scenario in the selected subdivision and aggregating framework outcomes.</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="subdivision-panel-body">
        <div className="error-block">
          <strong className="error-code">subdivision_run_error</strong>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="subdivision-panel-body">
        <div className="empty-state">
          Run a subdivision to compare dominant-framework percentages across all scenarios in that slice.
        </div>
      </div>
    )
  }

  const expectation = result.subdivision.expectation
  const expectedFramework = expectation?.expected_dominant_framework ?? result.subdivision.expected_framework ?? 'N/A'
  const expectationMatched = expectation?.expected_dominant_framework === result.summary.top_framework

  return (
    <div className="subdivision-panel-body">
      <div className="subdivision-summary-row">
        <div className="metric-card accent">
          <span className="metric-label">Subdivision</span>
          <strong className="metric-value">{result.subdivision.label}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Scenarios</span>
          <strong className="metric-value">{result.summary.scenario_count}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Top Framework</span>
          <strong className="metric-value">{result.summary.top_framework_label ?? 'Unresolved'}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Batch Duration</span>
          <strong className="metric-value">{formatDuration(result.summary.total_duration_ms)}</strong>
        </div>
      </div>

      <div className="subdivision-meta-strip">
        <span className="metric-chip">{result.summary.completion_rate_pct}% completed</span>
        <span className="metric-chip">{result.summary.reasoning_runtime_ready_pct}% reasoning ready</span>
        <span className="metric-chip">{result.summary.rag_runtime_ready_pct}% RAG ready</span>
        <span className="metric-chip">Expected: {expectedFramework}</span>
        {expectation && (
          <span className={`metric-chip ${expectationMatched ? 'status-chip-ok' : 'status-chip-error'}`}>
            {expectationMatched ? 'Observed framework matches expectation' : 'Observed framework diverges from expectation'}
          </span>
        )}
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
              <span className="expectation-label">Expected Behavior</span>
              <p>{expectation.expected_behavior}</p>
            </div>
            <div className="expectation-card">
              <span className="expectation-label">Expected Action Pattern</span>
              <p>{expectation.expected_action_pattern}</p>
            </div>
            <div className="expectation-card">
              <span className="expectation-label">Expected Contributors</span>
              <p>{expectation.expected_contributing_frameworks.join(', ')}</p>
            </div>
            <div className="expectation-card">
              <span className="expectation-label">What This Proves</span>
              <p>{expectation.proving_point}</p>
            </div>
          </div>

          <div className="expectation-rule">
            <span className="expectation-label">Critical Evaluation Rule</span>
            <p>{expectation.critical_evaluation_rule}</p>
          </div>
        </div>
      )}

      <div className="section-heading">
        <span>Framework Distribution</span>
        <span>{result.summary.scenario_count} scenario outputs</span>
      </div>

      <div className="framework-graph" data-testid="framework-graph">
        {result.framework_distribution.map((metric) => (
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

      <div className="section-heading">
        <span>Scenario Results</span>
        <span>{result.summary.failed_runs} failed</span>
      </div>

      <div className="subdivision-table-wrap">
        <table className="subdivision-table">
          <thead>
            <tr>
              <th scope="col">Scenario</th>
              <th scope="col">Framework</th>
              <th scope="col">Deterministic Best</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {result.scenario_results.map((scenario) => (
              <tr key={scenario.scenario_id}>
                <td>
                  <div className="subdivision-scenario-cell">
                    <strong>{scenario.scenario_label}</strong>
                    <span>{scenario.scenario_id}</span>
                  </div>
                </td>
                <td>{scenario.dominant_framework ?? 'Unresolved'}</td>
                <td>{scenario.deterministic_best_action ?? 'N/A'}</td>
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
    </div>
  )
}
