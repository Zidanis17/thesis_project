import type { ScenarioRunHistoryResponse, ScenarioRunRecord } from '../types'

interface RunHistoryPanelProps {
  history: ScenarioRunHistoryResponse | null
  isLoading: boolean
  isHydrating: boolean
  error: string | null
  selectedRunId: string | null
  currentRun: ScenarioRunRecord | null
  onRefresh: () => void
  onSelectRun: (runId: string) => void
  onLoadRun: (runId: string) => void
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(parsed)
}

function formatRate(value: number, total: number): string {
  if (!total) return '0%'
  return `${Math.round((value / total) * 100)}%`
}

function renderRunCard(title: string, run: ScenarioRunRecord | null, className: string) {
  return (
    <div className={`run-compare-card ${className}`}>
      <div className="run-compare-head">
        <span className="run-compare-title">{title}</span>
        {run && (
          <span className={`metric-chip ${run.status === 'error' ? 'status-chip-error' : 'status-chip-ok'}`}>
            {run.status}
          </span>
        )}
      </div>

      {run ? (
        <>
          <strong className="run-compare-primary">{run.dominant_framework ?? 'Unresolved'}</strong>
          <p className="run-compare-copy">{run.input_preview}</p>
          <div className="run-compare-meta">
            <span className="metric-chip">{run.deterministic_best_action ?? 'No best action'}</span>
            <span className="metric-chip">{run.resolved_input_mode ?? run.input_mode_hint}</span>
            {run.model_name && <span className="metric-chip">{run.model_name}</span>}
          </div>
        </>
      ) : (
        <div className="empty-state">Select or run a scenario to populate this comparison card.</div>
      )}
    </div>
  )
}

function buildComparisonNotes(currentRun: ScenarioRunRecord | null, selectedRun: ScenarioRunRecord | null): string[] {
  if (!currentRun || !selectedRun || currentRun.id === selectedRun.id) {
    return []
  }

  const notes: string[] = []
  if (currentRun.deterministic_best_action === selectedRun.deterministic_best_action) {
    notes.push(`Same deterministic action: ${currentRun.deterministic_best_action ?? 'N/A'}`)
  } else {
    notes.push(
      `Deterministic action changed from ${selectedRun.deterministic_best_action ?? 'N/A'} to ${currentRun.deterministic_best_action ?? 'N/A'}`,
    )
  }

  if (currentRun.dominant_framework === selectedRun.dominant_framework) {
    notes.push(`Same dominant framework: ${currentRun.dominant_framework ?? 'Unresolved'}`)
  } else {
    notes.push(
      `Dominant framework changed from ${selectedRun.dominant_framework ?? 'Unresolved'} to ${currentRun.dominant_framework ?? 'Unresolved'}`,
    )
  }

  if (currentRun.status !== selectedRun.status) {
    notes.push(`Run status changed from ${selectedRun.status} to ${currentRun.status}`)
  }

  return notes
}

export function RunHistoryPanel({
  history,
  isLoading,
  isHydrating,
  error,
  selectedRunId,
  currentRun,
  onRefresh,
  onSelectRun,
  onLoadRun,
}: RunHistoryPanelProps) {
  const selectedRun = history?.runs.find((run) => run.id === selectedRunId) ?? null
  const comparisonNotes = buildComparisonNotes(currentRun, selectedRun)

  if (isLoading && !history) {
    return (
      <div className="run-history-body">
        <div className="proc-banner" data-testid="run-history-loading">
          <div className="proc-bar">
            <div className="proc-fill" />
          </div>
          <p>Loading persisted scenario runs from SQLite.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="run-history-body">
      <div className="run-history-summary-row">
        <div className="metric-card accent">
          <span className="metric-label">Stored Runs</span>
          <strong className="metric-value">{history?.total_runs ?? 0}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Successful</span>
          <strong className="metric-value">{history?.success_runs ?? 0}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Failed</span>
          <strong className="metric-value">{history?.failed_runs ?? 0}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Success Rate</span>
          <strong className="metric-value">
            {history ? formatRate(history.success_runs, history.total_runs) : '0%'}
          </strong>
        </div>
      </div>

      <div className="run-history-toolbar">
        <div className="catalog-meta">
          <span className="catalog-chip accent">SQLite history</span>
          <span className="catalog-chip">{history?.runs.length ?? 0} loaded</span>
          {selectedRun && <span className="catalog-chip">Comparing {selectedRun.id.slice(0, 8)}</span>}
        </div>
        <button type="button" className="ghost-btn" onClick={onRefresh} disabled={isLoading || isHydrating}>
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="error-block">
          <strong className="error-code">run_history_error</strong>
          <p>{error}</p>
        </div>
      )}

      <div className="section-heading">
        <span>Compare Runs</span>
        <span>{isHydrating ? 'Loading stored replay...' : 'Summary-level comparison'}</span>
      </div>

      <div className="run-compare-grid">
        {renderRunCard('Current Run', currentRun, 'is-current')}
        {renderRunCard('Selected History Run', selectedRun, 'is-history')}
      </div>

      {comparisonNotes.length > 0 && (
        <div className="run-compare-notes">
          {comparisonNotes.map((note) => (
            <span key={note} className="metric-chip">
              {note}
            </span>
          ))}
        </div>
      )}

      <div className="section-heading">
        <span>Stored Runs</span>
        <span>{history?.total_runs ?? 0} total</span>
      </div>

      {!history?.runs.length ? (
        <div className="empty-state">
          Run a scenario once and it will appear here with its saved metadata, replay, and artifacts.
        </div>
      ) : (
        <div className="run-history-list" data-testid="run-history-list">
          {history.runs.map((run) => {
            const isSelected = run.id === selectedRunId
            const isCurrent = run.id === currentRun?.id
            return (
              <div
                key={run.id}
                className={`run-history-row ${isSelected ? 'is-selected' : ''} ${isCurrent ? 'is-current' : ''}`}
              >
                <div className="run-history-copy">
                  <div className="run-history-meta">
                    <strong>{formatTimestamp(run.created_at)}</strong>
                    <span className={`metric-chip ${run.status === 'error' ? 'status-chip-error' : 'status-chip-ok'}`}>
                      {run.status}
                    </span>
                    <span className="metric-chip">{run.resolved_input_mode ?? run.input_mode_hint}</span>
                    {isCurrent && <span className="metric-chip">Loaded</span>}
                  </div>
                  <p className="run-history-preview">{run.input_preview}</p>
                  <div className="run-history-tags">
                    <span className="metric-chip">{run.deterministic_best_action ?? 'No best action'}</span>
                    <span className="metric-chip">{run.dominant_framework ?? 'Unresolved'}</span>
                    <span className="metric-chip">{run.replay_stage_count} stages</span>
                    {run.model_name && <span className="metric-chip">{run.model_name}</span>}
                  </div>
                </div>
                <div className="run-history-actions">
                  <button
                    type="button"
                    className={`ghost-btn ${isSelected ? 'is-active' : ''}`}
                    onClick={() => onSelectRun(run.id)}
                    disabled={isHydrating}
                  >
                    {isSelected ? 'Comparing' : 'Compare'}
                  </button>
                  <button
                    type="button"
                    className="ghost-btn"
                    onClick={() => onLoadRun(run.id)}
                    disabled={isHydrating}
                  >
                    {isHydrating && isSelected ? 'Loading...' : 'Load'}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
