import type { ReplayStage } from '../types'

interface StageTimelineProps {
  stages: ReplayStage[]
  activeIndex: number
  isReplaying: boolean
  onSelectStage: (index: number) => void
}

export function StageTimeline({
  stages,
  activeIndex,
  isReplaying,
  onSelectStage,
}: StageTimelineProps) {
  return (
    <div className="stage-timeline">
      <div className="section-heading timeline-heading">
        <span>Pipeline Replay</span>
        <span className={`replay-pill ${isReplaying ? 'is-live' : ''}`}>
          {isReplaying ? 'Live Replay' : 'Settled'}
        </span>
      </div>
      <ol className="stage-list" data-testid="stage-timeline">
        {stages.map((stage, index) => {
          const stateClass =
            index === activeIndex ? 'is-active' : index < activeIndex ? 'is-complete' : 'is-pending'
          return (
            <li key={stage.stage_id} className={`stage-item ${stateClass} stage-${stage.status}`}>
              <button type="button" className="stage-card" onClick={() => onSelectStage(index)}>
                <div className="stage-row">
                  <span className="stage-index">{String(index + 1).padStart(2, '0')}</span>
                  <div className="stage-card-copy">
                    <span className="stage-label">{stage.label}</span>
                    <span className="stage-duration">{stage.duration_ms} ms</span>
                  </div>
                  <span className={`stage-status status-${stage.status}`}>{stage.status}</span>
                </div>
                <p className="stage-headline">{stage.headline}</p>
                <div className="stage-metrics">
                  {Object.entries(stage.metrics).map(([key, value]) => (
                    <span key={key} className="metric-chip">
                      {key}: {String(value)}
                    </span>
                  ))}
                </div>
              </button>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
