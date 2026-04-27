import type { CSSProperties } from 'react'
import type { ReplayStage } from '../types'
import { JsonTree } from './JsonTree'

interface StageInspectorProps {
  currentStage: ReplayStage | null
  previousStage: ReplayStage | null
  isReplaying: boolean
}

type ChangeKind = 'added' | 'updated' | 'removed'

interface ChangeEntry {
  path: string
  kind: ChangeKind
  beforeValue: unknown
  afterValue: unknown
}

const STAGE_GUIDE: Record<
  string,
  {
    receives: string
    transforms: string
    produces: string
  }
> = {
  input: {
    receives: 'The backend receives the submitted scenario from the editor.',
    transforms: 'It records the submission shape so the rest of the pipeline knows what kind of input arrived.',
    produces: 'The replay baseline becomes the reference point for every later stage.',
  },
  parser: {
    receives: 'The parser takes the submitted scenario and normalizes it into the schema used by downstream analysis.',
    transforms: 'It extracts structured entities, scene context, and validation details from the raw submission.',
    produces: 'The normalized scenario lands in the parser artifact for later scoring and reasoning.',
  },
  math: {
    receives: 'The mathematical layer reads the parsed scenario and evaluates available actions deterministically.',
    transforms: 'It calculates risk scores, violations, and action tradeoffs using the thesis ruleset.',
    produces: 'A decision-oriented risk matrix is appended to the snapshot.',
  },
  rag: {
    receives: 'The retrieval stage uses the parsed scenario and deterministic analysis to search for relevant supporting sources.',
    transforms: 'It gathers ethical frameworks and supporting legal or policy material that can ground the final recommendation.',
    produces: 'A retrieval artifact is attached, including runtime status and retrieved evidence.',
  },
  reasoning: {
    receives: 'The reasoning stage reviews the parser output, deterministic risk scores, and retrieved supporting material together.',
    transforms: 'It synthesizes the evidence into an ethical recommendation and explains which framework carries the most weight.',
    produces: 'The snapshot gains the recommendation payload that can be shown to the user.',
  },
  complete: {
    receives: 'The completed pipeline bundles the accumulated artifacts from every previous stage.',
    transforms: 'It condenses the run into a display-safe summary that is easy to replay in the UI.',
    produces: 'The final response is published with a summary plus the stage-by-stage replay log.',
  },
}

function getValueAtPath(value: unknown, path: string): unknown {
  if (path === '$') {
    return value
  }

  const segments = path.match(/(?:\.([^[.\]]+))|\[(\d+)\]/g) ?? []
  let current = value

  for (const segment of segments) {
    if (segment.startsWith('.')) {
      if (!current || typeof current !== 'object' || Array.isArray(current)) {
        return undefined
      }
      current = (current as Record<string, unknown>)[segment.slice(1)]
      continue
    }

    const index = Number(segment.slice(1, -1))
    if (!Array.isArray(current)) {
      return undefined
    }
    current = current[index]
  }

  return current
}

function formatMetricValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (value === null) {
    return 'null'
  }
  return JSON.stringify(value)
}

function humanizeLabel(value: string): string {
  return value.replace(/_/g, ' ')
}

function getTopLevelPath(path: string): string {
  const match = path.match(/^\$(?:\.[^[.\]]+|\[\d+\])?/)
  return match?.[0] ?? path
}

function formatPathList(paths: string[]): string {
  if (!paths.length) {
    return 'no new JSON locations'
  }
  if (paths.length === 1) {
    return paths[0]
  }
  if (paths.length === 2) {
    return `${paths[0]} and ${paths[1]}`
  }
  return `${paths.slice(0, 2).join(', ')}, and ${paths.length - 2} more`
}

function truncatePreview(text: string, maxLines = 9, maxChars = 360): string {
  const lines = text.split('\n')
  const clippedLines = lines.slice(0, maxLines)
  let clipped = clippedLines.join('\n')
  if (clipped.length > maxChars) {
    clipped = `${clipped.slice(0, maxChars - 3)}...`
  } else if (lines.length > maxLines) {
    clipped = `${clipped}\n...`
  }
  return clipped
}

function formatPreview(value: unknown): string {
  if (typeof value === 'undefined') {
    return 'Not present in this snapshot.'
  }
  if (typeof value === 'string') {
    return truncatePreview(JSON.stringify(value))
  }
  if (typeof value === 'number' || typeof value === 'boolean' || value === null) {
    return JSON.stringify(value)
  }
  return truncatePreview(JSON.stringify(value, null, 2))
}

function buildChangeEntries(
  previousSnapshot: Record<string, unknown> | null,
  currentSnapshot: Record<string, unknown> | null,
  highlightPaths: string[],
): ChangeEntry[] {
  const uniquePaths = [...new Set(highlightPaths)]
  return uniquePaths.map((path) => {
    const beforeValue = getValueAtPath(previousSnapshot ?? {}, path)
    const afterValue = getValueAtPath(currentSnapshot ?? {}, path)
    let kind: ChangeKind = 'updated'

    if (typeof beforeValue === 'undefined' && typeof afterValue !== 'undefined') {
      kind = 'added'
    } else if (typeof beforeValue !== 'undefined' && typeof afterValue === 'undefined') {
      kind = 'removed'
    }

    return {
      path,
      kind,
      beforeValue,
      afterValue,
    }
  })
}

function getNarrative(stage: ReplayStage, previousStage: ReplayStage | null, changeEntries: ChangeEntry[]) {
  const guide = STAGE_GUIDE[stage.stage_id] ?? {
    receives: 'This stage receives the current pipeline snapshot.',
    transforms: 'It applies its stage-specific transformation to the snapshot.',
    produces: 'The resulting snapshot is pushed to the replay timeline.',
  }
  const upstream = previousStage
    ? `It starts from the ${previousStage.label} output.`
    : 'It starts from the user submission.'
  const changedRoots = [...new Set(changeEntries.map((entry) => getTopLevelPath(entry.path)))]
  const changeSummary = changeEntries.length
    ? `${changeEntries.length} JSON ${changeEntries.length === 1 ? 'path changed' : 'paths changed'}, centered on ${formatPathList(changedRoots)}.`
    : 'This step did not introduce any new JSON differences.'
  const metrics = Object.entries(stage.metrics)
  const metricSummary = metrics.length
    ? `Key metrics: ${metrics.map(([key, value]) => `${humanizeLabel(key)} ${formatMetricValue(value)}`).join(', ')}.`
    : `The stage completed with a ${stage.status} status in ${stage.duration_ms} ms.`

  return {
    receives: `${guide.receives} ${upstream}`,
    transforms: `${guide.transforms} ${changeSummary}`,
    produces: `${guide.produces} ${metricSummary}`,
  }
}

export function StageInspector({ currentStage, previousStage, isReplaying }: StageInspectorProps) {
  if (!currentStage) {
    return (
      <section className="stage-inspector stage-inspector-empty">
        <div className="section-heading">
          <span>Stage Inspector</span>
          <span className="artifact-caption">Before and after snapshots will appear here</span>
        </div>
        <p className="empty-state">
          Run a scenario to inspect what each stage receives, what it emits, and which JSON paths changed.
        </p>
      </section>
    )
  }

  const previousSnapshot = previousStage?.snapshot ?? {}
  const changeEntries = buildChangeEntries(previousStage?.snapshot ?? {}, currentStage.snapshot, currentStage.highlight_paths)
  const narrative = getNarrative(currentStage, previousStage, changeEntries)
  const metricEntries = Object.entries(currentStage.metrics)

  return (
    <section className="stage-inspector" data-testid="stage-inspector">
      <div className="stage-inspector-header">
        <div className="stage-inspector-copy">
          <p className="eyebrow inspector-eyebrow">Stage Inspector</p>
          <h2>{currentStage.label}</h2>
          <p className="stage-inspector-headline">{currentStage.headline}</p>
        </div>
        <div className="stage-inspector-status">
          <span className={`stage-badge status-${currentStage.status}`}>{currentStage.status}</span>
          <span className="stage-duration">{currentStage.duration_ms} ms</span>
        </div>
      </div>

      <div className="stage-story-grid">
        <article className="story-card">
          <span className="story-label">Receives</span>
          <p>{narrative.receives}</p>
        </article>
        <article className="story-card">
          <span className="story-label">Transforms</span>
          <p>{narrative.transforms}</p>
        </article>
        <article className="story-card">
          <span className="story-label">Produces</span>
          <p>{narrative.produces}</p>
        </article>
      </div>

      <div className="inspector-metrics">
        <span className="metric-chip">status: {currentStage.status}</span>
        <span className="metric-chip">duration: {currentStage.duration_ms} ms</span>
        {metricEntries.map(([key, value]) => (
          <span key={key} className="metric-chip">
            {humanizeLabel(key)}: {formatMetricValue(value)}
          </span>
        ))}
      </div>

      <div className="stage-flow-shell" key={currentStage.stage_id}>
        <section className="flow-panel">
          <div className="flow-panel-header">
            <div>
              <span className="flow-panel-title">What Goes In</span>
              <p className="flow-panel-caption">
                {previousStage ? `Snapshot carried forward from ${previousStage.label}.` : 'Initial payload entering the pipeline.'}
              </p>
            </div>
            <span className="flow-source-tag">{previousStage ? previousStage.label : 'Submission'}</span>
          </div>
          <JsonTree
            value={previousSnapshot}
            highlightPaths={currentStage.highlight_paths}
            rootLabel={previousStage ? `${previousStage.label} Snapshot` : 'Incoming Snapshot'}
          />
        </section>

        <div className={`flow-bridge ${isReplaying ? 'is-live' : ''}`}>
          <div className="flow-node">
            <span className="flow-node-label">{currentStage.label}</span>
            <strong>{currentStage.status === 'success' ? 'Applied' : 'Processed'}</strong>
          </div>
          <div className="flow-track" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p className="flow-bridge-text">{currentStage.headline}</p>
          <div className="highlight-strip" data-testid="highlight-strip">
            {currentStage.highlight_paths.length ? (
              currentStage.highlight_paths.map((path) => (
                <span key={path} className="highlight-pill">
                  {path}
                </span>
              ))
            ) : (
              <span className="highlight-pill muted">No changed JSON paths</span>
            )}
          </div>
        </div>

        <section className="flow-panel">
          <div className="flow-panel-header">
            <div>
              <span className="flow-panel-title">What Comes Out</span>
              <p className="flow-panel-caption">Updated snapshot after the current stage finishes.</p>
            </div>
            <span className="flow-source-tag">Output</span>
          </div>
          <JsonTree
            value={currentStage.snapshot}
            highlightPaths={currentStage.highlight_paths}
            rootLabel={`${currentStage.label} Snapshot`}
          />
        </section>
      </div>

      <section className="change-ledger">
        <div className="section-heading">
          <span>Changed Paths</span>
          <span className="artifact-caption">
            {changeEntries.length ? `${changeEntries.length} highlighted change${changeEntries.length === 1 ? '' : 's'}` : 'No JSON differences'}
          </span>
        </div>

        {changeEntries.length ? (
          <div className="change-grid">
            {changeEntries.map((entry, index) => (
              <article
                key={entry.path}
                className={`change-card change-${entry.kind}`}
                style={{ '--change-order': `${index}` } as CSSProperties}
              >
                <div className="change-card-header">
                  <span className="change-path">{entry.path}</span>
                  <span className={`change-kind change-${entry.kind}`}>{entry.kind}</span>
                </div>
                <div className="change-delta">
                  <div className="change-preview">
                    <span className="change-preview-label">Before</span>
                    <pre>{formatPreview(entry.beforeValue)}</pre>
                  </div>
                  <div className="change-preview">
                    <span className="change-preview-label">After</span>
                    <pre>{formatPreview(entry.afterValue)}</pre>
                  </div>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-diff">
            <p className="empty-state">This stage preserved the snapshot without changing any highlighted JSON fields.</p>
          </div>
        )}
      </section>
    </section>
  )
}
