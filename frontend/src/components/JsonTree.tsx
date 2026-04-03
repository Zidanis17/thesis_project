import type { ReactNode } from 'react'

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

interface JsonTreeProps {
  value: unknown
  highlightPaths: string[]
  rootLabel?: string
}

function isHighlighted(path: string, highlightPaths: string[]): boolean {
  return highlightPaths.some(
    (candidate) =>
      candidate === path ||
      candidate.startsWith(`${path}.`) ||
      candidate.startsWith(`${path}[`),
  )
}

function renderPrimitive(value: string | number | boolean | null): string {
  return JSON.stringify(value)
}

function JsonBranch({
  value,
  path,
  label,
  depth,
  highlightPaths,
}: {
  value: JsonValue
  path: string
  label: string | null
  depth: number
  highlightPaths: string[]
}): ReactNode {
  const highlighted = isHighlighted(path, highlightPaths)
  const indentStyle = { paddingLeft: `${depth * 1.1}rem` }

  if (value === null || typeof value !== 'object') {
    return (
      <div className={`json-line ${highlighted ? 'is-highlighted' : ''}`} style={indentStyle}>
        {label ? <span className="json-key">{label}: </span> : null}
        <span className="json-value">{renderPrimitive(value)}</span>
      </div>
    )
  }

  if (Array.isArray(value)) {
    return (
      <div className={`json-block ${highlighted ? 'is-highlighted' : ''}`}>
        <div className="json-line" style={indentStyle}>
          {label ? <span className="json-key">{label}: </span> : null}
          <span>[</span>
        </div>
        {value.map((item, index) => (
          <JsonBranch
            key={`${path}[${index}]`}
            value={item as JsonValue}
            path={`${path}[${index}]`}
            label={null}
            depth={depth + 1}
            highlightPaths={highlightPaths}
          />
        ))}
        <div className="json-line" style={indentStyle}>
          <span>]</span>
        </div>
      </div>
    )
  }

  return (
    <div className={`json-block ${highlighted ? 'is-highlighted' : ''}`}>
      <div className="json-line" style={indentStyle}>
        {label ? <span className="json-key">{label}: </span> : null}
        <span>{'{'}</span>
      </div>
      {Object.entries(value).map(([key, childValue]) => (
        <JsonBranch
          key={`${path}.${key}`}
          value={childValue as JsonValue}
          path={path === '$' ? `$.${key}` : `${path}.${key}`}
          label={key}
          depth={depth + 1}
          highlightPaths={highlightPaths}
        />
      ))}
      <div className="json-line" style={indentStyle}>
        <span>{'}'}</span>
      </div>
    </div>
  )
}

export function JsonTree({ value, highlightPaths, rootLabel = 'Snapshot' }: JsonTreeProps) {
  return (
    <div className="json-tree" data-testid="json-tree">
      <JsonBranch
        value={(value ?? {}) as JsonValue}
        path="$"
        label={rootLabel}
        depth={0}
        highlightPaths={highlightPaths}
      />
    </div>
  )
}
