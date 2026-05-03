import type { ArtifactTabKey } from '../types'
import { JsonTree } from './JsonTree'

const ARTIFACT_LABELS: Record<ArtifactTabKey, string> = {
  parser_result: 'Parser',
  mathematical_layer_result: 'Math',
  agentic_assessment: 'Agentic',
  rag_retrieval_result: 'RAG',
  reasoning_result: 'Reasoning',
  agentic_validation_result: 'Validation',
}

interface ArtifactTabsProps {
  selectedTab: ArtifactTabKey
  onSelectTab: (tab: ArtifactTabKey) => void
  artifacts: Record<ArtifactTabKey, Record<string, unknown>>
}

export function ArtifactTabs({
  selectedTab,
  onSelectTab,
  artifacts,
}: ArtifactTabsProps) {
  return (
    <section className="artifact-panel">
      <div className="section-heading">
        <span>Artifacts</span>
        <span className="artifact-caption">Display-safe backend payloads</span>
      </div>
      <div className="artifact-tabs">
        {(Object.keys(ARTIFACT_LABELS) as ArtifactTabKey[]).map((tab) => (
          <button
            key={tab}
            type="button"
            className={`artifact-tab ${selectedTab === tab ? 'is-selected' : ''}`}
            onClick={() => onSelectTab(tab)}
          >
            {ARTIFACT_LABELS[tab]}
          </button>
        ))}
      </div>
      <div className="artifact-body">
        <JsonTree value={artifacts[selectedTab]} highlightPaths={[]} rootLabel={ARTIFACT_LABELS[selectedTab]} />
      </div>
    </section>
  )
}
