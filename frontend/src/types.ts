export type InputEditorMode = 'json' | 'text'
export type StageStatus = 'success' | 'warning' | 'skipped' | 'error'
export type ArtifactTabKey =
  | 'parser_result'
  | 'mathematical_layer_result'
  | 'rag_retrieval_result'
  | 'reasoning_result'

export interface ExampleItem {
  id: string
  label: string
  mode: InputEditorMode
  value: string | Record<string, unknown>
}

export interface HealthResponse {
  status: string
  knowledge_base_path: string | null
  rag: {
    runtime_available: boolean
    runtime_error: string | null
  }
  reasoning: {
    runtime_available: boolean
    runtime_error: string | null
    model_name: string | null
  }
  warnings: string[]
}

export interface ReplayStage {
  stage_id: string
  label: string
  status: StageStatus
  duration_ms: number
  headline: string
  snapshot: Record<string, unknown>
  highlight_paths: string[]
  metrics: Record<string, unknown>
}

export interface RunSuccessResponse {
  summary: {
    input_mode: string
    parser_warnings: string[]
    violated_rules: string[]
    deterministic_best_action: string
    recommended_action: string | null
    dominant_framework: string | null
    rag_runtime_available: boolean
    reasoning_runtime_available: boolean
    reasoning_runtime_error: string | null
    rag_runtime_error: string | null
  }
  artifacts: Record<ArtifactTabKey, Record<string, unknown>>
  replay: ReplayStage[]
}

export interface RunErrorResponse {
  error: {
    code: string
    message: string
  }
  replay: ReplayStage[]
}

export type RunEnvelope =
  | { kind: 'success'; payload: RunSuccessResponse }
  | { kind: 'error'; payload: RunErrorResponse }
