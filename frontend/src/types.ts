export type InputEditorMode = 'json' | 'text'
export type StageStatus = 'success' | 'warning' | 'skipped' | 'error'
export type RunStatus = 'success' | 'error'
export type EvaluationVariant = 'full_system' | 'no_rag' | 'no_math' | 'no_rag_no_math'
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
  subdivision_id?: string | null
  subdivision_label?: string | null
  expected_framework?: string | null
}

export interface SubdivisionExpectation {
  expected_dominant_framework: string
  decision_principle: string
  core_property: string
  expected_behavior: string
  expected_contributing_frameworks: string[]
  expected_action_pattern: string
  proving_point: string
  critical_evaluation_rule: string
}

export interface ScenarioSubdivision {
  id: string
  label: string
  scenario_count: number
  expected_framework?: string | null
  expectation?: SubdivisionExpectation | null
}

export interface ScenarioCatalogResponse {
  examples: ExampleItem[]
  subdivisions: ScenarioSubdivision[]
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

export interface ScenarioRunRecord {
  id: string
  created_at: string
  status: RunStatus
  input_mode_hint: InputEditorMode | 'auto'
  resolved_input_mode: string | null
  submitted_kind: 'json' | 'text'
  input_preview: string
  model_name: string | null
  deterministic_best_action: string | null
  dominant_framework: string | null
  rag_runtime_available: boolean
  reasoning_runtime_available: boolean
  error_code: string | null
  error_message: string | null
  replay_stage_count: number
}

export interface RunSuccessResponse {
  run: ScenarioRunRecord
  summary: {
    input_mode: string
    variant?: EvaluationVariant
    parser_warnings: string[]
    violated_rules: string[]
    deterministic_best_action: string | null
    dominant_framework: string | null
    rag_runtime_available: boolean
    math_runtime_available?: boolean
    reasoning_runtime_available: boolean
    reasoning_runtime_error: string | null
    rag_runtime_error: string | null
  }
  artifacts: Record<ArtifactTabKey, Record<string, unknown>>
  replay: ReplayStage[]
}

export interface RunErrorResponse {
  run: ScenarioRunRecord
  error: {
    code: string
    message: string
  }
  replay: ReplayStage[]
}

export type RunEnvelope =
  | { kind: 'success'; payload: RunSuccessResponse }
  | { kind: 'error'; payload: RunErrorResponse }

export interface ScenarioRunHistoryResponse {
  runs: ScenarioRunRecord[]
  total_runs: number
  success_runs: number
  failed_runs: number
}

export interface FrameworkDistributionMetric {
  framework_id: string | null
  framework_label: string
  count: number
  percentage: number
}

export type SubdivisionFrameworkMetric = FrameworkDistributionMetric

export interface EvaluationScenarioResult {
  scenario_id: string
  scenario_label: string
  subdivision_id: string | null
  subdivision_label: string | null
  expected_framework: string | null
  dominant_framework: string | null
  correct_prediction: boolean | null
  confidence: number | null
  contributing_frameworks: string[]
  weights: Record<string, number>
  retrieved_framework_ids: string[]
  expected_framework_retrieved: boolean
  top_retrieved_framework: string | null
  top_retrieval_score: number | null
  reasoning_contract_valid: boolean
  dominant_framework_valid: boolean
  weights_sum_to_one: boolean
  risk_matrix_preserved: boolean
  no_recommended_action: boolean
  violated_constraints_supported: boolean
  deterministic_best_action: string | null
  status: 'success' | 'error'
  duration_ms: number
  reasoning_runtime_available: boolean
  rag_runtime_available: boolean
  error_code: string | null
  error_message: string | null
}

export type SubdivisionScenarioResult = EvaluationScenarioResult

export interface PerFrameworkAccuracyMetric {
  framework_id: string
  expected_count: number
  correct_count: number
  incorrect_count: number
  accuracy_pct: number
}

export interface ConfusionMatrixRow {
  expected: string
  predictions: Record<string, number>
}

export interface ConfusionMatrix {
  labels: string[]
  rows: ConfusionMatrixRow[]
}

export interface EvaluationSummary {
  scenario_count: number
  completed_runs: number
  failed_runs: number
  completion_rate_pct: number
  correct_predictions: number
  incorrect_predictions: number
  accuracy_pct: number
  expected_framework: string | null
  expected_framework_total: number
  average_confidence: number | null
  average_confidence_correct: number | null
  average_confidence_incorrect: number | null
  reasoning_runtime_ready_pct: number
  rag_runtime_ready_pct: number
  rag_retrieval_hit_rate_pct: number
  risk_matrix_preservation_rate_pct: number
  reasoning_contract_validity_rate_pct: number
  weights_validity_rate_pct: number
  top_framework: string | null
  top_framework_label: string | null
  top_framework_percentage: number
  total_duration_ms: number
}

export interface EvaluationRunResponse {
  evaluation_id: string | null
  created_at: string
  scope: 'subdivision' | 'full_bank'
  variant: EvaluationVariant
  subdivision_id: string | null
  subdivision_label: string | null
  subdivision?: ScenarioSubdivision
  total_scenarios: number
  completed_runs: number
  failed_runs: number
  completion_rate_pct: number
  correct_predictions: number
  incorrect_predictions: number
  overall_accuracy_pct: number
  total_duration_ms: number
  expected_framework_distribution: FrameworkDistributionMetric[]
  framework_distribution: FrameworkDistributionMetric[]
  per_framework_accuracy: PerFrameworkAccuracyMetric[]
  confusion_matrix: ConfusionMatrix
  rag_retrieval_hit_rate_pct: number
  reasoning_contract_validity_rate_pct: number
  risk_matrix_preservation_rate_pct: number
  weights_validity_rate_pct: number
  summary: EvaluationSummary
  scenario_results: EvaluationScenarioResult[]
}

export type SubdivisionRunResponse = EvaluationRunResponse & {
  subdivision: ScenarioSubdivision
}

export interface EvaluationRunRecord {
  id: string
  created_at: string
  scope: 'subdivision' | 'full_bank'
  subdivision_id: string | null
  variant: string | null
  model_name: string | null
  total_scenarios: number
  completed_runs: number
  failed_runs: number
  overall_accuracy_pct: number
}

export interface EvaluationRunHistoryResponse {
  runs: EvaluationRunRecord[]
  total_runs: number
  full_bank_runs: number
  subdivision_runs: number
}

export interface LegacySubdivisionRunResponse {
  subdivision: ScenarioSubdivision
  summary: {
    scenario_count: number
    completed_runs: number
    failed_runs: number
    completion_rate_pct: number
    correct_predictions: number
    incorrect_predictions: number
    accuracy_pct: number
    expected_framework: string | null
    expected_framework_total: number
    average_confidence: number | null
    average_confidence_correct: number | null
    average_confidence_incorrect: number | null
    reasoning_runtime_ready_pct: number
    rag_runtime_ready_pct: number
    rag_retrieval_hit_rate_pct: number
    risk_matrix_preservation_rate_pct: number
    reasoning_contract_validity_rate_pct: number
    weights_validity_rate_pct: number
    top_framework: string | null
    top_framework_label: string | null
    top_framework_percentage: number
    total_duration_ms: number
  }
  framework_distribution: FrameworkDistributionMetric[]
  scenario_results: EvaluationScenarioResult[]
}
