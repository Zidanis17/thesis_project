import type {
  EvaluationRunHistoryResponse,
  EvaluationRunResponse,
  HealthResponse,
  InputEditorMode,
  RunEnvelope,
  RunErrorResponse,
  ScenarioRunHistoryResponse,
  RunSuccessResponse,
  ScenarioCatalogResponse,
  SubdivisionRunResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function readJson<T>(response: Response): Promise<T> {
  return (await response.json()) as T
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/v1/health`)
  if (!response.ok) {
    throw new Error('Failed to load backend health.')
  }
  return readJson<HealthResponse>(response)
}

export async function fetchScenarioCatalog(): Promise<ScenarioCatalogResponse> {
  const response = await fetch(`${API_BASE}/api/v1/examples`)
  if (!response.ok) {
    throw new Error('Failed to load showcase examples.')
  }
  return readJson<ScenarioCatalogResponse>(response)
}

export async function runScenario(
  input: string | Record<string, unknown>,
  inputModeHint: InputEditorMode,
): Promise<RunEnvelope> {
  const response = await fetch(`${API_BASE}/api/v1/scenario/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input,
      input_mode_hint: inputModeHint,
    }),
  })

  if (response.ok) {
    return { kind: 'success', payload: await readJson<RunSuccessResponse>(response) }
  }

  if (response.status === 400) {
    return { kind: 'error', payload: await readJson<RunErrorResponse>(response) }
  }

  throw new Error(`Backend request failed with status ${response.status}.`)
}

export async function fetchScenarioRunHistory(limit = 25): Promise<ScenarioRunHistoryResponse> {
  const response = await fetch(`${API_BASE}/api/v1/scenario/runs?limit=${limit}`)
  if (!response.ok) {
    throw new Error('Failed to load stored run history.')
  }
  return readJson<ScenarioRunHistoryResponse>(response)
}

export async function fetchScenarioRunById(runId: string): Promise<RunEnvelope> {
  const response = await fetch(`${API_BASE}/api/v1/scenario/runs/${runId}`)
  if (!response.ok) {
    const payload = await readJson<{ error?: { message?: string } }>(response)
    throw new Error(payload.error?.message ?? `Stored run request failed with status ${response.status}.`)
  }

  const payload = await readJson<RunSuccessResponse | RunErrorResponse>(response)
  if (payload.run.status === 'error') {
    return { kind: 'error', payload: payload as RunErrorResponse }
  }
  return { kind: 'success', payload: payload as RunSuccessResponse }
}

export async function runSubdivision(subdivisionId: string): Promise<SubdivisionRunResponse> {
  const response = await fetch(`${API_BASE}/api/v1/scenario/subdivision/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      subdivision_id: subdivisionId,
    }),
  })

  if (!response.ok) {
    const payload = await readJson<{ error?: { message?: string } }>(response)
    throw new Error(payload.error?.message ?? `Subdivision request failed with status ${response.status}.`)
  }

  return readJson<SubdivisionRunResponse>(response)
}

export async function runScenarioBank(variant: 'full_system' | 'no_rag' = 'full_system'): Promise<EvaluationRunResponse> {
  const response = await fetch(`${API_BASE}/api/v1/scenario/bank/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ variant }),
  })

  if (!response.ok) {
    const payload = await readJson<{ error?: { message?: string } }>(response)
    throw new Error(payload.error?.message ?? `Scenario bank request failed with status ${response.status}.`)
  }

  return readJson<EvaluationRunResponse>(response)
}

export async function fetchEvaluationHistory(limit = 25): Promise<EvaluationRunHistoryResponse> {
  const response = await fetch(`${API_BASE}/api/v1/evaluations?limit=${limit}`)
  if (!response.ok) {
    throw new Error('Failed to load stored evaluation history.')
  }
  return readJson<EvaluationRunHistoryResponse>(response)
}

export async function fetchEvaluationById(evaluationId: string): Promise<EvaluationRunResponse> {
  const response = await fetch(`${API_BASE}/api/v1/evaluations/${evaluationId}`)
  if (!response.ok) {
    const payload = await readJson<{ error?: { message?: string } }>(response)
    throw new Error(payload.error?.message ?? `Stored evaluation request failed with status ${response.status}.`)
  }
  return readJson<EvaluationRunResponse>(response)
}
