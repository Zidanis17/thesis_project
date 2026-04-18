import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const healthPayload = {
  status: 'ok',
  knowledge_base_path: 'C:/repo/backend/knowledge_base',
  rag: { runtime_available: true, runtime_error: null },
  reasoning: { runtime_available: true, runtime_error: null, model_name: 'gpt-4.1-mini' },
  warnings: [],
}

const catalogPayload = {
  examples: [
    {
      id: 'json-example',
      label: 'JSON Example',
      mode: 'json',
      value: {
        ego_vehicle: { speed_kmh: 50 },
        environment: { road_type: 'residential' },
      },
      subdivision_id: 'routine_rule_governed',
      subdivision_label: 'Routine Rule Governed',
      expected_framework: 'EF-02',
    },
    {
      id: 'text-example',
      label: 'Text Example',
      mode: 'text',
      value: 'A child is crossing the road.',
      subdivision_id: null,
      subdivision_label: null,
      expected_framework: null,
    },
  ],
  subdivisions: [
    {
      id: 'routine_rule_governed',
      label: 'Routine Rule Governed',
      scenario_count: 2,
      expected_framework: 'EF-02',
      expectation: {
        expected_dominant_framework: 'EF-02',
        decision_principle: 'Rule compliance',
        core_property: 'Collision remains avoidable and a safe RSS-compliant action exists.',
        expected_behavior: 'Reject unsafe maneuvers and choose the safe RSS-compliant trajectory.',
        expected_contributing_frameworks: ['EF-04', 'EF-06'],
        expected_action_pattern: 'safe RSS-compliant trajectory such as brake_straight',
        proving_point: 'The system prioritizes legal and safety constraints over optimization.',
        critical_evaluation_rule:
          'A prediction is correct when the dominant framework matches the subdivision expectation.',
      },
    },
  ],
}

const subdivisionPayload = {
  subdivision: {
    id: 'routine_rule_governed',
    label: 'Routine Rule Governed',
    scenario_count: 2,
    expected_framework: 'EF-02',
    expectation: {
      expected_dominant_framework: 'EF-02',
      decision_principle: 'Rule compliance',
      core_property: 'Collision remains avoidable and a safe RSS-compliant action exists.',
      expected_behavior: 'Reject unsafe maneuvers and choose the safe RSS-compliant trajectory.',
      expected_contributing_frameworks: ['EF-04', 'EF-06'],
      expected_action_pattern: 'safe RSS-compliant trajectory such as brake_straight',
      proving_point: 'The system prioritizes legal and safety constraints over optimization.',
      critical_evaluation_rule:
        'A prediction is correct when the dominant framework matches the subdivision expectation.',
    },
  },
  summary: {
    scenario_count: 2,
    completed_runs: 2,
    failed_runs: 0,
    completion_rate_pct: 100,
    reasoning_runtime_ready_pct: 100,
    rag_runtime_ready_pct: 100,
    top_framework: 'EF-02',
    top_framework_label: 'Deontological Rule-Based Safety',
    top_framework_percentage: 100,
    total_duration_ms: 240,
  },
  framework_distribution: [
    {
      framework_id: 'EF-02',
      framework_label: 'Deontological Rule-Based Safety',
      count: 2,
      percentage: 100,
    },
  ],
  scenario_results: [
    {
      scenario_id: 'json-example',
      scenario_label: 'JSON Example',
      subdivision_id: 'routine_rule_governed',
      subdivision_label: 'Routine Rule Governed',
      expected_framework: 'EF-02',
      status: 'success',
      duration_ms: 120,
      deterministic_best_action: 'brake_straight',
      dominant_framework: 'EF-02',
      reasoning_runtime_available: true,
      rag_runtime_available: true,
      error_code: null,
      error_message: null,
    },
    {
      scenario_id: 'json-example-2',
      scenario_label: 'JSON Example 2',
      subdivision_id: 'routine_rule_governed',
      subdivision_label: 'Routine Rule Governed',
      expected_framework: 'EF-02',
      status: 'success',
      duration_ms: 120,
      deterministic_best_action: 'brake_straight',
      dominant_framework: 'EF-02',
      reasoning_runtime_available: true,
      rag_runtime_available: true,
      error_code: null,
      error_message: null,
    },
  ],
}

function buildRunRecord(overrides?: {
  id?: string
  createdAt?: string
  status?: 'success' | 'error'
  inputPreview?: string
  bestAction?: string | null
  dominantFramework?: string | null
  ragRuntimeAvailable?: boolean
  reasoningRuntimeAvailable?: boolean
}) {
  return {
    id: overrides?.id ?? 'run-001',
    created_at: overrides?.createdAt ?? '2026-04-03T19:30:00Z',
    status: overrides?.status ?? 'success',
    input_mode_hint: 'json',
    resolved_input_mode: 'structured_json',
    submitted_kind: 'json',
    input_preview: overrides?.inputPreview ?? 'Stored scenario preview',
    model_name: 'gpt-5.4-mini',
    deterministic_best_action: overrides?.bestAction ?? 'brake_straight',
    dominant_framework: overrides?.dominantFramework ?? 'EF-02',
    rag_runtime_available: overrides?.ragRuntimeAvailable ?? true,
    reasoning_runtime_available: overrides?.reasoningRuntimeAvailable ?? true,
    error_code: null,
    error_message: null,
    replay_stage_count: 6,
  }
}

function buildRunPayload(overrides?: {
  runId?: string
  createdAt?: string
  inputPreview?: string
  bestAction?: string
  dominantFramework?: string | null
  ragRuntimeAvailable?: boolean
  reasoningRuntimeAvailable?: boolean
  rationale?: string
}) {
  const ragRuntimeAvailable = overrides?.ragRuntimeAvailable ?? true
  const reasoningRuntimeAvailable = overrides?.reasoningRuntimeAvailable ?? true
  const bestAction = overrides?.bestAction ?? 'brake_straight'
  const dominantFramework = overrides?.dominantFramework ?? (reasoningRuntimeAvailable ? 'EF-02' : null)
  const rationale = overrides?.rationale
    ?? (reasoningRuntimeAvailable
      ? 'EF-02 constrains the feasible set and EF-03 reinforces protection of the child.'
      : '')
  return {
    run: buildRunRecord({
      id: overrides?.runId,
      createdAt: overrides?.createdAt,
      inputPreview: overrides?.inputPreview,
      bestAction,
      dominantFramework,
      ragRuntimeAvailable,
      reasoningRuntimeAvailable,
    }),
    summary: {
      input_mode: 'structured_json',
      parser_warnings: [],
      violated_rules: [],
      deterministic_best_action: bestAction,
      dominant_framework: dominantFramework,
      rag_runtime_available: ragRuntimeAvailable,
      reasoning_runtime_available: reasoningRuntimeAvailable,
      reasoning_runtime_error: reasoningRuntimeAvailable ? null : 'Reasoning unavailable',
      rag_runtime_error: ragRuntimeAvailable ? null : 'RAG unavailable',
    },
    artifacts: {
      parser_result: { stage: 'parser' },
      mathematical_layer_result: { stage: 'math' },
      rag_retrieval_result: { stage: 'rag', runtime_available: ragRuntimeAvailable },
      reasoning_result: {
        stage: 'reasoning',
        runtime_available: reasoningRuntimeAvailable,
        rationale,
      },
    },
    replay: [
      {
        stage_id: 'input',
        label: 'Input',
        status: 'success',
        duration_ms: 5,
        headline: 'Received structured JSON.',
        snapshot: { input: { submitted_kind: 'json' } },
        highlight_paths: ['$.input'],
        metrics: { submitted_kind: 'json' },
      },
      {
        stage_id: 'parser',
        label: 'Parser',
        status: 'success',
        duration_ms: 15,
        headline: 'Parsed the scenario.',
        snapshot: {
          input: { submitted_kind: 'json' },
          parser_result: { environment: { road_type: 'residential' } },
        },
        highlight_paths: ['$.parser_result'],
        metrics: { input_mode: 'structured_json' },
      },
      {
        stage_id: 'math',
        label: 'Math',
        status: 'success',
        duration_ms: 20,
        headline: 'Computed risk metrics.',
        snapshot: {
          input: { submitted_kind: 'json' },
          parser_result: { environment: { road_type: 'residential' } },
          mathematical_layer_result: { best_action_by_total_risk: bestAction },
        },
        highlight_paths: ['$.mathematical_layer_result'],
        metrics: { best_action: bestAction },
      },
      {
        stage_id: 'rag',
        label: 'RAG',
        status: ragRuntimeAvailable ? 'success' : 'warning',
        duration_ms: 20,
        headline: ragRuntimeAvailable ? 'Retrieved frameworks.' : 'RAG unavailable',
        snapshot: {
          input: { submitted_kind: 'json' },
          parser_result: { environment: { road_type: 'residential' } },
          mathematical_layer_result: { best_action_by_total_risk: bestAction },
          rag_retrieval_result: { runtime_available: ragRuntimeAvailable },
        },
        highlight_paths: ['$.rag_retrieval_result'],
        metrics: { runtime_available: ragRuntimeAvailable },
      },
      {
        stage_id: 'reasoning',
        label: 'Reasoning',
        status: reasoningRuntimeAvailable ? 'success' : 'warning',
        duration_ms: 20,
        headline: reasoningRuntimeAvailable ? 'Generated ethical recommendation.' : 'Reasoning unavailable',
        snapshot: {
          input: { submitted_kind: 'json' },
          parser_result: { environment: { road_type: 'residential' } },
          mathematical_layer_result: { best_action_by_total_risk: bestAction },
          rag_retrieval_result: { runtime_available: ragRuntimeAvailable },
          reasoning_result: { runtime_available: reasoningRuntimeAvailable },
        },
        highlight_paths: ['$.reasoning_result'],
        metrics: { runtime_available: reasoningRuntimeAvailable },
      },
      {
        stage_id: 'complete',
        label: 'Complete',
        status: 'success',
        duration_ms: 25,
        headline: 'Pipeline response ready for replay.',
        snapshot: {
          input: { submitted_kind: 'json' },
          parser_result: { environment: { road_type: 'residential' } },
          mathematical_layer_result: { best_action_by_total_risk: bestAction },
          rag_retrieval_result: { runtime_available: ragRuntimeAvailable },
          reasoning_result: { runtime_available: reasoningRuntimeAvailable },
          summary: { dominant_framework: dominantFramework },
        },
        highlight_paths: ['$.summary'],
        metrics: { deterministic_best_action: bestAction },
      },
    ],
  }
}

const latestRunPayload = buildRunPayload({
  runId: 'run-001',
  createdAt: '2026-04-03T19:30:00Z',
  inputPreview: 'Current stored JSON scenario',
  bestAction: 'brake_straight',
  dominantFramework: 'EF-02',
})

const olderRunPayload = buildRunPayload({
  runId: 'run-000',
  createdAt: '2026-04-02T09:15:00Z',
  inputPreview: 'Earlier crossing scenario with a different outcome',
  bestAction: 'swerve_left',
  dominantFramework: 'EF-03',
})

const runHistoryPayload = {
  runs: [latestRunPayload.run, olderRunPayload.run],
  total_runs: 2,
  success_runs: 2,
  failed_runs: 0,
}

function mockFetch(runPayload = latestRunPayload) {
  return vi.fn(async (input: string | URL | RequestInfo, init?: RequestInit) => {
    const url = String(input)
    if (url.endsWith('/api/v1/health')) {
      return new Response(JSON.stringify(healthPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/examples')) {
      return new Response(JSON.stringify(catalogPayload), { status: 200 })
    }
    if (url.includes('/api/v1/scenario/runs?')) {
      return new Response(JSON.stringify(runHistoryPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/scenario/runs/run-001')) {
      return new Response(JSON.stringify(latestRunPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/scenario/runs/run-000')) {
      return new Response(JSON.stringify(olderRunPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/scenario/run') && init?.method === 'POST') {
      return new Response(JSON.stringify(runPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/scenario/subdivision/run') && init?.method === 'POST') {
      return new Response(JSON.stringify(subdivisionPayload), { status: 200 })
    }
    return new Response('{}', { status: 404 })
  })
}

describe('App', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('replays stages in order after the backend response arrives', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const user = userEvent.setup()

    render(<App />)

    const runButton = await screen.findByRole('button', { name: 'Run Scenario' })
    await user.click(runButton)

    expect(await screen.findByText('Received structured JSON.')).toBeInTheDocument()

    await waitFor(
      () => {
        expect(screen.getByText('Pipeline response ready for replay.')).toBeInTheDocument()
      },
      { timeout: 4000 },
    )
  })

  it('shows changed JSON paths while replaying the pipeline', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const user = userEvent.setup()

    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run Scenario' }))

    await waitFor(
      () => {
        expect(within(screen.getByTestId('stage-timeline')).getAllByRole('button')).toHaveLength(6)
        expect(screen.getByText('Settled')).toBeInTheDocument()
      },
      { timeout: 4000 },
    )

    const parserStage = within(screen.getByTestId('stage-timeline')).getByRole('button', { name: /Parser/i })
    await user.click(parserStage)

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Parser' })).toBeInTheDocument()
      expect(screen.getByTestId('highlight-strip')).toHaveTextContent(/parser_result/)
    })
  })

  it('renders degraded runtime warnings without failing the whole UI', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch(buildRunPayload({ ragRuntimeAvailable: false, reasoningRuntimeAvailable: false })),
    )
    const user = userEvent.setup()

    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run Scenario' }))

    await waitFor(() => {
      expect(screen.getByText(/RAG degraded:/)).toBeInTheDocument()
      expect(screen.getByText(/Reasoning degraded:/)).toBeInTheDocument()
    })
  })

  it('shows the backend rationale beside the dominant framework', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const user = userEvent.setup()

    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run Scenario' }))

    await waitFor(() => {
      expect(screen.getByText('Framework Rationale')).toBeInTheDocument()
      expect(
        screen.getByText('EF-02 constrains the feasible set and EF-03 reinforces protection of the child.'),
      ).toBeInTheDocument()
    })
  })

  it('blocks submission when the JSON editor contains invalid JSON', async () => {
    const fetchMock = mockFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)

    const editor = await screen.findByLabelText('Structured JSON editor')
    fireEvent.change(editor, { target: { value: '{ invalid json' } })
    await user.click(screen.getByRole('button', { name: 'Run Scenario' }))

    expect(await screen.findByText('Enter valid JSON before running the scenario.')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('renders subdivision metadata and the framework graph after a batch run', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const user = userEvent.setup()

    render(<App />)

    expect(await screen.findAllByText('Routine Rule Governed')).toHaveLength(2)

    await user.click(await screen.findByRole('button', { name: 'Run Subdivision' }))

    await waitFor(() => {
      expect(screen.getByTestId('framework-graph')).toBeInTheDocument()
      expect(screen.getAllByText('Deontological Rule-Based Safety')).toHaveLength(2)
      expect(screen.getByText('100.0% (2)')).toBeInTheDocument()
      const expectationPanel = screen.getByTestId('subdivision-expectation')
      expect(expectationPanel).toBeInTheDocument()
      expect(expectationPanel).toHaveTextContent('Rule compliance')
      expect(expectationPanel).toHaveTextContent(/dominant framework matches the subdivision expectation/i)
    })
  })

  it('shows stored run history, compares a prior run, and loads it back into the replay view', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const user = userEvent.setup()

    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Run Scenario' }))

    const historyList = await screen.findByTestId('run-history-list')
    expect(within(historyList).getByText('Current stored JSON scenario')).toBeInTheDocument()
    expect(within(historyList).getByText('Earlier crossing scenario with a different outcome')).toBeInTheDocument()

    const compareButtons = within(historyList).getAllByRole('button', { name: 'Compare' })
    await user.click(compareButtons[1])

    expect(await screen.findByText(/Deterministic action changed from swerve_left to brake_straight/i)).toBeInTheDocument()
    expect(screen.getByText(/Dominant framework changed from EF-03 to EF-02/i)).toBeInTheDocument()
    expect(screen.getAllByText('swerve_left')).toHaveLength(2)

    const loadButtons = within(historyList).getAllByRole('button', { name: 'Load' })
    await user.click(loadButtons[1])

    await waitFor(() => {
      expect(screen.getAllByText('swerve_left').length).toBeGreaterThan(2)
      expect(screen.getAllByText('EF-03').length).toBeGreaterThan(2)
    })
  })
})
