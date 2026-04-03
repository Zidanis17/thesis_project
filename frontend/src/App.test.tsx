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

const examplesPayload = {
  examples: [
    {
      id: 'json-example',
      label: 'JSON Example',
      mode: 'json',
      value: {
        ego_vehicle: { speed_kmh: 50 },
        environment: { road_type: 'residential' },
      },
    },
    {
      id: 'text-example',
      label: 'Text Example',
      mode: 'text',
      value: 'A child is crossing the road.',
    },
  ],
}

function buildRunPayload(overrides?: {
  ragRuntimeAvailable?: boolean
  reasoningRuntimeAvailable?: boolean
}) {
  const ragRuntimeAvailable = overrides?.ragRuntimeAvailable ?? true
  const reasoningRuntimeAvailable = overrides?.reasoningRuntimeAvailable ?? true
  return {
    summary: {
      input_mode: 'structured_json',
      parser_warnings: [],
      violated_rules: [],
      deterministic_best_action: 'brake_straight',
      recommended_action: reasoningRuntimeAvailable ? 'brake_straight' : null,
      dominant_framework: reasoningRuntimeAvailable ? 'EF-02' : null,
      rag_runtime_available: ragRuntimeAvailable,
      reasoning_runtime_available: reasoningRuntimeAvailable,
      reasoning_runtime_error: reasoningRuntimeAvailable ? null : 'Reasoning unavailable',
      rag_runtime_error: ragRuntimeAvailable ? null : 'RAG unavailable',
    },
    artifacts: {
      parser_result: { stage: 'parser' },
      mathematical_layer_result: { stage: 'math' },
      rag_retrieval_result: { stage: 'rag', runtime_available: ragRuntimeAvailable },
      reasoning_result: { stage: 'reasoning', runtime_available: reasoningRuntimeAvailable },
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
          mathematical_layer_result: { best_action_by_total_risk: 'brake_straight' },
        },
        highlight_paths: ['$.mathematical_layer_result'],
        metrics: { best_action: 'brake_straight' },
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
          mathematical_layer_result: { best_action_by_total_risk: 'brake_straight' },
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
          mathematical_layer_result: { best_action_by_total_risk: 'brake_straight' },
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
          mathematical_layer_result: { best_action_by_total_risk: 'brake_straight' },
          rag_retrieval_result: { runtime_available: ragRuntimeAvailable },
          reasoning_result: { runtime_available: reasoningRuntimeAvailable },
          summary: { recommended_action: reasoningRuntimeAvailable ? 'brake_straight' : null },
        },
        highlight_paths: ['$.summary'],
        metrics: { deterministic_best_action: 'brake_straight' },
      },
    ],
  }
}

function mockFetch(runPayload = buildRunPayload()) {
  return vi.fn(async (input: string | URL | RequestInfo, init?: RequestInit) => {
    const url = String(input)
    if (url.endsWith('/api/v1/health')) {
      return new Response(JSON.stringify(healthPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/examples')) {
      return new Response(JSON.stringify(examplesPayload), { status: 200 })
    }
    if (url.endsWith('/api/v1/scenario/run') && init?.method === 'POST') {
      return new Response(JSON.stringify(runPayload), { status: 200 })
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

  it('blocks submission when the JSON editor contains invalid JSON', async () => {
    const fetchMock = mockFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)

    const editor = await screen.findByLabelText('Structured JSON editor')
    fireEvent.change(editor, { target: { value: '{ invalid json' } })
    await user.click(screen.getByRole('button', { name: 'Run Scenario' }))

    expect(await screen.findByText('Enter valid JSON before running the scenario.')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
