import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const fakes = vi.hoisted(() => ({
  config: null as any,
  bindConversation: vi.fn(() => true),
  finishApiWorker: vi.fn(() => ({ info: { id: 'worker-1' }, report: { id: 'report-1' }, repeat: false })),
  failAgent: vi.fn(),
  persistCriticalSwarmNow: vi.fn(async () => true),
  invoke: vi.fn(async (name: string, args: unknown) => ({ content: [{ type: 'text', text: `${name.toUpperCase()}_OK ${JSON.stringify(args)}` }] }))
}));

vi.mock('../src/main/config.js', () => ({
  getConfig: () => fakes.config,
  effectiveCapabilities: () => ({ read: true, find: true, edit: true, command: true })
}));
vi.mock('../src/main/goal.js', () => ({
  goalEndpoint: () => ({ kind: 'custom' }),
  goalProviderKey: async () => 'test-api-key',
  resolveGoalBaseUrl: () => 'https://provider.example/v1'
}));
vi.mock('../src/main/logger.js', () => ({ logInfo: vi.fn(), logWarn: vi.fn() }));
vi.mock('../src/main/agents.js', () => ({
  bindConversation: fakes.bindConversation,
  failAgent: fakes.failAgent,
  finishApiWorker: fakes.finishApiWorker,
  persistCriticalSwarmNow: fakes.persistCriticalSwarmNow
}));
vi.mock('../src/main/skill-access.js', () => ({ withManagedSkills: (context: unknown) => context }));
vi.mock('../src/main/mcp/call-context.js', () => ({ emptyEvidence: () => ({}) }));
vi.mock('../src/main/mcp/kernel.js', () => ({
  createRegistrar: (_session: unknown, _context: unknown, _surface: unknown,
    onTool: (name: string, config: object) => void) => ({
    register: (name: string, config: object) => onTool(name, config),
    invokeNested: (...args: unknown[]) => fakes.invoke(...args as [string, unknown])
  })
}));
vi.mock('../src/main/mcp/tools-core.js', async () => {
  const { z } = await import('zod');
  return {
    registerCoreTools: (registrar: { register: (name: string, config: object) => void }) => {
      for (const name of ['read', 'find', 'apply_patch', 'exec_command', 'write_stdin', 'browser_tabs', 'computer', 'agents']) {
        registrar.register(name, { description: `${name} test tool`, inputSchema: z.object({}) });
      }
    }
  };
});

const { startApiWorker } = await import('../src/main/api-worker.js');
const realFetch = globalThis.fetch;

function setupConfig(): void {
  fakes.config = {
    multiAgent: { enabled: true, workerBackend: 'api' },
    goal: { model: 'glm-5.3-flash', reasoning: 'default' },
    roots: [{ name: 'work', path: 'C:/approved' }],
    readOnly: false,
    ui: { privacyScreenshots: true },
    sessions: { record: true }
  };
}

function worker() {
  return { runId: 'run-1', primeConversationId: null, id: 'worker-1', task: 'Inspect the approved workspace.', model: null, reasoningEffort: null };
}

async function waitForFinish(): Promise<void> {
  await vi.waitFor(() => expect(fakes.finishApiWorker).toHaveBeenCalledTimes(1), { timeout: 2_000 });
}

beforeEach(() => {
  setupConfig();
  fakes.bindConversation.mockClear().mockReturnValue(true);
  fakes.finishApiWorker.mockClear().mockReturnValue({ info: { id: 'worker-1' }, report: { id: 'report-1' }, repeat: false });
  fakes.failAgent.mockClear();
  fakes.persistCriticalSwarmNow.mockClear().mockResolvedValue(true);
  fakes.invoke.mockClear().mockImplementation(async (name: string, args: unknown) => ({
    content: [{ type: 'text', text: `${name.toUpperCase()}_OK ${JSON.stringify(args)}` }]
  }));
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe('API-backed workers', () => {
  it('uses the configured model, exposes only local Core tools and publishes a one-shot result', async () => {
    const requests: Array<{ url: string; init: RequestInit; body: any }> = [];
    globalThis.fetch = (async (input, init) => {
      const body = JSON.parse(String(init?.body));
      requests.push({ url: String(input), init: init!, body });
      return requests.length === 1
        ? Response.json({ choices: [{ message: { tool_calls: [
          { id: 'call-read', type: 'function', function: { name: 'read', arguments: '{"path":"/workspace/readme.txt"}' } },
          { id: 'call-exec', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"Get-ChildItem"}' } }
        ] } }] })
        : Response.json({ choices: [{ message: { content: 'RESULT: workspace inspected' } }] });
    }) as typeof fetch;
    fakes.invoke.mockImplementationOnce(async (name: string, args: unknown) => {
      // A settings save during tool execution must affect the next worker, not this run.
      fakes.config.goal.model = 'another-model';
      return { content: [{ type: 'text', text: `${name.toUpperCase()}_OK ${JSON.stringify(args)}` }] };
    });

    startApiWorker(worker());
    await waitForFinish();

    expect(fakes.bindConversation).toHaveBeenCalledWith('worker-1', 'api-worker:run-1:worker-1', 'run-1');
    expect(requests).toHaveLength(2);
    expect(requests[0]?.url).toBe('https://provider.example/v1/chat/completions');
    expect(requests[0]?.body.model).toBe('glm-5.3-flash');
    expect(requests[1]?.body.model).toBe('glm-5.3-flash');
    expect(new Headers(requests[0]?.init.headers).get('authorization')).toBe('Bearer test-api-key');
    expect(requests[0]?.body.tools.map((tool: any) => tool.function.name)).toEqual([
      'read', 'find', 'apply_patch', 'exec_command', 'write_stdin'
    ]);
    expect(fakes.invoke).toHaveBeenCalledWith('read', { path: '/workspace/readme.txt' }, expect.objectContaining({
      caller: expect.objectContaining({ conversationId: 'api-worker:run-1:worker-1', runId: 'run-1' })
    }));
    expect(fakes.invoke).toHaveBeenCalledWith('exec_command', { cmd: 'Get-ChildItem' }, expect.objectContaining({
      caller: expect.objectContaining({ conversationId: 'api-worker:run-1:worker-1', runId: 'run-1' })
    }));
    expect(requests[1]?.body.messages.slice(-2)).toEqual(expect.arrayContaining([
      expect.objectContaining({ role: 'tool', tool_call_id: 'call-read', content: expect.stringContaining('READ_OK') }),
      expect.objectContaining({ role: 'tool', tool_call_id: 'call-exec', content: expect.stringContaining('EXEC_COMMAND_OK') })
    ]));
    expect(fakes.finishApiWorker).toHaveBeenCalledWith('worker-1', 'RESULT: workspace inspected', 'run-1');
    expect(fakes.persistCriticalSwarmNow).toHaveBeenCalled();
    expect(fakes.failAgent).not.toHaveBeenCalled();
  });
});
