/**
 * API-backed sub-agent workers.
 *
 * This is deliberately a second transport for the existing broker, not a replacement for it.
 * The broker still owns worker slots, workspaces, reports and the prime/worker topology. The
 * only thing this file replaces is the browser-backed ChatGPT conversation used to execute a
 * worker assignment. ChatGPT workers remain available by switching workerBackend back to
 * `chatgpt`.
 */

import { randomUUID } from 'node:crypto';
import { getConfig, effectiveCapabilities } from './config.js';
import { goalEndpoint, goalProviderKey, resolveGoalBaseUrl } from './goal.js';
import { logInfo, logWarn } from './logger.js';
import {
  bindConversation,
  failAgent,
  finishApiWorker,
  persistCriticalSwarmNow,
  type WorkerSpawn
} from './agents.js';
import { withManagedSkills } from './skill-access.js';
import { createRegistrar, type ToolContext, type ToolResult } from './mcp/kernel.js';
import { registerCoreTools } from './mcp/tools-core.js';
import { toolSchemaJson } from './mcp/tool-declarations.js';
import { emptyEvidence, type CallContext } from './mcp/call-context.js';

type ApiMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: ApiToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string };

interface ApiToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface ApiCompletion {
  choices?: Array<{
    message?: {
      content?: unknown;
      tool_calls?: unknown;
    };
  }>;
  error?: { message?: unknown };
}

interface ApiWorkerHandle {
  controller: AbortController;
  runId: string;
  id: string;
}

interface ApiWorkerProvider {
  baseUrl: string;
  kind: string;
  key: string | null;
  model: string;
  reasoning: string;
}

const running = new Map<string, ApiWorkerHandle>();
const MAX_ROUNDS = 48;
const REQUEST_TIMEOUT_MS = 180_000;
const PROVIDER_ATTEMPTS = 2;
const PROVIDER_RETRY_BACKOFF_MS = 4_000;
const MAX_TOOL_RESULT_CHARS = 64_000;
const MAX_FINAL_CHARS = 4_000;
const API_TOOL_NAMES = new Set(['read', 'find', 'apply_patch', 'exec_command', 'write_stdin']);
const RETRYABLE_HTTP_STATUS = new Set([408, 429, 500, 502, 503, 504]);

function workerKey(runId: string, id: string): string {
  return `${runId}:${id}`;
}

function apiConversationId(runId: string, id: string): string {
  return `api-worker:${runId}:${id}`;
}

function textContent(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.map((part) => {
    if (!part || typeof part !== 'object') return '';
    const row = part as { type?: unknown; text?: unknown };
    return row.type === 'text' && typeof row.text === 'string' ? row.text : '';
  }).join('');
}

function parseToolCalls(value: unknown): ApiToolCall[] {
  if (!Array.isArray(value)) return [];
  const out: ApiToolCall[] = [];
  for (const row of value) {
    if (!row || typeof row !== 'object') continue;
    const call = row as { id?: unknown; type?: unknown; function?: unknown };
    if (typeof call.id !== 'string' || call.type !== 'function' || !call.function || typeof call.function !== 'object') continue;
    const fn = call.function as { name?: unknown; arguments?: unknown };
    if (typeof fn.name !== 'string' || typeof fn.arguments !== 'string') continue;
    out.push({ id: call.id, type: 'function', function: { name: fn.name, arguments: fn.arguments } });
  }
  return out;
}

function boundedToolText(result: ToolResult): string {
  const text = result.content.map((part) =>
    part.type === 'text' ? part.text : `[${part.mimeType} image omitted from API worker tool transcript]`
  ).join('\n');
  const prefix = result.isError ? 'TOOL_ERROR\n' : '';
  if (prefix.length + text.length <= MAX_TOOL_RESULT_CHARS) return prefix + text;
  return prefix + text.slice(0, MAX_TOOL_RESULT_CHARS - prefix.length - 80) + '\n[tool result truncated by API worker]';
}

function workerContext(): ToolContext {
  const config = getConfig();
  return withManagedSkills({
    roots: config.roots,
    caps: effectiveCapabilities(config),
    readOnly: config.readOnly,
    privacyScreenshots: config.ui.privacyScreenshots,
    sessionTools: config.sessions.record,
    agentTools: config.multiAgent.enabled
  });
}

function parentContext(worker: WorkerSpawn, conversationId: string): CallContext {
  const transportKey = `api-worker:${worker.runId}:${worker.id}`;
  return {
    publication: { completedAt: null, failed: false },
    startedAt: Date.now(),
    transportKey,
    agent: worker.id,
    allowUnattributed: true,
    caller: {
      transportKey,
      requestId: `apiw_${randomUUID().replaceAll('-', '')}`,
      conversationId,
      sessionId: null,
      runId: worker.runId
    },
    outcome: null,
    evidence: emptyEvidence()
  };
}

function createToolset(worker: WorkerSpawn, conversationId: string): {
  definitions: Array<{ type: 'function'; function: { name: string; description: string; parameters: object } }>;
  call: (name: string, args: unknown) => Promise<ToolResult>;
} {
  const declarations: Array<{ name: string; description: string; parameters: object }> = [];
  const reg = createRegistrar(null, workerContext(), 'core', (name, config) => {
    if (!API_TOOL_NAMES.has(name)) return;
    const raw = toolSchemaJson(config.inputSchema) as Record<string, unknown>;
    const { $schema: _schema, ...parameters } = raw;
    declarations.push({ name, description: config.description, parameters });
  });
  registerCoreTools(reg);
  return {
    definitions: declarations.map((tool) => ({ type: 'function' as const, function: tool })),
    call: (name, args) => {
      if (!API_TOOL_NAMES.has(name)) {
        return Promise.resolve({ content: [{ type: 'text', text: `UNKNOWN_TOOL: ${name} is not exposed to API workers.` }], isError: true });
      }
      return reg.invokeNested(name, args, parentContext(worker, conversationId));
    }
  };
}

function retryDelay(signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason ?? new Error('API worker cancelled'));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(done, PROVIDER_RETRY_BACKOFF_MS);
    function done(): void {
      signal.removeEventListener('abort', aborted);
      resolve();
    }
    function aborted(): void {
      clearTimeout(timer);
      signal.removeEventListener('abort', aborted);
      reject(signal.reason ?? new Error('API worker cancelled'));
    }
    signal.addEventListener('abort', aborted, { once: true });
  });
}

async function configuredProvider(): Promise<ApiWorkerProvider> {
  const config = getConfig();
  const endpoint = goalEndpoint();
  const key = await goalProviderKey(endpoint.kind);
  if (endpoint.kind === 'openrouter' && !key) throw new Error('No API key is configured for the selected worker provider.');
  return {
    baseUrl: resolveGoalBaseUrl(endpoint),
    kind: endpoint.kind,
    key,
    model: config.goal.model,
    reasoning: config.goal.reasoning
  };
}

async function providerRequestOnce(messages: ApiMessage[], tools: ReturnType<typeof createToolset>['definitions'], signal: AbortSignal,
  provider: ApiWorkerProvider): Promise<{ content: string; toolCalls: ApiToolCall[] }> {
  const body: Record<string, unknown> = {
    model: provider.model,
    messages,
    tools,
    tool_choice: 'auto',
    stream: false
  };
  if (provider.kind === 'custom' && provider.reasoning !== 'default') body['reasoning_effort'] = provider.reasoning;
  if (provider.kind !== 'custom') {
    body['reasoning'] = {
      ...(provider.reasoning === 'default' ? {} : { effort: provider.reasoning }),
      exclude: true
    };
    body['provider'] = { require_parameters: true };
  }
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const response = await fetch(`${provider.baseUrl}/chat/completions`, {
    method: 'POST',
    redirect: 'error',
    headers: {
      ...(provider.key ? { authorization: `Bearer ${provider.key}` } : {}),
      'content-type': 'application/json',
      ...(provider.kind === 'openrouter' ? {
        'HTTP-Referer': 'https://github.com/chat-on-steroids',
        'X-Title': 'Chat On Steroids'
      } : {})
    },
    body: JSON.stringify(body),
    signal: AbortSignal.any([signal, timeout])
  });
  const raw = (await response.text()).slice(0, 1024 * 1024);
  let parsed: ApiCompletion;
  try { parsed = raw ? JSON.parse(raw) as ApiCompletion : {}; }
  catch {
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${raw.slice(0, 800) || response.statusText}`);
    throw new Error(`Worker provider returned non-JSON HTTP ${response.status}.`);
  }
  if (!response.ok) {
    const detail = parsed.error && typeof parsed.error.message === 'string' ? parsed.error.message : raw.slice(0, 800);
    throw new Error(`HTTP ${response.status}: ${detail || response.statusText}`);
  }
  const message = parsed.choices?.[0]?.message;
  if (!message) throw new Error('Worker provider returned no assistant message.');
  return { content: textContent(message.content), toolCalls: parseToolCalls(message.tool_calls) };
}

async function providerRequest(messages: ApiMessage[], tools: ReturnType<typeof createToolset>['definitions'], signal: AbortSignal,
  provider: ApiWorkerProvider): Promise<{ content: string; toolCalls: ApiToolCall[] }> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= PROVIDER_ATTEMPTS; attempt++) {
    signal.throwIfAborted();
    try {
      return await providerRequestOnce(messages, tools, signal, provider);
    } catch (error) {
      if (signal.aborted) throw signal.reason ?? error;
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const status = /^HTTP (\d{3}):/.exec(message)?.[1];
      const timedOut = /timeout|timed out|aborted due to timeout/i.test(message) ||
        (error instanceof DOMException && error.name === 'TimeoutError');
      const transientHttp = !!status && RETRYABLE_HTTP_STATUS.has(Number(status));
      const transientNetwork = error instanceof TypeError;
      if (attempt >= PROVIDER_ATTEMPTS || (!timedOut && !transientHttp && !transientNetwork)) throw error;
      logWarn(`multi-agent: provider request attempt ${attempt}/${PROVIDER_ATTEMPTS} failed transiently (${message.slice(0, 240)}); retrying in ${PROVIDER_RETRY_BACKOFF_MS} ms`);
      await retryDelay(signal);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError ?? 'Worker provider request failed.'));
}

async function runWorker(worker: WorkerSpawn, signal: AbortSignal): Promise<string> {
  const config = getConfig();
  if (!config.multiAgent.enabled) throw new Error('Multi-agent mode is disabled.');
  const conversationId = apiConversationId(worker.runId, worker.id);
  if (!bindConversation(worker.id, conversationId, worker.runId)) {
    throw new Error('The broker could not bind the API worker to its worker slot.');
  }
  await persistCriticalSwarmNow();
  const toolset = createToolset(worker, conversationId);
  if (!toolset.definitions.length) throw new Error('No Core tools are enabled for API workers.');
  // A provider/model change applies to the next worker, not halfway through this assignment.
  // The key stays in memory for this run and is never written to the worker snapshot or logs.
  const provider = await configuredProvider();
  const messages: ApiMessage[] = [
    {
      role: 'system',
      content:
        'You are a Chat On Steroids worker agent. Complete the assigned job using the provided local tools. ' +
        'Work only inside the permissions and approved folders enforced by the tools. Do not change network, proxy, VPN, tunnel, firewall, routing, or ChatGPT connector settings unless the task explicitly asks for that. ' +
        'Do not invent tool results. Keep working through errors when a safe repair is available. When the assignment is complete, stop calling tools and return a concise handoff under RESULT / CHANGES / VALIDATION / BLOCKERS. '
    },
    { role: 'user', content: worker.task }
  ];
  logInfo(`multi-agent: ${worker.id} API worker started with ${provider.model}`);
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    signal.throwIfAborted();
    if (!getConfig().multiAgent.enabled) throw new Error('Multi-agent mode was disabled while the API worker was running.');
    const answer = await providerRequest(messages, toolset.definitions, signal, provider);
    if (!answer.toolCalls.length) {
      const final = answer.content.trim();
      if (!final) throw new Error('Worker provider returned an empty final answer.');
      return final.slice(0, MAX_FINAL_CHARS);
    }
    messages.push({ role: 'assistant', content: answer.content || null, tool_calls: answer.toolCalls });
    for (const call of answer.toolCalls) {
      signal.throwIfAborted();
      let args: unknown;
      try { args = call.function.arguments.trim() ? JSON.parse(call.function.arguments) : {}; }
      catch {
        messages.push({ role: 'tool', tool_call_id: call.id, content: `INVALID_ARGUMENTS: ${call.function.arguments.slice(0, 1000)}` });
        continue;
      }
      const result = await toolset.call(call.function.name, args);
      messages.push({ role: 'tool', tool_call_id: call.id, content: boundedToolText(result) });
    }
  }
  throw new Error(`API worker exceeded its ${MAX_ROUNDS}-round safety limit.`);
}

/** Start exactly one API execution for a broker worker slot. Safe to call on replay. */
export function startApiWorker(worker: WorkerSpawn): void {
  const key = workerKey(worker.runId, worker.id);
  if (running.has(key)) return;
  const controller = new AbortController();
  running.set(key, { controller, runId: worker.runId, id: worker.id });
  void (async () => {
    try {
      const report = await runWorker(worker, controller.signal);
      if (!finishApiWorker(worker.id, report, worker.runId)) return;
      await persistCriticalSwarmNow();
    } catch (error) {
      if (controller.signal.aborted) return;
      const reason = error instanceof Error ? error.message : String(error);
      logWarn(`multi-agent: ${worker.id} API worker failed — ${reason}`);
      failAgent(
        worker.id,
        reason,
        `[${worker.id} failed] API worker failed: ${reason}. ChatGPT worker mode is still available as a fallback.`,
        { revivable: false },
        worker.runId
      );
      await persistCriticalSwarmNow().catch(() => false);
    } finally {
      running.delete(key);
    }
  })();
}

/** Stop in-process API work when its owning broker run is explicitly ended. */
export function cancelApiWorkersForRun(runId: string): void {
  for (const [key, worker] of running) {
    if (worker.runId !== runId) continue;
    worker.controller.abort(new Error('worker_run_ended'));
    running.delete(key);
  }
}
