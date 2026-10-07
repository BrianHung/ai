// Long-running bridge that runs inside a sandbox alongside the `claude` CLI.
// The generic transport — WebSocket server, token auth, single-flight
// reconnect, the in-memory event log + `seq`, resume replay, and the
// lifecycle/meta files — lives in the shared `@ai-sdk/harness/bridge` runtime.
// This file supplies only the Claude-specific turn driver.

import {
  runBridge,
  type BridgeEvent,
  type BridgeTurn,
  type Experimental_BridgeUserMessage,
  type Experimental_BridgeUserMessageQueue,
} from '@ai-sdk/harness/bridge';
import {
  createCompactionLatch,
  type CompactionLatch,
} from './compaction-latch';
import { createPushableInput, type PushableInput } from './pushable-input';
import type { StartMessage } from '../claude-code-bridge-protocol';
import { randomUUID } from 'node:crypto';
import { argv, env as procEnv, stdout } from 'node:process';

/*
 * CONSTRAINT — the third-party imports below are NEVER bundled into the
 * compiled `bridge/index.mjs`. They are declared `external` in
 * tsdown.config.ts and resolved at runtime from the node_modules that this
 * bridge installs *inside the sandbox* from `src/bridge/package.json` (and
 * its pinned `pnpm-lock.yaml`). That bridge package.json — NOT this host
 * package — is the single source of truth for these packages and their
 * versions; the published `@ai-sdk/harness-claude-code` package does not
 * provide them at runtime.
 *
 * When adding or changing a third-party import here you MUST keep all three
 * in sync, or the bridge will either get the dependency bundled in or fail
 * to resolve it in the sandbox:
 *   1. the import statement below,
 *   2. the `external` array in tsdown.config.ts, and
 *   3. the dependency entry in `src/bridge/package.json`.
 */
import * as claudeAgentSdk from '@anthropic-ai/claude-agent-sdk';
import type {
  HookCallback,
  HookJSONOutput,
} from '@anthropic-ai/claude-agent-sdk';
import * as mcpServerModule from '@modelcontextprotocol/sdk/server/mcp.js';
import { createClaudeCodeSystemPrompt } from './claude-code-system-prompt';
import { toClaudeSkillsOption } from './claude-skills-option';
import {
  createClaudeStreamEventState,
  createEmitStreamEvent,
  defaultUsage,
  emitFinishStep,
  finishApprovalStep,
  isExternalMcpTool,
  mapUsage,
  resetClaudeStreamEventTurnState,
  type ClaudeMessage,
  type ClaudeStreamEventState,
} from './create-emit-stream-event';
import { jsonSchemaToZodObject } from './json-schema-to-zod';
import {
  resolveInactiveNativeTools,
  resolveNativeTools,
} from './tool-filtering';
import {
  claudeCodeQuestionKey,
  toClaudeCodeQuestionResult,
  toHarnessQuestionsInput,
} from './question-tool';

/*
 * Native Claude Code tool name → cross-harness common name. Tools outside this
 * map (e.g. `WebFetch`, `NotebookEdit`) have no common equivalent; their
 * native name is forwarded as-is on `tool-call` events.
 */
type CommonBuiltinToolName =
  | 'read'
  | 'write'
  | 'edit'
  | 'bash'
  | 'glob'
  | 'grep'
  | 'webSearch'
  | 'askUserQuestions';

const NATIVE_TO_COMMON: Readonly<Record<string, CommonBuiltinToolName>> = {
  Read: 'read',
  Write: 'write',
  Edit: 'edit',
  Bash: 'bash',
  Glob: 'glob',
  Grep: 'grep',
  WebSearch: 'webSearch',
  AskUserQuestion: 'askUserQuestions',
};

const NATIVE_TOOL_KINDS: Readonly<
  Record<string, 'readonly' | 'edit' | 'bash'>
> = {
  Read: 'readonly',
  Glob: 'readonly',
  Grep: 'readonly',
  WebSearch: 'readonly',
  WebFetch: 'readonly',
  TaskGet: 'readonly',
  TaskList: 'readonly',
  TaskOutput: 'readonly',
  ListMcpResources: 'readonly',
  ReadMcpResource: 'readonly',
  Write: 'edit',
  Edit: 'edit',
  NotebookEdit: 'edit',
  TodoWrite: 'edit',
  TaskCreate: 'edit',
  TaskUpdate: 'edit',
  TaskStop: 'edit',
  EnterWorktree: 'edit',
  ExitWorktree: 'edit',
  ExitPlanMode: 'edit',
  Skill: 'readonly',
  AskUserQuestion: 'readonly',
  ToolSearch: 'readonly',
  Bash: 'bash',
  Monitor: 'bash',
};

function toCommonName(nativeName: string): CommonBuiltinToolName | string {
  return NATIVE_TO_COMMON[nativeName] ?? nativeName;
}

const args = parseArgs(argv.slice(2));
const workdir = args.workdir;
const bridgeStateDir = args.bridgeStateDir;
if (!workdir) {
  emitFatal('Missing --workdir argument.');
}
if (!bridgeStateDir) {
  emitFatal('Missing --bridge-state-dir argument.');
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const claudeSdk = claudeAgentSdk as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mcpModule = mcpServerModule as any;

/**
 * The Claude session id most recently reported by the SDK, captured from the
 * message stream. Every turn in this bridge process may fork a new id (the
 * SDK's `continue`/`resume` create a new session linked to the previous one),
 * so the latest observation is the one a later resume must name.
 */
let lastClaudeSessionId: string | undefined;

/**
 * The long-lived query that `persistentQuery` turns share: one Claude Code
 * process for the bridge, fed one user message per turn, so background shells
 * and subagents keep running between turns. Undefined until the first
 * persistent turn, and again once that query ends or is replaced.
 */
let liveQuery: LiveQuery | undefined;

await runBridge<StartMessage>({
  bridgeType: 'claude-code',
  bridgeStateDir,
  onStart: (start, turn) => {
    if (start.persistentQuery === true) {
      return runPersistentTurn(start, turn);
    }
    // A per-turn query must not share the conversation with a live one.
    closeLiveQuery();
    return runTurn(start, turn);
  },
  // Claude Code's conversation state lives in the runtime's own store, keyed
  // by working directory. The resume payload names the exact conversation so a
  // later resume does not have to fall back to "most recent in this workdir".
  onStop: () => {
    closeLiveQuery();
    return lastClaudeSessionId == null
      ? {}
      : { claudeSessionId: lastClaudeSessionId };
  },
  onDestroy: () => closeLiveQuery(),
});

type Emit = (msg: Record<string, unknown>) => void;

function createPermissionOptions(input: {
  start: StartMessage;
  inactiveNativeTools: readonly string[];
  turn: Pick<BridgeTurn, 'requestToolApproval'>;
  emit: Emit;
  finishApprovalStep: (approvalId: string) => void;
  nativeToolCallNames: Map<string, string>;
  approvalRequestedToolUseIds: Set<string>;
}): Record<string, unknown> {
  const permissionMode = input.start.permissionMode ?? 'allow-all';
  const inactiveNativeTools = new Set(input.inactiveNativeTools);
  const permissionSettings = createPermissionSettings({
    permissionMode,
    inactiveNativeTools,
  });

  const baseOptions = {
    permissionMode:
      permissionMode === 'allow-all'
        ? 'bypassPermissions'
        : permissionMode === 'allow-edits'
          ? 'acceptEdits'
          : 'default',
    allowDangerouslySkipPermissions: permissionMode === 'allow-all',
    ...(permissionSettings ? { settings: permissionSettings } : {}),
  };

  if (permissionMode === 'allow-all') {
    return {
      ...baseOptions,
      /*
       * Claude Code exposes AskUserQuestion in headless SDK sessions only
       * when a permission prompt tool is configured. The stdio prompt tool
       * preserves that tool surface without supplying the canUseTool callback
       * that bypassPermissions guarantees it will never invoke.
       */
      permissionPromptToolName: 'stdio',
    };
  }

  return {
    ...baseOptions,
    canUseTool: async (
      toolName: string,
      toolInput: Record<string, unknown>,
      options: { toolUseID: string },
    ) => {
      if (toolName.startsWith('mcp__harness-tools__')) {
        return { behavior: 'allow', updatedInput: toolInput };
      }
      if (
        !inactiveNativeTools.has(toolName) &&
        !nativeToolRequiresApproval({
          nativeName: toolName,
          permissionMode,
        })
      ) {
        return { behavior: 'allow', updatedInput: toolInput };
      }

      const approvalId = options.toolUseID;
      input.approvalRequestedToolUseIds.add(approvalId);
      input.nativeToolCallNames.set(approvalId, toolName);
      input.emit({
        type: 'tool-call',
        toolCallId: approvalId,
        toolName: toCommonName(toolName),
        nativeName: toolName,
        input: JSON.stringify(toolInput ?? {}),
        providerExecuted: true,
        ...(isExternalMcpTool(toolName) ? { dynamic: true } : {}),
      });
      input.emit({
        type: 'tool-approval-request',
        approvalId,
        toolCallId: approvalId,
      });
      input.finishApprovalStep(approvalId);

      const decision = await input.turn.requestToolApproval(approvalId);
      return decision.approved
        ? { behavior: 'allow', updatedInput: toolInput, toolUseID: approvalId }
        : {
            behavior: 'deny',
            message: decision.reason ?? 'Denied',
            toolUseID: approvalId,
          };
    },
  };
}

function createQuestionPreToolUseHook(input: {
  turn: Pick<BridgeTurn, 'requestToolResult'>;
  emit: Emit;
  nativeToolCallNames: Map<string, string>;
}): HookCallback {
  return async (hookInput, toolUseID): Promise<HookJSONOutput> => {
    if (
      hookInput.hook_event_name !== 'PreToolUse' ||
      hookInput.tool_name !== 'AskUserQuestion'
    ) {
      return {};
    }

    const nativeInput = hookInput.tool_input as Parameters<
      typeof toHarnessQuestionsInput
    >[0];
    const canonicalInput = toHarnessQuestionsInput(nativeInput);
    const toolCallId = toolUseID ?? hookInput.tool_use_id;
    input.nativeToolCallNames.set(toolCallId, hookInput.tool_name);
    input.emit({
      type: 'tool-call',
      toolCallId,
      toolName: 'askUserQuestions',
      nativeName: hookInput.tool_name,
      input: JSON.stringify(canonicalInput),
      providerExecuted: false,
      providerMetadata: {
        'claude-code': {
          nativeRequest: nativeInput,
        },
      },
    });

    const questionKey = claudeCodeQuestionKey(nativeInput);
    const result = await input.turn.requestToolResult({
      toolCallId,
      matches: candidate => {
        const nativeRequest = candidate.toolResult?.providerOptions?.[
          'claude-code'
        ]?.nativeRequest as
          | Parameters<typeof claudeCodeQuestionKey>[0]
          | undefined;
        return (
          nativeRequest != null &&
          claudeCodeQuestionKey(nativeRequest) === questionKey
        );
      },
    });
    const nativeResult = toClaudeCodeQuestionResult({
      nativeInput,
      output: result.output as Parameters<
        typeof toClaudeCodeQuestionResult
      >[0]['output'],
    });

    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        ...(nativeResult.behavior === 'allow'
          ? {
              permissionDecision: 'allow',
              updatedInput: {
                ...nativeResult.updatedInput,
              } as Record<string, unknown>,
            }
          : {
              permissionDecision: 'deny',
              permissionDecisionReason: nativeResult.message,
            }),
      },
    };
  };
}

function createPermissionSettings(input: {
  permissionMode: 'allow-reads' | 'allow-edits' | 'allow-all';
  inactiveNativeTools: ReadonlySet<string>;
}): Record<string, unknown> | undefined {
  const askRules = new Set<string>();
  for (const [nativeName, kind] of Object.entries(NATIVE_TOOL_KINDS)) {
    if (
      input.inactiveNativeTools.has(nativeName) ||
      (input.permissionMode === 'allow-reads'
        ? kind === 'edit' || kind === 'bash'
        : input.permissionMode === 'allow-edits'
          ? kind === 'bash'
          : false)
    ) {
      askRules.add(`${nativeName}(*)`);
    }
  }

  if (askRules.size === 0) return undefined;

  return {
    permissions: { ask: [...askRules] },
    sandbox: { autoAllowBashIfSandboxed: false },
  };
}

function nativeToolRequiresApproval(input: {
  nativeName: string;
  permissionMode: 'allow-reads' | 'allow-edits' | 'allow-all';
}): boolean {
  if (input.permissionMode === 'allow-all') return false;
  const kind = NATIVE_TOOL_KINDS[input.nativeName] ?? 'edit';
  if (input.permissionMode === 'allow-edits') return kind === 'bash';
  return kind === 'edit' || kind === 'bash';
}

/**
 * The host-executed tools of a `start` as an in-process MCP server. Each call
 * is surfaced as a `tool-call` and blocks on the host's `tool-result`.
 */
function createHostToolsServer(input: {
  tools: NonNullable<StartMessage['tools']>;
  emit: Emit;
  requestToolResult: BridgeTurn['requestToolResult'];
}): Record<string, unknown> {
  const server = new mcpModule.McpServer({
    name: 'harness-tools',
    version: '1.0.0',
  });
  for (const tool of input.tools) {
    server.registerTool(
      tool.name,
      {
        description: tool.description ?? '',
        inputSchema: jsonSchemaToZodObject(tool.inputSchema),
      },
      async (
        ...handlerArgs: [
          Record<string, unknown>,
          { requestId: string | number; _meta?: Record<string, unknown> },
        ]
      ) => {
        const [toolInput, extra] = handlerArgs;
        const metadataToolCallId = extra._meta?.['claudecode/toolUseId'];
        const toolCallId =
          typeof metadataToolCallId === 'string'
            ? metadataToolCallId
            : randomUUID();
        input.emit({
          type: 'tool-call',
          toolCallId,
          toolName: tool.name,
          input: JSON.stringify(toolInput),
          providerExecuted: false,
        });
        const { output, isError } = await input.requestToolResult(toolCallId);
        input.emit({
          type: 'tool-result',
          toolCallId,
          toolName: tool.name,
          result: output ?? null,
          isError: !!isError,
        });
        return {
          content: [{ type: 'text', text: JSON.stringify(output ?? null) }],
          isError,
        };
      },
    );
  }
  return { type: 'sdk', name: 'harness-tools', instance: server };
}

/**
 * The Agent SDK options for a query driven by `start`. `turn` and `emit` back
 * the host round trips (approvals, questions, host tools); a persistent query
 * passes handles that route to whichever turn is active when they fire.
 */
function createQueryOptions(input: {
  start: StartMessage;
  firstTurn: boolean;
  turn: Pick<BridgeTurn, 'requestToolApproval' | 'requestToolResult'>;
  emit: Emit;
  state: ClaudeStreamEventState;
  onCompactSummary: (summary: string) => void;
  abortSignal: AbortSignal;
}): Record<string, unknown> {
  const { start, turn, emit, state } = input;
  const mcpServers: Record<string, unknown> = { ...start.mcpServers };
  if (start.tools && start.tools.length > 0) {
    mcpServers['harness-tools'] = createHostToolsServer({
      tools: start.tools,
      emit,
      requestToolResult: turn.requestToolResult,
    });
  }
  const skillsOption = toClaudeSkillsOption(start.skills);
  const nativeTools = resolveNativeTools(start.builtinToolFiltering);
  const inactiveNativeTools = resolveInactiveNativeTools(
    start.builtinToolFiltering,
  );
  const permissionOptions = createPermissionOptions({
    start,
    inactiveNativeTools,
    turn,
    emit,
    finishApprovalStep: approvalId => {
      finishApprovalStep({ state, emit, approvalId });
    },
    nativeToolCallNames: state.nativeToolCallNames,
    approvalRequestedToolUseIds: state.approvalRequestedToolUseIds,
  });
  const questionPreToolUseHook = createQuestionPreToolUseHook({
    turn,
    emit,
    nativeToolCallNames: state.nativeToolCallNames,
  });

  return {
    ...(start.model ? { model: start.model } : {}),
    ...(start.maxTurns !== undefined ? { maxTurns: start.maxTurns } : {}),
    ...(start.agentProgressSummaries !== undefined
      ? { agentProgressSummaries: start.agentProgressSummaries }
      : {}),
    ...(start.forwardSubagentText !== undefined
      ? { forwardSubagentText: start.forwardSubagentText }
      : {}),
    ...(start.env !== undefined ? { env: { ...procEnv, ...start.env } } : {}),
    ...(skillsOption ? { skills: skillsOption } : {}),
    ...(nativeTools !== undefined ? { tools: nativeTools } : {}),
    ...(inactiveNativeTools.length > 0
      ? { disallowedTools: inactiveNativeTools }
      : {}),
    systemPrompt: createClaudeCodeSystemPrompt(start.instructions),
    thinking: start.thinking,
    ...(start.effort !== undefined ? { effort: start.effort } : {}),
    ...(start.responseFormat?.type === 'json' &&
    start.responseFormat.schema != null
      ? {
          outputFormat: {
            type: 'json_schema' as const,
            schema: start.responseFormat.schema,
          },
        }
      : {}),
    includePartialMessages: true,
    // The `PostCompact` hook carries the compaction summary, which the
    // `compact_boundary` system message does not. Latch it for the unified
    // `compaction` event; return an empty output so compaction proceeds.
    hooks: {
      PreToolUse: [
        {
          matcher: 'AskUserQuestion',
          hooks: [questionPreToolUseHook],
        },
      ],
      PostCompact: [
        {
          hooks: [
            async (hookInput: { compact_summary?: unknown }) => {
              if (typeof hookInput?.compact_summary === 'string') {
                input.onCompactSummary(hookInput.compact_summary);
              }
              return {};
            },
          ],
        },
      ],
    },
    // Continuation rule, most specific first.
    //
    // `resumeSessionId` names the exact conversation and is what a
    // cross-process resume should use: `continue` means "most recent thread
    // in this workdir", which silently picks the wrong one once anything
    // else has run there. The bridge also retains the id observed during its
    // previous query, so every later query stays pinned to that conversation
    // even when the host detached and reattached between turns. `resume` and
    // `continue` are mutually exclusive in the SDK.
    //
    // Otherwise the host can force-continue by setting `start.continue`,
    // and turns after the first fall back to the legacy cwd-based behavior
    // when no exact id was observed.
    ...((start.resumeSessionId ?? lastClaudeSessionId)
      ? { resume: start.resumeSessionId ?? lastClaudeSessionId }
      : start.continue === true || !input.firstTurn
        ? { continue: true }
        : {}),
    ...permissionOptions,
    mcpServers,
    cwd: workdir,
    abortSignal: input.abortSignal,
  };
}

async function runTurn(start: StartMessage, turn: BridgeTurn): Promise<void> {
  const emit: Emit = msg => turn.emit(msg as BridgeEvent);

  // Local controller for the Claude query. Aborted either by the host (via the
  // shared runtime's `turn.abortSignal`) or by us on a terminal error.
  const abortCtl = new AbortController();
  // A host abort prefers the SDK's graceful `interrupt()` — Esc semantics: the
  // in-flight turn is persisted to the session transcript and settles with an
  // interrupted result, so a later resume (including the user's own
  // `claude --resume`) still sees the work done before the interrupt. The hard
  // abort kills the CLI process and loses that turn's records, so it is only
  // the fallback — armed unconditionally, because aborting an already-settled
  // query is a no-op — and the immediate path when the abort arrives before
  // the query exists.
  let gracefulAbort: (() => void) | undefined;
  let hardAbortTimer: ReturnType<typeof setTimeout> | undefined;
  const onHostAbort = (): void => {
    if (gracefulAbort) {
      gracefulAbort();
    } else {
      abortCtl.abort();
    }
  };
  if (turn.abortSignal.aborted) {
    abortCtl.abort();
  } else {
    turn.abortSignal.addEventListener('abort', onHostAbort, { once: true });
  }

  const streamEventState = createClaudeStreamEventState();

  // Compaction observation: merge Claude's `compact_boundary` message and
  // `PostCompact` hook (which arrive in either order) into one `compaction`
  // event. See `createCompactionLatch`.
  const compaction = createCompactionLatch(event => emit(event));

  // `stream-start` is emitted lazily on the first SDK message (below) so it can
  // carry the model the CLI resolved to, reported on the `system`/`init` message.

  const queryInput = createQueryInput({
    initialUserMessage: start.prompt,
    userMessages: turn.experimental_userMessages,
    abortSignal: abortCtl.signal,
  });

  const q = claudeSdk.query({
    prompt: queryInput.input,
    options: createQueryOptions({
      start,
      firstTurn: turn.firstTurn,
      turn,
      emit,
      state: streamEventState,
      onCompactSummary: summary => compaction.onSummary(summary),
      abortSignal: abortCtl.signal,
    }),
  });

  gracefulAbort = () => {
    // Backstop for the whole teardown, not just the interrupt call: if the
    // stream has not settled five seconds after a graceful interrupt was
    // requested, fall back to the hard abort. Aborting an already-settled
    // query is a no-op, and `unref` keeps the timer from pinning the bridge
    // process open on its own.
    hardAbortTimer = setTimeout(() => abortCtl.abort(), 5000);
    hardAbortTimer.unref?.();
    void Promise.resolve()
      .then(() => q.interrupt())
      .catch(() => abortCtl.abort());
  };
  let turnUsage: Record<string, unknown> | undefined;
  let totalCostUsd: number | undefined;
  let emittedTerminalError = false;
  let emittedTerminalFinish = false;

  const emitTerminalError = (message: string | undefined): void => {
    const normalized = message?.trim();
    if (!normalized || emittedTerminalError || emittedTerminalFinish) return;
    streamEventState.observedTerminalError = normalized;
    emittedTerminalError = true;
    // A turn the host itself stopped ends with an error-shaped result by
    // construction (an interrupted query reports a diagnostic, not success);
    // reporting the host's own stop as a terminal error makes every clean
    // interrupt look like a malfunction. The host has already settled the
    // turn on its side.
    if (!turn.abortSignal.aborted) {
      turn.emitError({
        error: normalized,
        message: 'claude-code terminal error',
      });
    }
    queryInput.close();
    abortCtl.abort();
  };

  const emitStreamEvent = createEmitStreamEvent({
    state: streamEventState,
    emit,
    emitWarning: turn.emitWarning,
    emitTerminalError,
    onCompactionBoundary: boundary => compaction.onBoundary(boundary),
    toCommonName,
  });

  try {
    for await (const msg of q as AsyncIterable<ClaudeMessage>) {
      if (abortCtl.signal.aborted) break;

      const type = msg.type;

      if (type === 'command_lifecycle') {
        queryInput.handleLifecycle(msg);
      }

      // Every SDK message carries the session id of the conversation it
      // belongs to. Track the latest so the stop payload and the terminal
      // finish metadata name the exact conversation.
      const sessionId = (msg as { session_id?: unknown }).session_id;
      if (typeof sessionId === 'string' && sessionId.length > 0) {
        lastClaudeSessionId = sessionId;
      }

      emitStreamEvent(msg);

      if (type === 'result') {
        if (msg.subtype === 'success') {
          // `success` does not mean the turn succeeded: the CLI flags a rejected
          // request with `is_error` and puts the message in `result`, which the
          // empty-result rescue below cannot catch.
          if (msg.is_error) {
            emitTerminalError(
              msg.result?.trim() ||
                streamEventState.observedTerminalError ||
                (typeof msg.api_error_status === 'number'
                  ? `Claude Code reported an API error (HTTP ${msg.api_error_status})`
                  : 'Claude Code reported a failed result'),
            );
            continue;
          }
          const emptyResult = !msg.result?.trim?.();
          if (emptyResult && streamEventState.observedTerminalError) {
            emitTerminalError(streamEventState.observedTerminalError);
            continue;
          }
          const usage = msg.usage ?? msg.message?.usage;
          const harnessUsage = mapUsage(usage);
          if (harnessUsage) turnUsage = addUsage(turnUsage, harnessUsage);
          if (typeof msg.total_cost_usd === 'number') {
            totalCostUsd = msg.total_cost_usd;
          }
          if (
            start.responseFormat?.type === 'json' &&
            msg.structured_output !== undefined
          ) {
            const id = randomUUID();
            emit({ type: 'text-start', id });
            emit({
              type: 'text-delta',
              id,
              delta: JSON.stringify(msg.structured_output),
            });
            emit({ type: 'text-end', id });
            streamEventState.stepOpen = true;
          }
          if (streamEventState.stepOpen) {
            emitFinishStep({
              state: streamEventState,
              emit,
              usage: streamEventState.pendingStepUsage ?? harnessUsage,
            });
          }
          queryInput.observeResult();
          if (!queryInput.hasActiveUserMessages()) {
            queryInput.close();
            break;
          }
        } else {
          emitTerminalError(
            (Array.isArray(msg.errors) ? msg.errors.join('\n') : undefined) ||
              streamEventState.observedTerminalError ||
              msg.result ||
              'Unknown error',
          );
        }
        continue;
      }

      if (queryInput.hasObservedResult && !queryInput.hasActiveUserMessages()) {
        queryInput.close();
        break;
      }
    }
  } catch (err) {
    // Same reasoning as `emitTerminalError`: a throw after the host's own
    // abort (e.g. the hard-abort fallback killing the CLI mid-iteration, or
    // a rejected `interrupt()`) is the stop the host asked for, not a
    // malfunction. The host has already settled the turn on its side.
    if (
      !turn.abortSignal.aborted &&
      !(abortCtl.signal.aborted && emittedTerminalError)
    ) {
      turn.emitError({ error: err, message: 'claude-code turn failed' });
    }
    return;
  } finally {
    // The turn is over; disarm the host-abort path first. An abort of this
    // turn's signal arriving after this point (e.g. an `abort` message racing
    // the next `start`) must not interrupt the disposed query or arm the
    // hard-abort fallback timer for it.
    gracefulAbort = undefined;
    if (hardAbortTimer != null) clearTimeout(hardAbortTimer);
    turn.abortSignal.removeEventListener('abort', onHostAbort);
    queryInput.close();
    // Dispose the query explicitly: with streaming input the SDK keeps its
    // CLI subprocess alive for more user messages, and a turn that ended
    // through an interrupt or error path can otherwise leak that process —
    // observed as orphaned `claude` processes holding the very conversation
    // the next turn continues.
    try {
      await (q as { return?: (value?: unknown) => Promise<unknown> }).return?.(
        undefined,
      );
    } catch {
      // Best effort; the abort controller tears the process down otherwise.
    }
  }

  if (emittedTerminalError) return;
  emittedTerminalFinish = true;
  void emittedTerminalFinish;
  emit({
    type: 'finish',
    finishReason: { unified: 'stop', raw: 'stop' },
    totalUsage: turnUsage ?? streamEventState.stepUsage ?? defaultUsage(),
    ...(totalCostUsd !== undefined || lastClaudeSessionId !== undefined
      ? {
          harnessMetadata: {
            'claude-code': {
              ...(totalCostUsd !== undefined ? { costUsd: totalCostUsd } : {}),
              // The conversation this turn belongs to, resumable outside the
              // SDK with `claude --resume <sessionId>` and captured by the
              // adapter for exact cross-process resume.
              ...(lastClaudeSessionId !== undefined
                ? { sessionId: lastClaudeSessionId }
                : {}),
            },
          },
        }
      : {}),
  });
}

type ClaudeQuery = AsyncIterable<ClaudeMessage> & {
  interrupt?: () => Promise<unknown>;
  close?: () => void;
  return?: (value?: unknown) => Promise<unknown>;
};

/** One host turn served by the live query. */
type PersistentTurn = {
  turn: BridgeTurn;
  emit: Emit;
  /** The turn asked for structured output, which arrives on the result. */
  jsonResponse: boolean;
  emitStreamEvent: (message: ClaudeMessage) => void;
  /** Client uuids of the user messages this turn sent: its prompt, then steering. */
  messageIds: Set<string>;
  submittedMessages: Map<string, Experimental_BridgeUserMessage>;
  observedResult: boolean;
  turnUsage: Record<string, unknown> | undefined;
  totalCostUsd: number | undefined;
  hardAbortTimer: ReturnType<typeof setTimeout> | undefined;
  settled: boolean;
  settle(): void;
};

type LiveQuery = {
  /** The `start` options the query was opened with, minus per-turn fields. */
  key: string;
  query: ClaudeQuery;
  input: PushableInput<unknown>;
  abortController: AbortController;
  state: ClaudeStreamEventState;
  compaction: CompactionLatch;
  current: PersistentTurn | undefined;
  /** The runtime's event sink, for messages that arrive between turns. */
  emit: Emit;
  closed: boolean;
};

const HARD_ABORT_MS = 5000;

/**
 * The `start` fields a live query is bound to. Everything but the prompt and
 * the continuation hints becomes a query option, so a turn that changes any of
 * them needs a fresh query.
 */
function persistentQueryKey(start: StartMessage): string {
  const {
    type: _type,
    prompt: _prompt,
    resumeSessionId: _resumeSessionId,
    continue: _continue,
    debug: _debug,
    ...options
  } = start as StartMessage & { debug?: unknown };
  return JSON.stringify(options);
}

/**
 * Serve a turn from the live query, opening one first when there is none or
 * the turn's options differ. The turn ends at the result that answers its
 * prompt; the query, and any background work it runs, stays up for the next
 * turn.
 */
async function runPersistentTurn(
  start: StartMessage,
  turn: BridgeTurn,
): Promise<void> {
  const key = persistentQueryKey(start);
  if (liveQuery != null && (liveQuery.closed || liveQuery.key !== key)) {
    closeLiveQuery();
  }
  // The host already settled an aborted turn; do not start work for it.
  if (turn.abortSignal.aborted) return;
  const live = liveQuery ?? openLiveQuery({ start, turn, key });
  live.emit = msg => turn.emit(msg as BridgeEvent);

  let finished!: () => void;
  const done = new Promise<void>(resolve => {
    finished = resolve;
  });
  {
    const emit: Emit = msg => turn.emit(msg as BridgeEvent);
    const onHostAbort = (): void => interruptPersistentTurn(live, active);
    const active: PersistentTurn = {
      turn,
      emit,
      jsonResponse: start.responseFormat?.type === 'json',
      emitStreamEvent: () => {},
      messageIds: new Set(),
      submittedMessages: new Map(),
      observedResult: false,
      turnUsage: undefined,
      totalCostUsd: undefined,
      hardAbortTimer: undefined,
      settled: false,
      settle: () => {
        if (active.settled) return;
        active.settled = true;
        if (live.current === active) live.current = undefined;
        turn.abortSignal.removeEventListener('abort', onHostAbort);
        if (active.hardAbortTimer != null) clearTimeout(active.hardAbortTimer);
        finished();
      },
    };
    resetClaudeStreamEventTurnState(live.state);
    active.emitStreamEvent = createEmitStreamEvent({
      state: live.state,
      emit,
      emitWarning: turn.emitWarning,
      emitTerminalError: message =>
        failPersistentTurn(live, active, message, { fatal: true }),
      onCompactionBoundary: boundary => live.compaction.onBoundary(boundary),
      toCommonName,
    });
    live.current = active;
    turn.abortSignal.addEventListener('abort', onHostAbort, { once: true });
    void forwardSteering(live, active);

    const promptId = randomUUID();
    active.messageIds.add(promptId);
    live.input.push(
      toSdkUserMessage({ text: start.prompt, messageId: promptId }),
    );
    if (live.closed) {
      turn.emitError({
        error: new Error('Claude Code exited before the turn started.'),
        message: 'claude-code turn failed',
      });
      active.settle();
    }
  }
  await done;
}

function openLiveQuery(input: {
  start: StartMessage;
  turn: BridgeTurn;
  key: string;
}): LiveQuery {
  const abortController = new AbortController();
  const state = createClaudeStreamEventState();
  const live: LiveQuery = {
    key: input.key,
    query: undefined as unknown as ClaudeQuery,
    input: createPushableInput<unknown>(),
    abortController,
    state,
    compaction: createCompactionLatch(event => live.current?.emit(event)),
    current: undefined,
    emit: msg => input.turn.emit(msg as BridgeEvent),
    closed: false,
  };
  // Host round trips go to whichever turn is active when they fire. A request
  // between turns (a background subagent asking for approval, say) has no host
  // to answer it, so it is refused.
  const routedTurn: Pick<
    BridgeTurn,
    'requestToolApproval' | 'requestToolResult'
  > = {
    requestToolApproval: approvalId =>
      live.current != null
        ? live.current.turn.requestToolApproval(approvalId)
        : Promise.resolve({
            approved: false,
            reason: 'No turn is active to approve this tool call.',
          }),
    requestToolResult: request =>
      live.current != null
        ? live.current.turn.requestToolResult(request)
        : Promise.resolve({
            output: 'No turn is active to run this tool.',
            isError: true,
          }),
  };
  live.query = claudeSdk.query({
    prompt: live.input,
    options: createQueryOptions({
      start: input.start,
      firstTurn: input.turn.firstTurn,
      turn: routedTurn,
      emit: msg => live.current?.emit(msg),
      state,
      onCompactSummary: summary => live.compaction.onSummary(summary),
      abortSignal: abortController.signal,
    }),
  }) as ClaudeQuery;
  liveQuery = live;
  void pumpLiveQuery(live);
  return live;
}

/** Route every message of the live query, for as long as it runs. */
async function pumpLiveQuery(live: LiveQuery): Promise<void> {
  let failure: unknown;
  try {
    for await (const msg of live.query) {
      if (live.closed) break;
      const sessionId = (msg as { session_id?: unknown }).session_id;
      if (typeof sessionId === 'string' && sessionId.length > 0) {
        lastClaudeSessionId = sessionId;
      }
      const active = live.current;
      if (msg.type === 'command_lifecycle' && active != null) {
        handleSteeringLifecycle(active, msg);
      }
      if (active == null) {
        forwardBetweenTurns(live, msg);
        continue;
      }
      handlePersistentMessage(live, active, msg);
    }
  } catch (err) {
    failure = err;
  } finally {
    live.closed = true;
    live.input.close();
    if (liveQuery === live) liveQuery = undefined;
    const active = live.current;
    if (active != null && !active.settled) {
      if (!active.turn.abortSignal.aborted) {
        active.turn.emitError({
          error:
            failure ??
            new Error('Claude Code exited before the turn finished.'),
          message: 'claude-code turn failed',
        });
      }
      active.settle();
    }
  }
}

/**
 * Between turns the query keeps running: background shells finish, subagents
 * report, and the CLI may answer a task notification on its own. Each message
 * goes to the host as a `raw` part (partial stream events excepted), never as
 * a regular stream part, which would read as part of the next turn.
 */
function forwardBetweenTurns(live: LiveQuery, msg: ClaudeMessage): void {
  if (msg.type === 'stream_event') return;
  live.emit({ type: 'raw', rawValue: msg });
}

function handlePersistentMessage(
  live: LiveQuery,
  active: PersistentTurn,
  msg: ClaudeMessage,
): void {
  // An interrupted turn ends at the result the interrupt produces.
  if (msg.type === 'result' && active.turn.abortSignal.aborted) {
    active.settle();
    return;
  }

  active.emitStreamEvent(msg);
  if (active.settled) return;

  if (msg.type !== 'result') {
    if (active.observedResult && !hasActiveSteering(active)) {
      finishPersistentTurn(live, active);
    }
    return;
  }

  // A result for a turn the CLI started itself (answering a background task
  // notification while this turn's prompt waited in its queue) is not ours.
  if (!resultAnswersTurn(active, msg)) return;

  if (msg.subtype !== 'success') {
    failPersistentTurn(
      live,
      active,
      (Array.isArray(msg.errors) ? msg.errors.join('\n') : undefined) ||
        live.state.observedTerminalError ||
        msg.result ||
        'Unknown error',
      { fatal: false },
    );
    return;
  }
  if (msg.is_error) {
    failPersistentTurn(
      live,
      active,
      msg.result?.trim() ||
        live.state.observedTerminalError ||
        (typeof msg.api_error_status === 'number'
          ? `Claude Code reported an API error (HTTP ${msg.api_error_status})`
          : 'Claude Code reported a failed result'),
      { fatal: false },
    );
    return;
  }
  if (!msg.result?.trim?.() && live.state.observedTerminalError) {
    failPersistentTurn(live, active, live.state.observedTerminalError, {
      fatal: false,
    });
    return;
  }
  const harnessUsage = mapUsage(msg.usage ?? msg.message?.usage);
  if (harnessUsage) active.turnUsage = addUsage(active.turnUsage, harnessUsage);
  if (typeof msg.total_cost_usd === 'number') {
    active.totalCostUsd = msg.total_cost_usd;
  }
  if (active.jsonResponse && msg.structured_output !== undefined) {
    const id = randomUUID();
    active.emit({ type: 'text-start', id });
    active.emit({
      type: 'text-delta',
      id,
      delta: JSON.stringify(msg.structured_output),
    });
    active.emit({ type: 'text-end', id });
    live.state.stepOpen = true;
  }
  if (live.state.stepOpen) {
    emitFinishStep({
      state: live.state,
      emit: active.emit,
      usage: live.state.pendingStepUsage ?? harnessUsage,
    });
  }
  active.observedResult = true;
  if (!hasActiveSteering(active)) finishPersistentTurn(live, active);
}

/**
 * Whether a result answers this turn. Current CLIs name the user messages a
 * result consumed; a result naming none of this turn's, while user sends are
 * still queued, belongs to a turn the CLI started on its own. A result that
 * names nothing (older CLIs) is taken as this turn's.
 */
function resultAnswersTurn(
  active: PersistentTurn,
  msg: ClaudeMessage,
): boolean {
  const result = msg as ClaudeMessage & {
    user_message_uuid?: unknown;
    user_message_uuids?: unknown;
    queued_turn_count?: unknown;
  };
  const ids = Array.isArray(result.user_message_uuids)
    ? result.user_message_uuids.filter(
        (id): id is string => typeof id === 'string',
      )
    : typeof result.user_message_uuid === 'string'
      ? [result.user_message_uuid]
      : [];
  if (ids.length === 0) return true;
  if (ids.some(id => active.messageIds.has(id))) return true;
  return !(
    typeof result.queued_turn_count === 'number' && result.queued_turn_count > 0
  );
}

function hasActiveSteering(active: PersistentTurn): boolean {
  return (
    active.submittedMessages.size > 0 ||
    active.turn.experimental_userMessages.pendingCount > 0
  );
}

function finishPersistentTurn(live: LiveQuery, active: PersistentTurn): void {
  if (active.settled) return;
  active.emit({
    type: 'finish',
    finishReason: { unified: 'stop', raw: 'stop' },
    totalUsage: active.turnUsage ?? live.state.stepUsage ?? defaultUsage(),
    ...(active.totalCostUsd !== undefined || lastClaudeSessionId !== undefined
      ? {
          harnessMetadata: {
            'claude-code': {
              ...(active.totalCostUsd !== undefined
                ? { costUsd: active.totalCostUsd }
                : {}),
              ...(lastClaudeSessionId !== undefined
                ? { sessionId: lastClaudeSessionId }
                : {}),
            },
          },
        }
      : {}),
  });
  active.settle();
}

/**
 * End the turn with an error. A failed result leaves the CLI ready for the
 * next turn, so the query stays up; a fatal stream error (rejected
 * credentials, an unrecoverable API status) closes it, and the next turn
 * resumes the conversation in a fresh one.
 */
function failPersistentTurn(
  live: LiveQuery,
  active: PersistentTurn,
  message: string | undefined,
  { fatal }: { fatal: boolean },
): void {
  const normalized = message?.trim();
  if (!normalized || active.settled) return;
  live.state.observedTerminalError = normalized;
  // The host's own stop ends with an error-shaped result; it is not a fault.
  if (!active.turn.abortSignal.aborted) {
    active.turn.emitError({
      error: normalized,
      message: 'claude-code terminal error',
    });
  }
  active.settle();
  if (fatal) closeLiveQuery(live);
}

/**
 * A host stop interrupts the CLI's current turn (Esc semantics), which keeps
 * the process and its background work. If the interrupt does not settle the
 * turn in time, the query is closed instead.
 */
function interruptPersistentTurn(
  live: LiveQuery,
  active: PersistentTurn,
): void {
  if (active.settled) return;
  active.hardAbortTimer = setTimeout(() => closeLiveQuery(live), HARD_ABORT_MS);
  active.hardAbortTimer.unref?.();
  void Promise.resolve()
    .then(() => live.query.interrupt?.())
    .catch(() => closeLiveQuery(live));
}

async function forwardSteering(
  live: LiveQuery,
  active: PersistentTurn,
): Promise<void> {
  for await (const message of active.turn.experimental_userMessages) {
    if (active.settled || live.closed) {
      message.reject(
        new Error('The bridge turn ended before accepting the user message.'),
      );
      continue;
    }
    active.submittedMessages.set(message.messageId, message);
    active.messageIds.add(message.messageId);
    live.input.push(
      toSdkUserMessage({
        text: message.text,
        messageId: message.messageId,
        priority: 'next',
      }),
    );
  }
}

function handleSteeringLifecycle(
  active: PersistentTurn,
  message: ClaudeMessage,
): void {
  const lifecycle = message as ClaudeMessage & {
    command_uuid?: string;
    state?: 'queued' | 'started' | 'completed' | 'cancelled' | 'discarded';
  };
  if (lifecycle.command_uuid == null || lifecycle.state == null) return;
  const submitted = active.submittedMessages.get(lifecycle.command_uuid);
  if (submitted == null) return;
  if (lifecycle.state === 'queued' || lifecycle.state === 'started') {
    submitted.accept();
    return;
  }
  if (lifecycle.state === 'cancelled' || lifecycle.state === 'discarded') {
    submitted.reject(
      new Error(`Claude Code ${lifecycle.state} the user message.`),
    );
  }
  active.submittedMessages.delete(lifecycle.command_uuid);
}

/** Close the live query and its CLI process, ending any background work. */
function closeLiveQuery(live: LiveQuery | undefined = liveQuery): void {
  if (live == null) return;
  if (liveQuery === live) liveQuery = undefined;
  if (live.closed) return;
  live.closed = true;
  live.input.close();
  live.abortController.abort();
  try {
    if (live.query.close != null) {
      live.query.close();
    } else {
      void live.query.return?.(undefined)?.catch(() => {});
    }
  } catch {
    // Best effort; the abort tears the process down otherwise.
  }
}

function toSdkUserMessage(options: {
  text: string;
  messageId: string;
  priority?: 'next';
}): unknown {
  return {
    type: 'user',
    message: {
      role: 'user',
      content: [{ type: 'text', text: options.text }],
    },
    parent_tool_use_id: null,
    uuid: options.messageId,
    ...(options.priority == null ? {} : { priority: options.priority }),
  };
}

function createQueryInput({
  initialUserMessage,
  userMessages,
  abortSignal,
}: {
  initialUserMessage: string;
  userMessages: Experimental_BridgeUserMessageQueue;
  abortSignal: AbortSignal;
}): {
  input: AsyncIterable<unknown>;
  close(error?: unknown): void;
  handleLifecycle(message: ClaudeMessage): void;
  hasActiveUserMessages(): boolean;
  observeResult(): void;
  readonly hasObservedResult: boolean;
} {
  let closed = false;
  let observedResult = false;
  const submittedMessages = new Map<string, Experimental_BridgeUserMessage>();
  const close = (error?: unknown): void => {
    if (closed) return;
    closed = true;
    userMessages.close(error);
  };
  if (abortSignal.aborted) {
    close(abortSignal.reason);
  } else {
    abortSignal.addEventListener('abort', () => close(abortSignal.reason), {
      once: true,
    });
  }

  const toUserMessage = (options: {
    text: string;
    messageId: string;
    priority?: 'next';
  }): unknown => ({
    type: 'user',
    message: {
      role: 'user',
      content: [{ type: 'text', text: options.text }],
    },
    parent_tool_use_id: null,
    uuid: options.messageId,
    ...(options.priority == null ? {} : { priority: options.priority }),
  });

  const messageIterator = userMessages[Symbol.asyncIterator]();

  return {
    close,
    handleLifecycle: message => {
      const lifecycle = message as ClaudeMessage & {
        command_uuid?: string;
        state?: 'queued' | 'started' | 'completed' | 'cancelled' | 'discarded';
      };
      if (lifecycle.command_uuid == null || lifecycle.state == null) return;
      const submitted = submittedMessages.get(lifecycle.command_uuid);
      if (submitted == null) return;
      if (lifecycle.state === 'queued' || lifecycle.state === 'started') {
        submitted.accept();
        return;
      }
      if (lifecycle.state === 'cancelled' || lifecycle.state === 'discarded') {
        submitted.reject(
          new Error(`Claude Code ${lifecycle.state} the user message.`),
        );
      }
      submittedMessages.delete(lifecycle.command_uuid);
    },
    hasActiveUserMessages: () =>
      submittedMessages.size > 0 || userMessages.pendingCount > 0,
    observeResult: () => {
      observedResult = true;
    },
    get hasObservedResult() {
      return observedResult;
    },
    input: {
      [Symbol.asyncIterator]() {
        let sentInitial = false;
        return {
          async next() {
            if (closed || abortSignal.aborted) {
              return {
                value: undefined,
                done: true,
              } as IteratorResult<unknown>;
            }
            if (!sentInitial) {
              sentInitial = true;
              return {
                value: toUserMessage({
                  text: initialUserMessage,
                  messageId: randomUUID(),
                }),
                done: false,
              };
            }
            const nextMessage = await messageIterator.next();
            if (nextMessage.done) {
              return {
                value: undefined,
                done: true,
              } as IteratorResult<unknown>;
            }
            submittedMessages.set(
              nextMessage.value.messageId,
              nextMessage.value,
            );
            return {
              value: toUserMessage({
                text: nextMessage.value.text,
                messageId: nextMessage.value.messageId,
                priority: 'next',
              }),
              done: false,
            };
          },
        };
      },
    },
  };
}

function addUsage(
  total: Record<string, unknown> | undefined,
  usage: Record<string, unknown>,
): Record<string, unknown> {
  if (total == null) return usage;
  const result: Record<string, unknown> = { ...total };
  for (const [key, value] of Object.entries(usage)) {
    const previous = result[key];
    if (typeof value === 'number' && typeof previous === 'number') {
      result[key] = previous + value;
    } else if (
      value != null &&
      previous != null &&
      typeof value === 'object' &&
      typeof previous === 'object' &&
      !Array.isArray(value) &&
      !Array.isArray(previous)
    ) {
      result[key] = addUsage(
        previous as Record<string, unknown>,
        value as Record<string, unknown>,
      );
    } else {
      result[key] = value;
    }
  }
  return result;
}

function parseArgs(args: string[]): {
  workdir?: string;
  bridgeStateDir?: string;
} {
  const out: { workdir?: string; bridgeStateDir?: string } = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--workdir' && i + 1 < args.length) {
      out.workdir = args[++i];
    } else if (args[i] === '--bridge-state-dir' && i + 1 < args.length) {
      out.bridgeStateDir = args[++i];
    }
  }
  return out;
}

function emitFatal(message: string): never {
  stdout.write(JSON.stringify({ type: 'bridge-fatal', message }) + '\n');
  process.exit(1);
}
