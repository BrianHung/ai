import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { createPushableInput } from './pushable-input';

type QueryArgs = {
  prompt: AsyncIterable<unknown>;
  options: Record<string, unknown>;
};

/** A scripted stand-in for the Agent SDK's streaming-input `Query`. */
type FakeQuery = {
  args: QueryArgs;
  inputs: Array<Record<string, unknown>>;
  send(message: Record<string, unknown>): void;
  end(): void;
  interrupt: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  nextInput(): Promise<Record<string, unknown>>;
};

type Turn = {
  emitted: Record<string, unknown>[];
  errors: unknown[];
  abort: AbortController;
  done: Promise<void>;
};

const state = vi.hoisted(() => ({
  queries: [] as FakeQuery[],
  onStart: undefined as
    | ((start: unknown, turn: unknown) => Promise<void>)
    | undefined,
  onStop: undefined as (() => unknown) | undefined,
  originalArgv: [] as string[],
}));

vi.mock('@anthropic-ai/claude-agent-sdk', async () => {
  const { createPushableInput: pushable } = await import('./pushable-input');
  return {
    query: (args: QueryArgs) => {
      const output = pushable<Record<string, unknown>>();
      const endOutput = output.close.bind(output);
      const inputs: Array<Record<string, unknown>> = [];
      const pending: Array<Record<string, unknown>> = [];
      const waiters: Array<(value: Record<string, unknown>) => void> = [];
      const fake: FakeQuery = {
        args,
        inputs,
        send: message => output.push(message),
        end: () => endOutput(),
        interrupt: vi.fn(async () => {}),
        close: vi.fn(() => endOutput()),
        nextInput: () =>
          pending.length > 0
            ? Promise.resolve(pending.shift()!)
            : new Promise(resolve => {
                waiters.push(resolve);
              }),
      };
      void (async () => {
        for await (const input of args.prompt) {
          const value = input as Record<string, unknown>;
          inputs.push(value);
          const waiter = waiters.shift();
          if (waiter) waiter(value);
          else pending.push(value);
        }
      })();
      state.queries.push(fake);
      return Object.assign(output, {
        interrupt: fake.interrupt,
        close: fake.close,
      });
    },
  };
});

vi.mock('@modelcontextprotocol/sdk/server/mcp.js', () => ({
  McpServer: class {
    registerTool(): void {}
  },
}));

vi.mock('@ai-sdk/harness/bridge', () => ({
  runBridge: async (options: {
    onStart: (start: unknown, turn: unknown) => Promise<void>;
    onStop?: () => unknown;
  }) => {
    state.onStart = options.onStart;
    state.onStop = options.onStop;
  },
}));

const baseStart = {
  type: 'start',
  thinking: { type: 'disabled' },
  persistentQuery: true,
};

let firstTurn = true;

function startTurn(prompt: string, extra: Record<string, unknown> = {}): Turn {
  const emitted: Record<string, unknown>[] = [];
  const errors: unknown[] = [];
  const abort = new AbortController();
  const userMessages = Object.assign(createPushableInput<never>(), {
    pendingCount: 0,
  });
  const turn = {
    emit: (event: Record<string, unknown>) => emitted.push(event),
    emitWarning: () => {},
    emitError: (input: unknown) => errors.push(input),
    requestToolResult: async () => ({ output: {} }),
    requestToolApproval: async () => ({ approved: true }),
    experimental_userMessages: userMessages,
    abortSignal: abort.signal,
    firstTurn,
    bridgeLog: () => {},
  };
  firstTurn = false;
  const done = state.onStart!({ ...baseStart, prompt, ...extra }, turn).then(
    () => userMessages.close(),
  );
  return { emitted, errors, abort, done };
}

const result = (
  extra: Record<string, unknown> = {},
): Record<string, unknown> => ({
  type: 'result',
  subtype: 'success',
  result: 'ok',
  session_id: 'claude-session-1',
  ...extra,
});

const promptText = (input: Record<string, unknown>) =>
  (input.message as { content: Array<{ text: string }> }).content[0]!.text;

const flush = () => new Promise(resolve => setTimeout(resolve, 0));

describe('Claude Code bridge persistent query', () => {
  beforeEach(async () => {
    state.queries = [];
    firstTurn = true;
    state.originalArgv = [...process.argv];
    process.argv.splice(
      0,
      process.argv.length,
      'node',
      'bridge.mjs',
      '--workdir',
      '/tmp/harness-claude-code-test/work',
      '--bridge-state-dir',
      '/tmp/harness-claude-code-test/state',
    );
    await import('./index');
  });

  afterEach(() => {
    process.argv.splice(0, process.argv.length, ...state.originalArgv);
    vi.resetModules();
  });

  test('serves consecutive turns from one query and one CLI process', async () => {
    const first = startTurn('List the files.');
    const query = state.queries[0]!;
    const firstInput = await query.nextInput();
    expect(promptText(firstInput)).toBe('List the files.');
    query.send({
      type: 'system',
      subtype: 'init',
      session_id: 'claude-session-1',
    });
    query.send(result({ user_message_uuid: firstInput.uuid }));
    await first.done;
    expect(first.emitted.at(-1)).toMatchObject({
      type: 'finish',
      harnessMetadata: { 'claude-code': { sessionId: 'claude-session-1' } },
    });

    const second = startTurn('Now read the README.');
    const secondInput = await query.nextInput();
    expect(promptText(secondInput)).toBe('Now read the README.');
    query.send(result({ user_message_uuid: secondInput.uuid }));
    await second.done;

    expect(state.queries).toHaveLength(1);
    expect(query.close).not.toHaveBeenCalled();
    expect(second.emitted.at(-1)).toMatchObject({ type: 'finish' });
  });

  test('a background task started in one turn reports its end in the next', async () => {
    const first = startTurn('Start a slow build in the background.');
    const query = state.queries[0]!;
    const firstInput = await query.nextInput();
    query.send({
      type: 'system',
      subtype: 'task_started',
      task_id: 'bash-1',
      task_type: 'local_bash',
      description: 'sleep 20',
      is_backgrounded: true,
    });
    query.send(result({ user_message_uuid: firstInput.uuid }));
    await first.done;

    const second = startTurn('How is it going?');
    const secondInput = await query.nextInput();
    query.send({
      type: 'system',
      subtype: 'task_notification',
      task_id: 'bash-1',
      status: 'completed',
      summary: 'done',
    });
    query.send(result({ user_message_uuid: secondInput.uuid }));
    await second.done;

    expect(first.emitted).toContainEqual({
      type: 'raw',
      rawValue: expect.objectContaining({ subtype: 'task_started' }),
    });
    expect(second.emitted).toContainEqual({
      type: 'raw',
      rawValue: expect.objectContaining({
        subtype: 'task_notification',
        status: 'completed',
      }),
    });
  });

  test('forwards messages that arrive between turns as raw parts', async () => {
    const first = startTurn('Start a background agent.');
    const query = state.queries[0]!;
    const firstInput = await query.nextInput();
    query.send(result({ user_message_uuid: firstInput.uuid }));
    await first.done;

    query.send({
      type: 'system',
      subtype: 'task_notification',
      task_id: 'agent-1',
      status: 'completed',
    });
    query.send({
      type: 'stream_event',
      event: { type: 'content_block_delta' },
      parent_tool_use_id: null,
    });
    query.send({
      type: 'assistant',
      parent_tool_use_id: 'toolu_agent',
      message: { content: [{ type: 'text', text: 'Subagent says hi.' }] },
    });
    await flush();

    expect(first.emitted.slice(-2)).toEqual([
      {
        type: 'raw',
        rawValue: expect.objectContaining({ subtype: 'task_notification' }),
      },
      {
        type: 'raw',
        rawValue: expect.objectContaining({
          parent_tool_use_id: 'toolu_agent',
        }),
      },
    ]);
  });

  test('a result for a turn the CLI started itself does not end the host turn', async () => {
    const first = startTurn('Wait for the build.');
    const query = state.queries[0]!;
    const input = await query.nextInput();
    query.send(
      result({
        result: 'The background build finished.',
        user_message_uuids: ['cli-meta-turn'],
        queued_turn_count: 1,
      }),
    );
    await flush();
    expect(first.emitted.some(event => event.type === 'finish')).toBe(false);

    query.send(result({ user_message_uuids: [input.uuid] }));
    await first.done;
    expect(first.emitted.filter(event => event.type === 'finish')).toHaveLength(
      1,
    );
  });

  test('a turn with different options replaces the query and resumes the conversation', async () => {
    const first = startTurn('Hello.');
    const query = state.queries[0]!;
    const input = await query.nextInput();
    query.send(result({ user_message_uuid: input.uuid }));
    await first.done;

    const second = startTurn('Use another model.', { model: 'opus' });
    const replacement = state.queries[1]!;
    const secondInput = await replacement.nextInput();
    replacement.send(result({ user_message_uuid: secondInput.uuid }));
    await second.done;

    expect(query.close).toHaveBeenCalled();
    expect(replacement.args.options).toMatchObject({
      model: 'opus',
      resume: 'claude-session-1',
    });
  });

  test('a host abort interrupts the CLI turn and keeps the process', async () => {
    const first = startTurn('Run the whole test suite.');
    const query = state.queries[0]!;
    await query.nextInput();
    first.abort.abort();
    await flush();
    expect(query.interrupt).toHaveBeenCalledOnce();
    query.send({
      type: 'result',
      subtype: 'error_during_execution',
      errors: ['interrupted'],
      session_id: 'claude-session-1',
    });
    await first.done;
    expect(first.errors).toEqual([]);
    expect(query.close).not.toHaveBeenCalled();

    const second = startTurn('Something else.');
    const input = await query.nextInput();
    query.send(result({ user_message_uuid: input.uuid }));
    await second.done;
    expect(state.queries).toHaveLength(1);
  });

  test('closes the query when the interrupt fails', async () => {
    const first = startTurn('Run forever.');
    const query = state.queries[0]!;
    query.interrupt.mockRejectedValueOnce(new Error('no control channel'));
    await query.nextInput();
    first.abort.abort();
    await first.done;
    expect(query.close).toHaveBeenCalled();
    expect(first.errors).toEqual([]);
  });

  test('a failed result ends the turn but keeps the process', async () => {
    const first = startTurn('Do something.');
    const query = state.queries[0]!;
    const input = await query.nextInput();
    query.send(
      result({
        is_error: true,
        result: 'API Error: overloaded',
        user_message_uuid: input.uuid,
      }),
    );
    await first.done;
    expect(first.errors).toHaveLength(1);
    expect(query.close).not.toHaveBeenCalled();
  });

  test('a fatal stream error closes the process; the next turn resumes in a new one', async () => {
    const first = startTurn('Do something.');
    const query = state.queries[0]!;
    await query.nextInput();
    query.send({
      type: 'system',
      subtype: 'init',
      session_id: 'claude-session-1',
    });
    query.send({
      type: 'system',
      subtype: 'api_retry',
      error_status: 401,
      error: 'invalid api key',
    });
    await first.done;
    expect(first.errors).toHaveLength(1);
    expect(query.close).toHaveBeenCalled();

    const second = startTurn('Try again.');
    const replacement = state.queries[1]!;
    const input = await replacement.nextInput();
    replacement.send(result({ user_message_uuid: input.uuid }));
    await second.done;
    expect(replacement.args.options).toMatchObject({
      resume: 'claude-session-1',
    });
  });

  test('ends the turn with an error when the CLI exits mid-turn', async () => {
    const first = startTurn('Do something.');
    const query = state.queries[0]!;
    await query.nextInput();
    query.end();
    await first.done;
    expect(first.errors).toHaveLength(1);
  });

  test('stopping the bridge closes the query', async () => {
    const first = startTurn('Hello.');
    const query = state.queries[0]!;
    const input = await query.nextInput();
    query.send(result({ user_message_uuid: input.uuid }));
    await first.done;

    expect(await state.onStop!()).toEqual({
      claudeSessionId: 'claude-session-1',
    });
    expect(query.close).toHaveBeenCalled();
  });

  test('a per-turn start closes a live query first', async () => {
    const first = startTurn('Hello.');
    const query = state.queries[0]!;
    const input = await query.nextInput();
    query.send(result({ user_message_uuid: input.uuid }));
    await first.done;

    startTurn('Without persistence.', { persistentQuery: undefined });
    expect(query.close).toHaveBeenCalled();
  });
});
