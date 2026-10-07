/**
 * An open-ended async iterable fed by `push`: the streaming input of a query
 * that outlives a single turn. Iteration waits for the next pushed value and
 * ends once `close` is called and the queue has drained.
 */
export type PushableInput<T> = AsyncIterable<T> & {
  push(value: T): void;
  close(): void;
  readonly closed: boolean;
};

export function createPushableInput<T>(): PushableInput<T> {
  const queue: T[] = [];
  const waiters: Array<(result: IteratorResult<T>) => void> = [];
  let closed = false;

  return {
    get closed() {
      return closed;
    },
    push(value) {
      if (closed) return;
      const waiter = waiters.shift();
      if (waiter != null) {
        waiter({ done: false, value });
      } else {
        queue.push(value);
      }
    },
    close() {
      if (closed) return;
      closed = true;
      while (waiters.length > 0) {
        waiters.shift()!({ done: true, value: undefined });
      }
    },
    [Symbol.asyncIterator]() {
      return {
        next: () => {
          if (queue.length > 0) {
            return Promise.resolve({ done: false, value: queue.shift()! });
          }
          if (closed) {
            return Promise.resolve({ done: true, value: undefined });
          }
          return new Promise<IteratorResult<T>>(resolve => {
            waiters.push(resolve);
          });
        },
      };
    },
  };
}
