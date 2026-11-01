/** Previous values of the properties that changed, keyed by name (the shape of Lit's changedProperties). */
export type Changes = Map<string, unknown>;

/**
 * Turns the named fields of `obj` into accessors. Assignments that change a value are batched,
 * and `onChange` gets them in a microtask, with each key mapped to the value it had before the batch.
 */
export function reactive<T extends object>(obj: T, keys: readonly (keyof T & string)[], onChange: (changes: Changes) => void) {
  let pending: Changes | null = null;
  for (const key of keys) {
    let value = obj[key];
    Object.defineProperty(obj, key, {
      enumerable: true,
      configurable: true,
      get: () => value,
      set: (v) => {
        if (Object.is(v, value)) return;
        if (!pending) {
          pending = new Map();
          queueMicrotask(() => {
            const changes = pending!;
            pending = null;
            onChange(changes);
          });
        }
        if (!pending.has(key)) pending.set(key, value);
        value = v;
      },
    });
  }
}
