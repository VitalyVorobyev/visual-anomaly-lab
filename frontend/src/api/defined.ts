/**
 * Drop the keys whose value is `undefined`.
 *
 * With `exactOptionalPropertyTypes` (from `@vitavision/config-ts`), `{ note: undefined }` is
 * not a `{ note?: string }`. Our own types say `?: T | undefined` where a value may be
 * absent; this is for the props and request bodies typed elsewhere — lab-ui, Konva, the
 * generated API client — so a call site can pass "maybe a value" without a spread per key.
 * The result is what those call sites always meant: the key is left out.
 */

type Defined<T> = { [K in keyof T as undefined extends T[K] ? never : K]: T[K] } & {
  [K in keyof T as undefined extends T[K] ? K : never]?: Exclude<T[K], undefined>;
};

export function defined<T extends object>(value: T): Defined<T> {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as Defined<T>;
}
