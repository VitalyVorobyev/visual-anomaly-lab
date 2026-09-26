/**
 * Which method a new experiment starts from, and the order the rest are offered in.
 *
 * Both answers come from the registry (`status`, `recommended_for`), which records the
 * verdicts of the public gates; nothing here names a method. The order is the one a reader
 * should consider them in: the task's default, then what cleared its gate, then what has
 * not, and the numpy floor last — it is the thing the others are measured against, not a
 * place to start.
 */

import type { ModelDescription, Task } from "./client";

const STATUS_RANK: Record<ModelDescription["status"], number> = {
  supported: 1,
  experimental: 2,
  floor: 3,
};

export function isRecommended(method: ModelDescription, task: Task): boolean {
  return method.recommended_for.includes(task);
}

function rank(method: ModelDescription, task: Task): number {
  return isRecommended(method, task) ? 0 : STATUS_RANK[method.status];
}

/** The task's methods, recommended first and the floor last; ties keep registry order. */
export function orderMethods(methods: readonly ModelDescription[], task: Task): ModelDescription[] {
  return methods
    .map((method, index) => ({ method, index }))
    .sort((a, b) => rank(a.method, task) - rank(b.method, task) || a.index - b.index)
    .map(({ method }) => method);
}

/** The method a new experiment of this task starts from: the recommended one, else the first. */
export function defaultMethod(
  methods: readonly ModelDescription[],
  task: Task,
): ModelDescription | undefined {
  return orderMethods(methods, task)[0];
}

/**
 * An options schema with only the fields that apply to one task.
 *
 * A field that reads only one kind of result declares it with `x-tasks`; an unmarked field
 * applies to every task. The backend owns that list, so a new task or a new field needs no
 * change here.
 */
export function schemaForTask<S extends Record<string, unknown>>(schema: S, task: Task): S {
  const properties = schema.properties as Record<string, { "x-tasks"?: unknown }> | undefined;
  if (!properties) return schema;
  const kept = Object.fromEntries(
    Object.entries(properties).filter(([, node]) => {
      const tasks = node["x-tasks"];
      return !Array.isArray(tasks) || tasks.includes(task);
    }),
  );
  return { ...schema, properties: kept };
}
