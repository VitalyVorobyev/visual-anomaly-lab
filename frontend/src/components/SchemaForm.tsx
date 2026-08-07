/**
 * The adapter options form, generated from the adapter's own JSON Schema.
 *
 * The point is that adding an import adapter costs a Python module and a registry entry
 * and *nothing here* — which is what ADR-0006 promises and what a hardcoded form quietly
 * took away. Every decision about which control a field deserves lives in
 * `api/schemaForm.ts`; this file only renders.
 *
 * Two behaviours are worth knowing while reading the markup:
 *
 *   * An empty box means **unset**, and the backend's default applies. The default is
 *     shown as a placeholder so an operator can see what happens if they say nothing,
 *     without that value being claimed as their choice.
 *   * A field the schema requires is marked, and the caller blocks the scan on it. The
 *     first adapter with a required option is what made this necessary.
 */

import { Field, inputClasses } from "./ui";
import type { FieldSpec, RawValues } from "../api/schemaForm";

export function SchemaForm({
  fields,
  values,
  onChange,
}: {
  fields: FieldSpec[];
  values: RawValues;
  onChange: (next: RawValues) => void;
}) {
  if (fields.length === 0) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">
        This adapter takes no options.
      </p>
    );
  }

  const set = (name: string, value: string | boolean) => onChange({ ...values, [name]: value });
  const render = (field: FieldSpec) => (
    <SchemaField
      key={field.name}
      field={field}
      value={values[field.name] ?? ""}
      onChange={(value) => set(field.name, value)}
    />
  );

  const primary = fields.filter((field) => !field.advanced);
  const advanced = fields.filter((field) => field.advanced);

  // Every option has a working default — which is the normal shape of a *model's*
  // hyperparameters, where an adapter's "which folder holds the good images" has no
  // sensible default at all. Folding all of them away would leave a box containing
  // nothing but a disclosure, so they open instead, under a line saying they are
  // optional. The rule stays the same; only what it does with an empty primary set.
  const startOpen = primary.length === 0;

  return (
    <div className="flex flex-col gap-4">
      {primary.map(render)}

      {/* Folded away rather than dropped: every option stays reachable, but "where are
          the good images" should not be the tenth question on the screen. */}
      {advanced.length > 0 && (
        <details
          open={startOpen}
          className="rounded border border-slate-200 px-3 py-2 dark:border-slate-700"
        >
          <summary className="cursor-pointer text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
            {startOpen
              ? `${advanced.length} options, every one already set to a working default`
              : `${advanced.length} more, already set to a working default`}
          </summary>
          <div className="mt-3 flex flex-col gap-4">{advanced.map(render)}</div>
        </details>
      )}
    </div>
  );
}

function SchemaField({
  field,
  value,
  onChange,
}: {
  field: FieldSpec;
  value: string | boolean;
  onChange: (value: string | boolean) => void;
}) {
  // A checkbox reads as a statement, so it carries its own label rather than sitting
  // under one — and it is the one control with no "unset" position.
  if (field.kind === "boolean") {
    return (
      <div className="flex flex-col gap-1">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            aria-label={field.label}
            checked={value === true}
            onChange={(event) => onChange(event.target.checked)}
          />
          <span>{field.label}</span>
        </label>
        {field.description && <Hint>{field.description}</Hint>}
      </div>
    );
  }

  const text = typeof value === "string" ? value : "";
  const label = field.required ? `${field.label} (required)` : field.label;

  return (
    <div className="flex flex-col gap-1">
      <Field label={label}>
        {field.kind === "json" ? (
          <textarea
            className={`${inputClasses} font-mono`}
            rows={2}
            aria-label={field.label}
            placeholder={field.placeholder}
            value={text}
            onChange={(event) => onChange(event.target.value)}
          />
        ) : (
          <input
            className={`${inputClasses} ${field.kind === "number" ? "" : "font-mono"}`}
            type={field.kind === "number" ? "number" : "text"}
            aria-label={field.label}
            placeholder={field.placeholder}
            value={text}
            onChange={(event) => onChange(event.target.value)}
          />
        )}
      </Field>
      {field.description && <Hint>{field.description}</Hint>}
      {field.kind === "string-list" && <Hint>Comma-separated.</Hint>}
    </div>
  );
}

function Hint({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-slate-500 dark:text-slate-400">{children}</p>;
}
