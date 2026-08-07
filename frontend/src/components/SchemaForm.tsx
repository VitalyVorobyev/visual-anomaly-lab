/**
 * The options form, generated from a plugin's own JSON Schema.
 *
 * The point is that adding an import adapter or a method costs a Python module and a
 * registry entry and *nothing here* — which is what ADR-0006 and ADR-0007 promise and what
 * a hardcoded form quietly took away. Every decision about which control a field deserves
 * lives in `api/schemaForm.ts`; this file only renders.
 *
 * Two behaviours are worth knowing while reading the markup:
 *
 *   * An empty control means **unset**, and the backend's default applies. The default is
 *     shown — as a placeholder, as the pre-highlighted segment, or as the picker's leading
 *     entry — so an operator can see what happens if they say nothing, without that value
 *     being claimed as their choice.
 *   * A field the schema requires is marked, and the caller blocks the run on it.
 *
 * The layout is a two-column grid because the alternative was measured: eight EfficientAD
 * hyperparameters, each a full-width control over a full-width description, ran to two
 * screens of scrolling for a form where nothing needed to be touched at all. Fields whose
 * value is genuinely wide — a list, a JSON object, a statement with a switch — still span
 * both columns.
 */

import {
  Disclosure,
  Field,
  NumberInput,
  SegmentedControl,
  Select,
  Switch,
  Textarea,
  Input,
} from "./ui";
import type { FieldSpec, RawValues } from "../api/schemaForm";

/**
 * Which kinds are worth a full row. Everything else pairs up.
 *
 * A JSON object needs the width to be editable at all, and a switch reads as a sentence
 * with its explanation beside it. A comma-separated list does *not* qualify: three of them
 * in a row is how `folder_classes` ended up as tall as the form this layout was meant to
 * fix.
 */
const WIDE: ReadonlySet<FieldSpec["kind"]> = new Set(["json", "boolean"]);

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
    return <p className="text-sm text-fg-muted">This plugin takes no options.</p>;
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

  // Where nothing is required — the normal shape of a *model's* hyperparameters, as against
  // an adapter's "which folder holds the good images" — there is nothing to fold away
  // from. Hiding every field behind a disclosure would leave a box containing only a
  // disclosure, so they simply render.
  if (primary.length === 0) {
    return <Grid>{advanced.map(render)}</Grid>;
  }

  return (
    <div className="flex flex-col gap-4">
      <Grid>{primary.map(render)}</Grid>
      {advanced.length > 0 && (
        <Disclosure summary="Advanced" count={advanced.length}>
          <Grid>{advanced.map(render)}</Grid>
        </Disclosure>
      )}
    </div>
  );
}

function Grid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-1 gap-x-5 gap-y-4 sm:grid-cols-2">{children}</div>;
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
  const span = WIDE.has(field.kind) ? "sm:col-span-2" : undefined;

  // A switch reads as a statement and carries its own label, rather than sitting under one.
  // It is also the one control with no "unset" position.
  if (field.kind === "boolean") {
    return (
      <div className={span}>
        <Switch
          checked={value === true}
          onCheckedChange={onChange}
          label={field.label}
          description={field.description}
        />
      </div>
    );
  }

  const text = typeof value === "string" ? value : "";
  const fallback = field.fallback === undefined || field.fallback === null
    ? undefined
    : String(field.fallback);

  if (field.kind === "choice-inline") {
    return (
      <Field
        as="group"
        label={field.label}
        description={field.description}
        required={field.required}
        className={span}
      >
        <SegmentedControl
          aria-label={field.label}
          value={text}
          defaultValue={fallback}
          options={field.options}
          onValueChange={onChange}
        />
      </Field>
    );
  }

  if (field.kind === "choice") {
    return (
      <Field
        as="group"
        label={field.label}
        description={field.description}
        required={field.required}
        className={span}
      >
        <Select
          aria-label={field.label}
          value={text}
          onValueChange={onChange}
          options={field.options}
          placeholder={fallback === undefined ? "Choose…" : `Default · ${fallback}`}
          unsetLabel={fallback === undefined ? undefined : `Default · ${fallback}`}
        />
      </Field>
    );
  }

  return (
    <Field
      label={field.label}
      description={
        field.kind === "string-list"
          ? [field.description, "Comma-separated."].filter(Boolean).join(" ")
          : field.description
      }
      required={field.required}
      annotation={field.range}
      className={span}
    >
      {field.kind === "json" ? (
        <Textarea
          rows={2}
          aria-label={field.label}
          placeholder={field.placeholder}
          value={text}
          onChange={(event) => onChange(event.target.value)}
        />
      ) : field.kind === "number" ? (
        <NumberInput
          aria-label={field.label}
          placeholder={field.placeholder}
          min={field.min}
          max={field.max}
          step={field.step}
          value={text}
          onChange={(event) => onChange(event.target.value)}
        />
      ) : (
        <Input
          className="font-mono"
          aria-label={field.label}
          placeholder={field.placeholder}
          value={text}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </Field>
  );
}
