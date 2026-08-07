/**
 * Turning an adapter's JSON Schema into a form, and a form back into options.
 *
 * ADR-0007 and system-design §5 both say an adapter's options model "drives the UI form",
 * and until now that was a claim rather than a mechanism: the import screen hardcoded one
 * option and relied on Python defaults for the rest. That was survivable while a single
 * adapter existed with defaults that fit the one dataset on hand. It stops being
 * survivable the moment an adapter has a *required* option nobody can type.
 *
 * The logic lives here rather than in the component because it is where the decisions
 * are: what control a schema node deserves, what an empty box means, and when a value is
 * worth sending at all. Rendering is the easy half.
 *
 * Only the shapes pydantic actually emits are handled. A node this does not recognise
 * becomes a JSON textarea rather than disappearing — an operator can always express what
 * the model accepts, even where the form has nothing prettier to offer.
 */

/** The subset of JSON Schema that pydantic emits for an options model. */
export type SchemaNode = {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  items?: SchemaNode;
  anyOf?: SchemaNode[];
  additionalProperties?: SchemaNode | boolean;
  enum?: unknown[];
};

export type OptionsSchema = {
  properties?: Record<string, SchemaNode>;
  required?: string[];
};

export type FieldKind = "text" | "number" | "boolean" | "string-list" | "json";

export type FieldSpec = {
  name: string;
  label: string;
  description: string | null;
  kind: FieldKind;
  required: boolean;
  /** Rendered into the empty control as a hint, never submitted on the user's behalf. */
  placeholder: string;
  /** What the backend uses when the field is left empty. Shown, not sent. */
  fallback: unknown;
  /** Behind a disclosure: the adapter already has a working answer for this one. */
  advanced: boolean;
};

/**
 * Which options an operator has to think about, and which already have an answer.
 *
 * The schema says this without anyone having to maintain a second list: a field whose
 * default is *empty* — `[]`, `null`, `""`, or absent entirely — is one the adapter cannot
 * do anything useful with until you fill it in. A field with a real default already
 * works, and the researcher asking "where are the good images" should not have to read
 * past nine vocabulary options to answer it.
 *
 * The rule earns itself on all three adapters: it promotes `normal_dirs` and `defect_dirs`
 * for `folder_classes`, `csv_path` and the filter pair for `csv_table`, and `channels` for
 * `channel_folders`, while demoting every label and extension vocabulary.
 */
function isAdvanced(node: SchemaNode): boolean {
  const fallback = node.default;
  if (fallback === undefined || fallback === null) return false;
  if (Array.isArray(fallback)) return fallback.length > 0;
  if (fallback === "") return false;
  if (typeof fallback === "object") return Object.keys(fallback).length > 0;
  return true;
}

/** Raw control state: what the operator typed, not what will be sent. */
export type RawValues = Record<string, string | boolean>;

/**
 * Look through pydantic's `str | None` encoding to the type that carries the meaning.
 *
 * An optional string arrives as `anyOf: [{type: "string"}, {type: "null"}]`, and treating
 * that as an unrecognised node would give every nullable option a JSON textarea.
 */
function unwrap(node: SchemaNode): SchemaNode {
  if (!node.anyOf) return node;
  const meaningful = node.anyOf.filter((branch) => branch.type !== "null");
  return meaningful.length === 1 ? { ...node, ...meaningful[0] } : node;
}

function kindOf(node: SchemaNode): FieldKind {
  const resolved = unwrap(node);
  if (resolved.type === "boolean") return "boolean";
  if (resolved.type === "integer" || resolved.type === "number") return "number";
  if (resolved.type === "string") return "text";
  if (resolved.type === "array" && unwrap(resolved.items ?? {}).type === "string") {
    return "string-list";
  }
  return "json";
}

/**
 * Humanise `normal_dirs` into `Normal dirs`.
 *
 * Pydantic emits a `title` for every field whether or not anyone wrote one, and its
 * generated form is Title Case — `Csv Path`, `Defect Type From Dir` — which reads like a
 * spreadsheet header rather than a form label. So a title is used only when it is *not*
 * the one pydantic would have generated, which is exactly when a human chose it.
 */
function labelFor(name: string, node: SchemaNode): string {
  const words = name.split("_");
  const generated = words.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
  if (node.title && node.title !== generated) return node.title;

  const spaced = words.join(" ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function placeholderFor(kind: FieldKind, fallback: unknown): string {
  if (kind === "string-list") {
    return Array.isArray(fallback) && fallback.length > 0 ? fallback.join(", ") : "a, b, c";
  }
  if (kind === "json") return "{}";
  if (fallback === null || fallback === undefined) return "";
  return String(fallback);
}

export function describeFields(schema: OptionsSchema): FieldSpec[] {
  const required = new Set(schema.required ?? []);
  return Object.entries(schema.properties ?? {}).map(([name, node]) => {
    const kind = kindOf(node);
    return {
      name,
      label: labelFor(name, node),
      description: node.description ?? null,
      kind,
      required: required.has(name),
      fallback: node.default,
      placeholder: placeholderFor(kind, node.default),
      advanced: !required.has(name) && isAdvanced(node),
    };
  });
}

/**
 * The controls' starting state.
 *
 * Text and list controls start **empty even when the schema has a default**, and that is
 * deliberate: a pre-filled box says "this is your value" when what is true is "this is
 * what happens if you say nothing". Pre-filling would also mean every scan sent every
 * default back, so a later change to a default would silently not reach anyone who had
 * once opened the form. The default is shown as the placeholder instead.
 *
 * Booleans are the exception — a checkbox has no empty state, so it starts at the
 * default and is always sent.
 */
export function initialValues(fields: FieldSpec[]): RawValues {
  const values: RawValues = {};
  for (const field of fields) {
    values[field.name] = field.kind === "boolean" ? field.fallback === true : "";
  }
  return values;
}

/**
 * Build the options object to send.
 *
 * An untouched control contributes **nothing**, so the backend's own default applies and
 * stays the single definition of it. That is what keeps a default from existing twice, in
 * Python and in TypeScript, with the two free to drift.
 */
export function toOptions(fields: FieldSpec[], values: RawValues): Record<string, unknown> {
  const options: Record<string, unknown> = {};

  for (const field of fields) {
    const raw = values[field.name];

    if (field.kind === "boolean") {
      // Always sent: a checkbox cannot express "unset", so leaving it out would make its
      // rendered state and the value in force disagree whenever the default is `true`.
      options[field.name] = raw === true;
      continue;
    }

    const text = typeof raw === "string" ? raw.trim() : "";
    if (text === "") continue;

    switch (field.kind) {
      case "string-list":
        options[field.name] = text
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
        break;
      case "number": {
        const parsed = Number(text);
        if (Number.isFinite(parsed)) options[field.name] = parsed;
        break;
      }
      case "json":
        try {
          options[field.name] = JSON.parse(text);
        } catch {
          // Left out rather than sent as a string: the backend would reject it with a
          // less useful message than the one `jsonErrors` puts next to the field.
        }
        break;
      default:
        options[field.name] = text;
    }
  }

  return options;
}

/** Which JSON fields do not parse, so the form can say so before the scan starts. */
export function jsonErrors(fields: FieldSpec[], values: RawValues): string[] {
  return fields
    .filter((field) => field.kind === "json")
    .filter((field) => {
      const raw = values[field.name];
      if (typeof raw !== "string" || raw.trim() === "") return false;
      try {
        JSON.parse(raw);
        return false;
      } catch {
        return true;
      }
    })
    .map((field) => field.label);
}

/** Required fields the operator has not filled in. Blocks the scan button. */
export function missingRequired(fields: FieldSpec[], values: RawValues): string[] {
  return fields
    .filter((field) => field.required)
    .filter((field) => {
      const raw = values[field.name];
      if (typeof raw === "boolean") return false;
      return (raw ?? "").trim() === "";
    })
    .map((field) => field.label);
}
