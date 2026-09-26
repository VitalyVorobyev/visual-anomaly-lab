/**
 * The UI rules from CLAUDE.md that a grep can hold, held as a ratchet.
 *
 * Each rule is a pattern that must not appear in application source. The ones that still
 * do are listed in `KNOWN` with their count, so the suite is green today and fails the
 * moment a new occurrence lands — or the moment one is fixed and its allowance is not
 * lowered, which keeps the list honest instead of letting it rot into a permission slip.
 * The goal is an empty `KNOWN`.
 *
 * Comments are stripped first: several files explain *why* they do not use a `<select>`
 * or a `<details>`, and that explanation is not a violation.
 */
import { describe, expect, it } from "vitest";

const RULES = {
  /** A raw `<table>` instead of lab-ui's `Table`, which carries the density and the rules. */
  "raw-table": /<table\b/g,
  /** A literal colour ignores the theme; tokens are `surface`, `line`, `fg-muted`, `signal`… */
  "hex-colour": /\b(?:bg|text|border|ring|fill|stroke|from|to|via|outline|shadow)-\[#/g,
  /** A raw Tailwind ramp step compiles, looks almost right, and quietly ignores the theme. */
  "ramp-step":
    /\b(?:bg|text|border|ring|fill|stroke|from|to|via|outline|divide|placeholder|accent|caret|decoration)-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/g,
  /** A window shortcut goes through `useHotkeys`, which owns the one guard. */
  "bare-keydown": /addEventListener\(\s*["']keydown/g,
  /** Bare form controls: lab-ui has `Select`, `Slider`, `Checkbox`, `Switch`. */
  "bare-control": /<select\b|<input[^>]*type=["'](?:range|checkbox)/g,
  /** A raw `<details>` renders with no caret; use `Disclosure`. */
  "raw-details": /<details\b/g,
  /** A control never nests inside a link — the click has to be cancelled to stop navigation.
   *  lab-ui's `ButtonLink` is the one element that looks like a button and navigates. */
  "link-wraps-button": /<Link\b[^>]*>\s*<Button\b/g,
} as const;

type Rule = keyof typeof RULES;

/** Where each rule is still broken today. Lower a count when you fix one; never raise it. */
const KNOWN: Partial<Record<Rule, Record<string, number>>> = {
  "raw-table": {
    "components/diagnostics/DiagnosticViews.tsx": 1,
    "components/diagnostics/ModuleTree.tsx": 1,
    "routes/ImportRoute.tsx": 2,
    "routes/compare/AgreementTable.tsx": 1,
    "routes/compare/ConfigDiff.tsx": 1,
    "routes/compare/MetricTable.tsx": 1,
    "routes/experiment/ResultsPanel.tsx": 1,
  },
  "hex-colour": {
    "components/JobProgress.tsx": 2,
    "components/diagnostics/DiagnosticViews.tsx": 1,
    "routes/ExperimentSampleRoute.tsx": 1,
    "routes/experiment/GalleryTab.tsx": 1,
  },
  // The one listener every other screen goes through.
  "bare-keydown": { "hooks/useHotkeys.ts": 1 },
};

const sources = import.meta.glob<string>(["./**/*.ts", "./**/*.tsx", "!./**/*.test.*"], {
  query: "?raw",
  import: "default",
  eager: true,
});

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'`])\/\/.*$/gm, "$1");
}

function occurrences(rule: Rule): Record<string, number> {
  const found: Record<string, number> = {};
  for (const [path, source] of Object.entries(sources)) {
    const count = stripComments(source).match(RULES[rule])?.length ?? 0;
    if (count > 0) found[path.replace(/^\.\//, "")] = count;
  }
  return found;
}

describe("UI rules", () => {
  it("reads the application source", () => {
    expect(Object.keys(sources).length).toBeGreaterThan(50);
  });

  for (const rule of Object.keys(RULES) as Rule[]) {
    it(`${rule}: nothing new, and fixed ones are struck off`, () => {
      expect(occurrences(rule)).toEqual(KNOWN[rule] ?? {});
    });
  }
});
