/**
 * The colours a new annotation class is given — data, not chrome.
 *
 * A class's colour is stored with the class (`AnnotationLabel.color`, a `#rrggbb` the
 * backend keeps and the reader can change with a colour picker), so it is a value, not a
 * design token: it has to be the same literal in both themes and on the server. Hence the
 * one file in `src/` exempt from `tokensOnly` for this reason (see `eslint.config.js`).
 */

/** Distinct in both themes and from each other; a new class takes the first one unused. */
export const CLASS_PALETTE = [
  "#e8590c",
  "#1c7ed6",
  "#2f9e44",
  "#ae3ec9",
  "#f59f00",
  "#0c8599",
  "#d6336c",
  "#5c940d",
] as const;

export function nextClassColour(labels: readonly { color: string }[]): string {
  const used = new Set(labels.map((label) => label.color.toLowerCase()));
  return CLASS_PALETTE.find((colour) => !used.has(colour)) ?? CLASS_PALETTE[labels.length % CLASS_PALETTE.length]!;
}
