/**
 * Every key and gesture the editor answers, opened with `?`.
 *
 * Rendered from `EDITOR_BINDINGS`, the same list `useEditorKeymap` resolves keystrokes
 * against, so the sheet cannot describe a key that does nothing or miss one that does.
 */

import { Fragment } from "react";

import {
  EDITOR_BINDINGS,
  type EditorBinding,
  type KeyGroup,
} from "../../components/annotation/editorKeys";
import { Dialog } from "@vitavision/lab-ui";

const ORDER: KeyGroup[] = ["Tools", "Document", "Navigation", "View", "Label", "Canvas", "Pointer"];

const HEADINGS: Record<KeyGroup, string> = {
  Tools: "Tools",
  Document: "Document",
  Navigation: "Navigation",
  View: "View",
  Label: "Sample label",
  Canvas: "With the canvas focused",
  Pointer: "Pointer",
};

export function ShortcutSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const groups = ORDER.map((group) => ({
    group,
    bindings: EDITOR_BINDINGS.filter((binding) => binding.group === group),
  })).filter((entry) => entry.bindings.length > 0);

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Keyboard shortcuts"
      description="⌘ is Ctrl outside macOS. Letters ignore Shift unless a row says otherwise."
    >
      <div className="-mr-2 mt-3 flex max-h-[65vh] flex-col gap-4 overflow-y-auto pr-2">
        {groups.map(({ group, bindings }) => (
          <section key={group} aria-label={HEADINGS[group]}>
            <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-fg-muted">
              {HEADINGS[group]}
            </h3>
            <dl className="grid grid-cols-[6.5rem_1fr] items-baseline gap-x-3 gap-y-1 text-xs">
              {bindings.map((binding) => (
                <Row key={`${binding.scope}-${binding.keys.join("")}`} binding={binding} />
              ))}
            </dl>
          </section>
        ))}
      </div>
    </Dialog>
  );
}

function Row({ binding }: { binding: EditorBinding }) {
  return (
    <>
      <dt className="flex flex-wrap gap-1 whitespace-nowrap">
        {binding.keys.map((key, index) => (
          <Fragment key={key}>
            {index > 0 && <span className="text-fg-subtle">/</span>}
            <kbd className="rounded-control border border-line bg-raised px-1.5 font-mono text-[10px] text-fg">
              {key}
            </kbd>
          </Fragment>
        ))}
      </dt>
      <dd className="text-fg-muted">{binding.description}</dd>
    </>
  );
}
