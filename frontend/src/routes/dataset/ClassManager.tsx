/**
 * A dataset's annotation classes: add one, rename it, recolour it, reorder it.
 *
 * The taxonomy has had a table and create/update routes since annotation shipped, and no
 * screen — so every dataset had exactly the one `defect` class the backend seeds, and the
 * editor's class picker (shown only when there is more than one) was unreachable. A
 * segmentation or detection task is a taxonomy first (ADR-0039), so this is where it starts.
 *
 * The key is the class's identity: it is what every region stores, so it is derived once
 * from the name at creation and never changes. A rename changes what readers see and
 * nothing a region points at. There is no delete, because the backend has none — a class
 * that regions still name cannot simply disappear.
 */

import { ArrowDown, ArrowUp, Plus } from "lucide-react";
import { useState } from "react";

import type { AnnotationLabel } from "../../api/client";
import { Button, Disclosure, ErrorBox, Field, Input } from "@vitavision/lab-ui";
import {
  useAnnotationLabels,
  useCreateAnnotationLabel,
  useUpdateAnnotationLabel,
} from "../../hooks/useAnnotations";

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

/** `Surface scratch` → `surface_scratch`, made unique against the keys already taken. */
export function classKeyFor(name: string, taken: readonly string[]): string {
  let base = name
    .toLowerCase()
    .normalize("NFKD")
    // The accents NFKD split off, so `ü` keeps its letter instead of becoming `u_`.
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "")
    .slice(0, 50);
  if (base === "") base = "class";
  // A key has to start with a letter; `3D dent` keeps its digit rather than losing it.
  else if (!/^[a-z]/.test(base)) base = `class_${base}`;
  let key = base;
  for (let n = 2; taken.includes(key); n += 1) key = `${base}_${n}`;
  return key;
}

export function nextClassColour(labels: readonly { color: string }[]): string {
  const used = new Set(labels.map((label) => label.color.toLowerCase()));
  return CLASS_PALETTE.find((colour) => !used.has(colour)) ?? CLASS_PALETTE[labels.length % CLASS_PALETTE.length]!;
}

export function ClassManager({ datasetId }: { datasetId: number }) {
  const labels = useAnnotationLabels(datasetId);
  const create = useCreateAnnotationLabel(datasetId);
  const update = useUpdateAnnotationLabel(datasetId);
  const [name, setName] = useState("");
  const rows = labels.data ?? [];
  const key = classKeyFor(name, rows.map((label) => label.key));

  const move = (index: number, by: -1 | 1) => {
    const a = rows[index];
    const b = rows[index + by];
    if (!a || !b) return;
    // Positions may tie (every seeded class starts at 0), so reorder by index, not by swap.
    const order = rows.map((label) => label.key);
    order.splice(index, 1);
    order.splice(index + by, 0, a.key);
    order.forEach((labelKey, position) => {
      const label = rows.find((entry) => entry.key === labelKey);
      if (label && label.position !== position) update.mutate({ ...label, position });
    });
  };

  const add = () => {
    if (name.trim() === "") return;
    create.mutate(
      {
        key,
        name: name.trim(),
        color: nextClassColour(rows),
        position: rows.length,
      },
      { onSuccess: () => setName("") },
    );
  };

  return (
    <Disclosure summary="Classes" count={rows.length}>
      <div className="flex flex-col gap-3">
        <p className="text-xs text-fg-muted">
          A region stores its class's key, so renaming or recolouring a class changes every
          region that uses it and loses none. The editor offers a class picker once there is
          more than one.
        </p>

        <ul className="flex flex-col divide-y divide-line rounded-control border border-line">
          {rows.map((label, index) => (
            <ClassRow
              key={label.key}
              label={label}
              first={index === 0}
              last={index === rows.length - 1}
              onChange={(next) => update.mutate(next)}
              onMove={(by) => move(index, by)}
            />
          ))}
        </ul>

        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            add();
          }}
        >
          <Field
            label="New class"
            description={name.trim() ? <span className="font-mono">key {key}</span> : undefined}
          >
            <Input
              aria-label="New class name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Surface scratch"
              className="w-64"
            />
          </Field>
          <Button
            type="submit"
            icon={<Plus />}
            disabled={name.trim() === ""}
            loading={create.isPending}
          >
            Add class
          </Button>
        </form>

        {labels.error && <ErrorBox>{labels.error.message}</ErrorBox>}
        {create.error && <ErrorBox>{create.error.message}</ErrorBox>}
        {update.error && <ErrorBox>{update.error.message}</ErrorBox>}
      </div>
    </Disclosure>
  );
}

function ClassRow({
  label,
  first,
  last,
  onChange,
  onMove,
}: {
  label: AnnotationLabel;
  first: boolean;
  last: boolean;
  onChange: (label: AnnotationLabel) => void;
  onMove: (by: -1 | 1) => void;
}) {
  const [name, setName] = useState(label.name);
  const commitName = () => {
    const trimmed = name.trim();
    if (trimmed === "") setName(label.name);
    else if (trimmed !== label.name) onChange({ ...label, name: trimmed });
  };

  return (
    <li className="flex items-center gap-3 px-3 py-2">
      <input
        type="color"
        aria-label={`Colour of ${label.name}`}
        value={label.color}
        onChange={(event) => onChange({ ...label, color: event.target.value })}
        className="size-6 shrink-0 cursor-pointer rounded-control border border-line bg-transparent"
      />
      <Input
        aria-label={`Name of ${label.key}`}
        value={name}
        onChange={(event) => setName(event.target.value)}
        onBlur={commitName}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
        className="max-w-64"
      />
      <span className="font-mono text-[11px] text-fg-subtle">{label.key}</span>
      <span className="ml-auto flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          icon={<ArrowUp />}
          aria-label={`Move ${label.name} up`}
          disabled={first}
          onClick={() => onMove(-1)}
        />
        <Button
          variant="ghost"
          size="sm"
          icon={<ArrowDown />}
          aria-label={`Move ${label.name} down`}
          disabled={last}
          onClick={() => onMove(1)}
        />
      </span>
    </li>
  );
}
