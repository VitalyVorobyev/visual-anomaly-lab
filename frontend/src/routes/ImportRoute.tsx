/**
 * Import: configure, scan, **review**, commit (ADR-0006).
 *
 * The review step is the point of this screen and the reason import is not one click.
 * A fully automatic importer that guessed silently would produce a corrupt catalog
 * discovered much later, after experiments had been run against it — so nothing is
 * written until a human has looked at what the adapter proposed and pressed commit.
 *
 * ADR-0006 also names the failure mode honestly: for a regular dataset this is ceremony,
 * and it will be tempting to click through without reading. The warnings panel is
 * therefore an explicit acknowledgement rather than a notice, and it does not appear at
 * all when there is nothing to say.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { api, unwrap } from "../api/client";
import { Badge, Button, Checkbox, CountRun, Empty, ErrorBox, Field, Input, PageHeader, Panel, SchemaForm, Section, Select, cn, describeFields, initialValues, jsonErrors, missingRequired, toOptions, type OptionsSchema, type RawValues } from "@vitavision/lab-ui";
import type {
  AdapterInfo,
  ChannelMapping,
  CommitResponse,
  Label,
  Manifest,
  ManifestSample,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { hasDirectoryPicker, pickDirectory } from "../api/shell";
import { Check } from "lucide-react";

import { JobProgress } from "../components/JobProgress";
import { WarningsPanel, commitBlocked } from "../components/WarningsPanel";
import { isTerminal, useJob } from "../hooks/useJob";

type Stage =
  | { kind: "configure" }
  | { kind: "scanning"; jobId: number }
  | { kind: "review"; manifestId: string }
  | { kind: "committed"; result: CommitResponse };

const STAGES: { kind: Stage["kind"]; label: string }[] = [
  { kind: "configure", label: "Configure" },
  { kind: "scanning", label: "Scan" },
  { kind: "review", label: "Review" },
  { kind: "committed", label: "Commit" },
];

/**
 * Where you are in the import, and how much is left.
 *
 * The four stages were a state machine with no rendering: each one replaced the screen
 * entirely, so a reader who had just watched a scan finish had no way to tell whether
 * reviewing was the last step or the second of five. Nothing is written until the last one,
 * which is exactly the fact a progress indicator is for.
 */
function Stepper({ stage }: { stage: Stage["kind"] }) {
  const current = STAGES.findIndex((entry) => entry.kind === stage);

  return (
    <ol className="flex flex-wrap items-center gap-2" aria-label="Import progress">
      {STAGES.map((entry, index) => {
        const done = index < current;
        const active = index === current;
        return (
          <li key={entry.kind} className="flex items-center gap-2">
            {index > 0 && <span className="h-px w-6 bg-line" aria-hidden />}
            <span
              aria-current={active ? "step" : undefined}
              className={cn(
                "flex items-center gap-1.5 rounded-control px-2 py-1 text-xs font-medium",
                active && "bg-signal/12 text-signal ring-1 ring-signal/25",
                done && "text-fg-muted",
                !active && !done && "text-fg-subtle",
              )}
            >
              {done ? (
                <Check className="size-3.5" aria-label="done" />
              ) : (
                <span className="font-mono tabular-nums">{index + 1}</span>
              )}
              {entry.label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

export function ImportRoute() {
  const [stage, setStage] = useState<Stage>({ kind: "configure" });

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Import a dataset"
        meta="Images are read where they are and never copied. A scan proposes; nothing is written until you commit."
      />

      <Stepper stage={stage.kind} />

      {stage.kind === "configure" && (
        <ConfigureStep onStarted={(jobId) => setStage({ kind: "scanning", jobId })} />
      )}
      {stage.kind === "scanning" && (
        <ScanStep
          jobId={stage.jobId}
          onScanned={(manifestId) => setStage({ kind: "review", manifestId })}
          onRestart={() => setStage({ kind: "configure" })}
        />
      )}
      {stage.kind === "review" && (
        <ReviewStep
          manifestId={stage.manifestId}
          onCommitted={(result) => setStage({ kind: "committed", result })}
          onRestart={() => setStage({ kind: "configure" })}
        />
      )}
      {stage.kind === "committed" && (
        <CommittedStep result={stage.result} onRestart={() => setStage({ kind: "configure" })} />
      )}
    </div>
  );
}

function ConfigureStep({ onStarted }: { onStarted: (jobId: number) => void }) {
  const [rootPath, setRootPath] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [adapter, setAdapter] = useState("folder_classes");
  const [options, setOptions] = useState<RawValues>({});

  const adapters = useQuery<AdapterInfo[]>({
    queryKey: queryKeys.adapters(),
    queryFn: async () => unwrap(await api.GET("/api/import/adapters"), "the adapter list"),
  });

  const chosen = adapters.data?.find((entry) => entry.name === adapter);
  // Recomputed from the chosen adapter's schema, so switching adapters swaps the whole
  // form rather than leaving one adapter's options attached to another's scan.
  const fields = useMemo(
    () => (chosen ? describeFields(chosen.options_schema as OptionsSchema) : []),
    [chosen],
  );

  const pickAdapter = (name: string) => {
    setAdapter(name);
    const next = adapters.data?.find((entry) => entry.name === name);
    setOptions(next ? initialValues(describeFields(next.options_schema as OptionsSchema)) : {});
  };

  // The first render arrives before the adapter list does, so the defaults for the
  // adapter already selected have to be filled in once it lands.
  useEffect(() => {
    if (fields.length > 0 && Object.keys(options).length === 0) setOptions(initialValues(fields));
  }, [fields, options]);

  const scan = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/import/scan", {
          body: {
            root_path: rootPath.trim(),
            dataset_name: datasetName.trim(),
            adapter,
            options: toOptions(fields, options),
          },
        }),
        "a scan job",
      ),
    onSuccess: (job) => onStarted(job.id),
  });

  const malformed = jsonErrors(fields, options);
  const missing = missingRequired(fields, options);
  const ready =
    rootPath.trim() !== "" &&
    datasetName.trim() !== "" &&
    missing.length === 0 &&
    malformed.length === 0;

  return (
    <Panel title="What to scan">
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault();
          scan.mutate();
        }}
      >
        <Field label="Source directory">
          <div className="flex gap-2">
            <Input
              className="font-mono"
              placeholder="/absolute/path/to/images"
              value={rootPath}
              onChange={(event) => setRootPath(event.target.value)}
            />
            {/* Only the desktop shell can return a path the backend is able to open. */}
            {hasDirectoryPicker() && (
              <Button
                onClick={() => {
                  void pickDirectory().then((chosenPath) => {
                    if (chosenPath) setRootPath(chosenPath);
                  });
                }}
              >
                Browse…
              </Button>
            )}
          </div>
        </Field>

        <Field label="Dataset name">
          <Input
            value={datasetName}
            onChange={(event) => setDatasetName(event.target.value)}
          />
        </Field>

        <Field as="group" label="Adapter" description={chosen?.summary}>
          <Select
            aria-label="Adapter"
            value={adapter}
            onValueChange={pickAdapter}
            options={(adapters.data ?? []).map((entry) => ({
              value: entry.name,
              label: entry.name,
            }))}
          />
        </Field>

        <Section
          title={`${adapter} options`}
          hint="Anything left empty keeps the default shown inside it."
        >
          <SchemaForm fields={fields} values={options} onChange={setOptions} />
        </Section>

        {scan.error && <ErrorBox>{scan.error.message}</ErrorBox>}
        {malformed.length > 0 && (
          <ErrorBox>{malformed.join(", ")} is not valid JSON.</ErrorBox>
        )}

        <div className="flex items-center gap-2">
          <Button type="submit" variant="primary" disabled={!ready || scan.isPending}>
            Scan
          </Button>
          {missing.length > 0 && (
            <span className="text-xs text-warn">
              {missing.join(", ")} {missing.length === 1 ? "is" : "are"} required.
            </span>
          )}
        </div>
      </form>
    </Panel>
  );
}

function ScanStep({
  jobId,
  onScanned,
  onRestart,
}: {
  jobId: number;
  onScanned: (manifestId: string) => void;
  onRestart: () => void;
}) {
  const { job, lines, error } = useJob(jobId);
  const manifestId = job?.result?.["manifest_id"] as string | undefined;
  const done = isTerminal(job?.status);

  return (
    <Panel title="Scanning">
      <div className="flex flex-col gap-4">
        <JobProgress jobId={jobId} job={job} lines={lines} error={error} />
        <div className="flex gap-2">
          {done && manifestId && (
            <Button variant="primary" onClick={() => onScanned(manifestId)}>
              Review the proposal
            </Button>
          )}
          {done && <Button onClick={onRestart}>Start over</Button>}
        </div>
      </div>
    </Panel>
  );
}

function ReviewStep({
  manifestId,
  onCommitted,
  onRestart,
}: {
  manifestId: string;
  onCommitted: (result: CommitResponse) => void;
  onRestart: () => void;
}) {
  const queryClient = useQueryClient();
  const [edited, setEdited] = useState<Manifest | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);

  const manifest = useQuery<Manifest>({
    queryKey: queryKeys.manifest(manifestId),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/import/manifests/{manifest_id}", {
          params: { path: { manifest_id: manifestId } },
        }),
        "the manifest",
      ),
  });

  const commit = useMutation({
    mutationFn: async (body: Manifest) =>
      unwrap(await api.POST("/api/import/commit", { body: { manifest: body } }), "a commit result"),
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
      onCommitted(result);
    },
  });

  if (manifest.isPending) return <Empty>Loading the proposal…</Empty>;
  if (manifest.error) return <ErrorBox>{manifest.error.message}</ErrorBox>;

  const current = edited ?? manifest.data;
  const warnings = current.warnings;
  const mustAcknowledge = commitBlocked(warnings, acknowledged);

  const setChannels = (channels: string[]) =>
    setEdited({ ...current, channels });

  const setGroupLabel = (groupKey: string, label: Label) =>
    setEdited({
      ...current,
      samples: current.samples.map((sample: ManifestSample) =>
        sample.group_key === groupKey ? { ...sample, label } : sample,
      ),
    });

  return (
    <div className="flex flex-col gap-4">
      <Panel title="What the scan found">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-5">
          <Stat label="Samples" value={current.stats.samples} />
          <Stat label="Images" value={current.stats.images} />
          <Stat label="Masks" value={current.stats.masks} />
          <Stat label="Excluded" value={current.stats.files_excluded} />
          <Stat label="Skipped" value={current.stats.files_skipped} />
        </dl>
        <LabelRun samples={current.samples} />
        <ImportedSplit samples={current.samples} />
        <p className="mt-3 font-mono text-xs break-all text-fg-muted">
          {current.root_path}
        </p>
      </Panel>

      <WarningsPanel
        warnings={warnings}
        acknowledged={acknowledged}
        onAcknowledge={setAcknowledged}
      />

      <ChannelPanel
        manifest={current}
        proposed={manifest.data.channels}
        onChange={setChannels}
      />

      <GroupPanel manifest={current} onLabel={setGroupLabel} />

      {commit.error && <ErrorBox>{commit.error.message}</ErrorBox>}

      <div className="flex items-center gap-2">
        <Button
          variant="primary"
          disabled={mustAcknowledge || commit.isPending}
          onClick={() => commit.mutate(current)}
        >
          {commit.isPending ? "Committing…" : "Commit into the catalog"}
        </Button>
        <Button onClick={onRestart}>Start over</Button>
        {mustAcknowledge && (
          <span className="text-xs text-warn">
            Acknowledge the warnings first.
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * What the labels came out as — the number an operator is really checking on this screen.
 *
 * A dataset that imports entirely unlabelled is the single most common misconfiguration
 * of a folder- or table-driven adapter, and it is invisible in a sample count.
 */
function LabelRun({ samples }: { samples: ManifestSample[] }) {
  const counts = { normal: 0, defect: 0, unlabeled: 0 };
  for (const sample of samples) counts[sample.label] += 1;

  return (
    <div className="mt-3">
      <CountRun
        counts={[
          ["normal", counts.normal, "normal"],
          ["defect", counts.defect, "defect"],
          ["unlabeled", counts.unlabeled, "neutral"],
        ]}
      />
    </div>
  );
}

/**
 * The partition the source published, when it published one.
 *
 * Shown before commit because adopting it is the difference between a number that can be
 * compared to the paper's and one that only looks like it can. Absent entirely for the
 * datasets that ship no split, rather than rendering three zeroes.
 */
function ImportedSplit({ samples }: { samples: ManifestSample[] }) {
  const counts = { train: 0, val: 0, test: 0 };
  let placed = 0;
  for (const sample of samples) {
    if (sample.subset) {
      counts[sample.subset] += 1;
      placed += 1;
    }
  }
  if (placed === 0) return null;

  return (
    <div className="mt-3">
      <p className="mb-1 text-xs font-medium text-fg-muted">
        Published split
      </p>
      <CountRun
        counts={[
          ["train", counts.train, "neutral"],
          ["val", counts.val, "neutral"],
          ["test", counts.test, "neutral"],
        ]}
      />
      <p className="mt-1 text-xs text-fg-muted">
        Create a split with the <span className="font-mono">imported</span> strategy after
        committing to use this partition instead of drawing one.
      </p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="text-xs font-medium text-fg-muted">
        {label}
      </dt>
      <dd className="font-mono text-lg">{value}</dd>
    </div>
  );
}

function ChannelPanel({
  manifest,
  proposed,
  onChange,
}: {
  manifest: Manifest;
  proposed: string[];
  onChange: (channels: string[]) => void;
}) {
  const toggle = (channel: string) => {
    const next = manifest.channels.includes(channel)
      ? manifest.channels.filter((name) => name !== channel)
      : [...proposed.filter((name: string) => manifest.channels.includes(name) || name === channel)];
    onChange(next);
  };

  return (
    <Panel title={`Channels (${manifest.channels.length})`}>
      <p className="mb-3 text-sm text-fg-muted">
        The matcher is a convenience, not an authority. Anything you exclude here imports
        with no channel rather than a guessed one.
      </p>
      {proposed.length === 0 ? (
        <Empty>This dataset has no channels — one image per sample.</Empty>
      ) : (
        <table className="w-full text-left text-sm">
          <thead className="text-xs font-medium text-fg-muted">
            <tr>
              <th className="pb-2">Keep</th>
              <th className="pb-2">Source directory</th>
              <th className="pb-2">Read as</th>
              <th className="pb-2">Matched by</th>
            </tr>
          </thead>
          <tbody>
            {manifest.channel_mapping.map((row: ChannelMapping) => (
              <tr key={row.source} className="border-t border-line">
                <td className="py-1.5">
                  <Checkbox
                    aria-label={`Keep ${row.channel}`}
                    checked={manifest.channels.includes(row.channel)}
                    onCheckedChange={() => toggle(row.channel)}
                  />
                </td>
                <td className="py-1.5 font-mono text-xs">{row.source}</td>
                <td className="py-1.5 font-mono text-xs">{row.channel}</td>
                <td className="py-1.5">
                  <Badge tone={row.matched_by === "exact" ? "neutral" : "info"}>
                    {row.matched_by}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function GroupPanel({
  manifest,
  onLabel,
}: {
  manifest: Manifest;
  onLabel: (groupKey: string, label: Label) => void;
}) {
  const groups = new Map<string, { total: number; labels: Set<Label>; channels: Set<number> }>();
  for (const sample of manifest.samples) {
    const entry = groups.get(sample.group_key) ?? {
      total: 0,
      labels: new Set<Label>(),
      channels: new Set<number>(),
    };
    entry.total += 1;
    entry.labels.add(sample.label);
    entry.channels.add(sample.images.length);
    groups.set(sample.group_key, entry);
  }

  return (
    <Panel title={`Capture groups (${groups.size})`}>
      <p className="mb-3 text-sm text-fg-muted">
        Labels came from directory names. Correct a whole group here; individual samples
        can be relabelled in the browser afterwards.
      </p>
      <table className="w-full text-left text-sm">
        <thead className="text-xs font-medium text-fg-muted">
          <tr>
            <th className="pb-2">Group</th>
            <th className="pb-2">Samples</th>
            <th className="pb-2">Images each</th>
            <th className="pb-2">Label</th>
          </tr>
        </thead>
        <tbody>
          {[...groups.entries()].map(([groupKey, entry]) => (
            <tr key={groupKey} className="border-t border-line">
              <td className="py-1.5 font-mono text-xs break-all">{groupKey}</td>
              <td className="py-1.5 font-mono text-xs">{entry.total}</td>
              <td className="py-1.5 font-mono text-xs">
                {[...entry.channels].sort((a, b) => a - b).join(", ")}
              </td>
              <td className="py-1.5">
                <Select
                  className="w-36"
                  aria-label={`Label for ${groupKey}`}
                  value={entry.labels.size === 1 ? ([...entry.labels][0] ?? "") : ""}
                  placeholder="mixed"
                  options={[
                    { value: "normal", label: "normal" },
                    { value: "defect", label: "defect" },
                    { value: "unlabeled", label: "unlabeled" },
                  ]}
                  onValueChange={(value) => onLabel(groupKey, value as Label)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function CommittedStep({
  result,
  onRestart,
}: {
  result: CommitResponse;
  onRestart: () => void;
}) {
  const navigate = useNavigate();

  return (
    <Panel title={result.dataset_created ? "Dataset created" : "Dataset updated"}>
      <div className="flex flex-col gap-4">
        <CountRun
          counts={[
            ["samples added", result.samples_created, "normal"],
            ["samples updated", result.samples_updated, "neutral"],
            ["images added", result.images_created, "normal"],
            ["images updated", result.images_updated, "neutral"],
            ...(result.masks > 0
              ? ([["masks attached", result.masks, "normal"]] as [string, number, "normal"][])
              : []),
          ]}
        />

        {result.missing_paths.length > 0 && (
          <div>
            <p className="text-sm">
              {result.missing_paths.length} recorded file(s) are no longer in the source
              tree. They were <strong>reported, not deleted</strong> — an unmounted disk
              should not become data loss.
            </p>
            <ul className="mt-1 max-h-32 overflow-y-auto rounded bg-raised p-2 font-mono text-xs ">
              {result.missing_paths.map((path) => (
                <li key={path} className="truncate">
                  {path}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex gap-2">
          <Button variant="primary" onClick={() => void navigate(`/datasets/${result.dataset_id}`)}>
            Browse the dataset
          </Button>
          <Button onClick={onRestart}>Import another</Button>
        </div>
      </div>
    </Panel>
  );
}
