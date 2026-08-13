/**
 * The primitive set (ADR-0021).
 *
 * This replaces the single 143-line `components/ui.tsx`, which held a status colour, a
 * panel border and a button shape and described itself as "the cheap alternative" to a
 * component library. It stayed cheap for two milestones and then stopped: by the end of M4
 * the app had eight native `<select>`s the OS styled for us, three unstyled range inputs,
 * five hand-rolled tables, a destructive action with no confirmation, and no visible focus
 * ring anywhere.
 *
 * Import from `components/ui`; the individual modules are an implementation detail.
 */

export { cn, focusRing, focusRingInset } from "./cn";

export { Badge, CountRun, StatusDot, type Tone } from "./Badge";
export { Button } from "./Button";
export { ConfirmDialog, Dialog, DialogClose } from "./Dialog";
export { Disclosure } from "./Disclosure";
export { Callout, Empty, ErrorBox, ProgressBar, Skeleton, SkeletonRows } from "./Feedback";
export { Field } from "./Field";
export { Input, NumberInput, Textarea, controlClasses, inputClasses } from "./Input";
export {
  PageHeader,
  Panel,
  ReadoutStrip,
  Section,
  type ReadoutItem,
} from "./Panel";
export { SegmentedControl } from "./SegmentedControl";
export { Select, type SelectOption } from "./Select";
export { Slider } from "./Slider";
export { Table, type Column } from "./Table";
export { Checkbox, Switch } from "./Toggle";
export { ToggleChip } from "./ToggleChip";
export { InfoHint, Tooltip, TooltipProvider } from "./Tooltip";
