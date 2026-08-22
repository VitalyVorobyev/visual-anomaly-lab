/**
 * How the annotation editor's channel panes are chosen, and how long a view survives.
 *
 * Both decisions used to be implicit in component state that a remount silently reset:
 * changing the second channel of a side-by-side comparison dropped the whole workspace back
 * to one pane, because the editor was keyed by image id and every `useState` in it went with
 * the key. Pulling the rules out here makes them assertable without a Konva harness, and
 * makes it obvious which state belongs to the reader rather than to the document.
 */

/** How the whole channel column is shown at once. */
export type PaneMode = "single" | "compare" | "overlay";

/**
 * Which sibling channel the second pane shows.
 *
 * Positions rather than image ids, because the preference has to outlive the image it was
 * expressed on: a reader who put `dark` beside `bright` still wants a second pane after
 * moving to the next part, where `dark` is a different image entirely. Two images of one
 * sample can also carry the same channel name, or none at all, so the name is not an
 * identity either (`ChannelTabs` makes the same choice).
 *
 * The wrap is what stops a two-channel sample from merely swapping its panes: with `bright`
 * active and `dark` preferred, switching to `dark` would otherwise resolve the reference back
 * to `dark` and show the same photograph twice.
 */
export function resolveReference(
  count: number,
  activeIndex: number,
  preferred: number | null,
): number | null {
  if (count < 2 || activeIndex < 0 || activeIndex >= count) return null;
  if (preferred !== null && preferred !== activeIndex && preferred >= 0 && preferred < count) {
    return preferred;
  }
  return (activeIndex + 1) % count;
}

/**
 * The identity of the frame a pan and zoom belong to.
 *
 * A view survives moving between the channels of one part — they are exposures of the same
 * object, and re-finding a defect at 8× after every channel switch is the opposite of the
 * workflow. It does not survive moving to another part, or to a channel that does not share
 * the source frame, because there the stored pan points at nothing in particular.
 */
export function paneFrame(sampleId: number, width: number, height: number): string {
  return `${sampleId}:${width}x${height}`;
}
