# Annotation

Annotations are versioned source-frame truth. The editor is optimized around the image: controls and local
navigation live around a full-height canvas rather than pushing it below a long form.

## Editor controls

- Draw polygon vertices around a defect and close the contour.
- Drag a box (`R`) from corner to corner around an object; under Select, drag a corner to resize it.
- Pick the class for new regions with `2`–`9`, in the order the class picker lists them.
- Paint or erase a region for irregular defects.
- Select, move, insert, or remove contour vertices before committing.
- Undo and redo edits within the draft.
- Move the view by dragging with either mouse button; there is no separate pan tool.
- Choose **1:1** for one image pixel per display pixel or **Fit** for the largest contained view.
- Double-click the canvas to fit; double-click again to restore the previous scale and center.
- Step through the current filtered sample set without returning to the gallery.

The editor distinguishes viewport operations from geometry operations. Zooming and panning never alter the
annotation. Pointer coordinates are transformed back through the viewport into source-image coordinates
before editing.

## Automatic contour derivation

A roughly painted region can seed contour derivation. The backend derives candidate connected boundaries
from the marked region and returns editable geometry. This is an acceleration tool, not an automatic label:
the proposal remains a draft until reviewed and committed. Small islands and holes should be visible and
configurable rather than silently discarded.

## Suggestions from Explore

The sample viewer's Explore section can send a mask to the editor: a thresholded similarity, one k-means
cluster of a frozen encoder's patches, a MobileSAM candidate, or one SAM 3 instance found by a phrase. It opens under **Contour assist** as a
suggestion, like MobileSAM's own; **Accept mask** or **Editable contour** adds it in the class selected in
the editor, and **Discard suggestion** drops it. Nothing reaches the draft until it is accepted.

## Revision model

An edit begins from one immutable annotation revision or an empty draft. Saving creates a new revision with
provenance; it does not mutate the previous one. The active revision may change without changing historical
experiments. Masks imported from a source dataset retain their source provenance and can be opened as the
base for a corrected revision.

Annotations belong to individual images, while the defect label belongs to the sample. Multi-channel samples
may therefore be defective with geometry on one or several views. The UI must not invent identical masks for
other channels.

## Quality checks

Before using masks for pixel metrics:

1. inspect contours at 1:1 on several small and large defects;
2. check source-image borders and non-square images;
3. verify holes and disconnected regions;
4. compare mask area and bounding box against the visible defect;
5. reopen the committed revision to prove serialization round-trips.

The detailed storage and conflict rules are in the [annotation handbook](https://github.com/VitalyVorobyev/visual-anomaly-lab/blob/main/docs/architecture/annotations.md).
