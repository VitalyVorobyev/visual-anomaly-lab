# ADR-0010: Classical circular-part baseline algorithm

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-06.

> **Read [`architecture/methods.md`](../architecture/methods.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

## Context

The brief asks for "a simple non-neural image-comparison baseline" and suggests registration, a
robust normal reference, pixel differences, and an optional polar transform for circular objects.
This method is the **vertical slice**: it runs on CPU in seconds over 189 samples, needs no
downloads, and therefore lets the entire pipeline — import, split, train, infer, score, overlay,
evaluate — be built and validated before any deep-learning dependency lands (ADR-0008).

The showcase dataset shows a centred circular part under **arbitrary rotation**, and each part
carries asymmetric surface features: strong, legitimate structure that any naive difference against a
fixed reference would flag as anomalous. Rotation must therefore be estimated, not assumed.

Unlike the deep-learning methods, this one is permitted to be **dataset-specific**
(`dataset_specific=True`, ADR-0007). It encodes what this particular circular part *is*.

## Decision

A polar-registration + robust-statistics pipeline. **Fit:**

1. **Circle detection.** A coarse Hough transform on a downsampled, blurred image seeds a centre and
   radius; *N* radial rays then locate subpixel rim-edge points, and a RANSAC / Taubin least-squares
   circle fit rejects outliers from the asymmetric surface features and from defects. A
   **dataset-median centre/radius prior** serves three roles: initialization, sanity check, and a
   **flagged fallback** when the fit fails its residual check.
2. **Geometry once per sample.** The circle is detected on the **cleanest channel** (expected
   bright-field; to be verified empirically) and **shared across that sample's other channels** —
   the captures are near-simultaneous from the same camera, so the part has not moved.
3. **Polar unwrap.** `cv2.warpPolar` about the fitted centre, roughly 1024 angular by *R* radial
   samples, grayscale float. Rotation becomes translation along one axis.
4. **Orientation.** Estimated by **FFT-based 1D circular cross-correlation** of multi-band angular
   signatures (radial bands covering the asymmetric-feature region), with a parabola fit for
   subpixel peak location. This was chosen over explicitly detecting the individual features:
   simpler and more robust to damage. An explicit feature detector remains a backlog fallback. The
   reference is bootstrapped by anchoring on one normal sample, aligning all normals to it,
   averaging, and re-aligning to that average for one or two iterations.
5. **Photometric normalization.** Median intensity is matched on a stable annulus, mitigating the
   lighting drift observed between `set1` and `set2`.
6. **Robust reference.** Per channel, per polar pixel: `μ(θ,r)` = median and
   `σ(θ,r) = max(1.4826·MAD, σ_min)` over the aligned normals. Naturally high-variance regions —
   feature edges, specular highlights — acquire large `σ` and **self-desensitize**. No hand-drawn
   masks.

**Predict:** circle fit (prior fallback) → polar → orientation-align → photometric normalize →
`z = |I − μ| / σ` → light Gaussian smoothing → **inverse polar warp back to Cartesian**, so overlays
register with the original image (ADR-0007). The image score is a **high percentile (p99.5) or
top-k-pixel mean** — robust where a plain maximum would track a single hot pixel. A thin rim margin
(specular ring) and everything at `r > R` are excluded.

**Known failure modes**, documented rather than hidden: dome-channel speculars shift with part tilt
(locally desensitized by MAD; accepted); damaged or deformed asymmetric features corrupt orientation
(mitigated by multi-band correlation and percentile scoring); severe rim damage breaks the circle fit
(residual check → prior fallback → flagged); set-to-set lighting drift (photometric normalization,
with per-set references as an escape hatch); polar sampling is anisotropic near the centre (accepted
for research use).

**Capabilities:** `requires_training=True`, `produces_anomaly_map=True`, `channel_aware=True`,
`dataset_specific=True`, CPU.

## Consequences

A fully interpretable detector — every score traces to a `(θ, r)` cell and a z-value — that trains
in seconds and gives the workbench something real to evaluate from day one. Robust statistics remove
the need for manual masking, and per-channel references mean each illumination is judged against its
own normal appearance.

Negative consequences, accepted honestly:

- **It does not generalize.** A circular outline and this part's asymmetric surface features are
  assumptions. On a non-circular part this method is worthless, and `dataset_specific=True` is an
  admission, not a mitigation.
- **A registration failure is a false positive factory.** Sub-degree orientation error smears the
  whole reference; the pipeline has several such single points of failure (centre, radius, angle,
  photometric scale) chained in series.
- **Sensitivity is spatially uneven by construction.** Self-desensitization means genuine defects on
  feature edges or in specular regions are exactly the ones most likely to be missed — a systematic
  blind spot aligned with real defect locations.
- **The prior fallback hides degradation.** A flagged fallback still produces a score, and unless the
  flag is surfaced prominently in the UI it will be read as a normal result.
- **Small-sample reference statistics.** With ~60 normals per channel, MAD estimates are noisy;
  `σ_min` is a hand-tuned floor doing real work.
- **Several parameters are empirical** (band placement, `σ_min`, smoothing width, percentile, rim
  margin) and were chosen for this dataset. Retuning is manual and unguided.
