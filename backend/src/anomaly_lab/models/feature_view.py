"""Making a high-dimensional feature map look at, for the diagnostics contract.

A deep method's internal features are the most informative thing it has and the least
viewable: 384 teacher channels or 1536 concatenated backbone channels cannot be put on a
screen directly, and picking three by index shows whichever three the author happened to
type.

`pca_to_rgb` is the answer both methods reached for independently, so it lives here rather
than twice. **numpy only** — no torch at module scope and none inside — so it is tested in
the CI job that installs without the `dl` extra, alongside `introspect.build_tree`, which
is split from `introspect.collect` for exactly the same reason.

What it produces is a **false-colour** image and is labelled as one everywhere it is shown.
The three leading components are the directions the features vary in most across the frame,
which is what a reader wants from such a picture; they are not channels, and the colours
carry no fixed meaning between two runs.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-9


def pca_to_rgb(features: np.ndarray) -> np.ndarray:
    """Reduce a `(C, H, W)` feature map to `(H, W, 3)` in `[0, 1]` by PCA.

    The result is the `image` diagnostic kind: rank 3, three trailing channels, already
    normalized to the unit interval, so it is rendered as-is with no colormap and no
    display range (`DiagnosticWriter._widen_range` deliberately records none for it).

    SVD rather than an explicit covariance eigendecomposition: the covariance matrix is
    small either way — 384x384, 1536x1536 — but forming it squares the condition number
    for no benefit here.
    """
    if features.ndim != 3:
        msg = f"pca_to_rgb needs a (C, H, W) feature map, got shape {features.shape}"
        raise ValueError(msg)

    channels, height, width = features.shape
    if channels < 3:
        msg = f"pca_to_rgb needs at least 3 channels to produce 3 components, got {channels}"
        raise ValueError(msg)

    flat = features.reshape(channels, height * width).T.astype(np.float64)
    centred = flat - flat.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projected = (centred @ components[:3].T).reshape(height, width, 3)

    low = projected.min(axis=(0, 1), keepdims=True)
    high = projected.max(axis=(0, 1), keepdims=True)
    # A constant component — a feature map with no spatial variation at all — would divide
    # by zero and produce NaNs that reach the renderer as a black tile with no explanation.
    return ((projected - low) / np.maximum(high - low, _EPSILON)).astype(np.float32)
