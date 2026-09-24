"""Removing where a patch is from what a frozen ViT says about it (INSID3's positional debiasing).

A DINO patch feature carries its position as well as its content: two patches of the same
material at the same grid cell of two images are closer than the same material at two
cells. Matching a reference's region against a query then rewards being in the same place,
which is exactly wrong for an object that may appear anywhere.

INSID3 estimates the positional part from an image with no content — Gaussian noise
through the encoder — and projects it out. The features of a noise image are a `(P, D)`
matrix whose rows differ only by position; its top `s` right singular vectors span the
directions position lives in, and every feature used for matching is projected onto their
orthogonal complement. INSID3 fixes `s = 500` for a 1024-wide ViT-L.

Only the arithmetic is here, in numpy, so it is tested without torch; the encoder pass
is `dino_backbone.noise_patch_features`.
"""

from __future__ import annotations

import numpy as np

INSID3_RANK = 500


def positional_basis(noise_features: np.ndarray, rank: int) -> np.ndarray:
    """`(D, r)` orthonormal columns: the top right singular vectors of `(P, D)` features.

    `r` is `rank` clipped to what the matrix can hold. A small grid has fewer patches than
    INSID3's rank, and a basis as wide as every direction the noise spans would remove
    everything — so the rank is also held below half of `min(P, D)`, and the caller logs what
    it resolved to.
    """
    patches, width = noise_features.shape
    resolved = resolved_rank(rank, patches=patches, width=width)
    if resolved == 0:
        return np.zeros((width, 0), dtype=np.float32)
    _, _, right = np.linalg.svd(noise_features.astype(np.float64), full_matrices=False)
    return np.ascontiguousarray(right[:resolved].T, dtype=np.float32)


def resolved_rank(rank: int, *, patches: int, width: int) -> int:
    return max(0, min(rank, min(patches, width) // 2))


def debias(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project `(..., D)` features onto the complement of the basis, then back to unit length.

    Renormalised because every matcher downstream reads cosine similarity from a dot product.
    """
    if basis.shape[1] == 0:
        return features
    projected = features - (features @ basis) @ basis.T
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    return np.asarray(projected / np.maximum(norms, 1e-12), dtype=features.dtype)
