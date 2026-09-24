"""Prototype matching over frozen patch features, for few-shot segmentation (ADR-0040).

The arithmetic `fss_dino` reproduces from FSSDINO, and that `proto_seg` builds on, kept in
numpy so it is tested without torch:

- a reference's mask is brought to the patch grid by bilinear interpolation, and a patch
  belongs to the class when at least half of it is covered;
- a class's patch features are summarised by `k` prototypes from spherical (cosine)
  k-means, seeded;
- a query patch's similarity to each prototype is a cosine map;
- a class's Gram matrix `G = F^T F / N` gives each query patch an energy `q^T G q`, the
  channel-correlation signal FSSDINO adds, min-max normalised per image to `[0, 1]`;
- a class's maps are upsampled to pixels and combined as `mean(maps) * max(maps)`, and a
  pixel goes to the class with the higher combined score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from anomaly_lab.models.refine import upsample


def patch_coverage(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """`(rows * cols,)` share of each patch the mask covers, by bilinear interpolation."""
    rows, cols = grid
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = image.resize((cols, rows), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).reshape(-1)


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return np.asarray(rows / np.maximum(norms, 1e-12), dtype=np.float32)


def spherical_kmeans(
    features: np.ndarray, k: int, *, iterations: int, rng: np.random.Generator
) -> np.ndarray:
    """`(k', D)` unit centroids of unit `(N, D)` features by cosine k-means; `k' = min(k, N)`.

    Seeded through `rng` alone, starting from `k'` distinct samples. A cluster that empties
    keeps its previous centroid rather than being reseeded, so the result depends on the
    seed and nothing else.
    """
    count = len(features)
    if count == 0:
        raise ValueError("k-means needs at least one feature")
    k = min(k, count)
    centroids = features[rng.choice(count, size=k, replace=False)].copy()
    for _ in range(iterations):
        assignment = np.argmax(features @ centroids.T, axis=1)
        updated = centroids.copy()
        for cluster in range(k):
            members = features[assignment == cluster]
            if len(members):
                updated[cluster] = members.sum(axis=0)
        updated = _unit(updated)
        if np.allclose(updated, centroids):
            break
        centroids = updated
    return _unit(centroids)


def gram_matrix(features: np.ndarray) -> np.ndarray:
    """`(D, D)` channel correlation of a class's `(N, D)` features."""
    return np.asarray(features.T @ features / max(len(features), 1), dtype=np.float32)


def gram_energy(query: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """`(P,)` energy `q^T G q` per query patch, min-max normalised to `[0, 1]` over the image."""
    energy = np.einsum("pd,de,pe->p", query, gram, query)
    low, high = float(energy.min()), float(energy.max())
    if high <= low:
        return np.zeros_like(energy, dtype=np.float32)
    return np.asarray((energy - low) / (high - low), dtype=np.float32)


def class_maps(
    query: np.ndarray, prototypes: np.ndarray, gram: np.ndarray | None
) -> list[np.ndarray]:
    """One class's `(P,)` maps for one query: a cosine map per prototype, then the Gram energy."""
    maps = list((query @ prototypes.T).T)
    if gram is not None:
        maps.append(gram_energy(query, gram))
    return [np.asarray(entry, dtype=np.float32) for entry in maps]


def combined_score(
    maps: Sequence[np.ndarray], grid: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    """`mean(maps) * max(maps)` after each map is upsampled to `size` (`(w, h)`)."""
    rows, cols = grid
    upsampled = np.stack([upsample(entry.reshape(rows, cols), size) for entry in maps])
    return np.asarray(upsampled.mean(axis=0) * upsampled.max(axis=0), dtype=np.float32)


def foreground_probability(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """The foreground's share of the two combined scores, each floored at zero.

    `>= 0.5` exactly where the foreground's score is the higher of two positive scores, so
    the evaluator's rule and the argmax agree; where neither score is positive, 0.5.
    """
    fg = np.maximum(foreground, 0.0)
    bg = np.maximum(background, 0.0)
    total = fg + bg
    return np.asarray(np.where(total > 0, fg / np.maximum(total, 1e-12), 0.5), dtype=np.float32)


def mean_prototype(features: np.ndarray) -> np.ndarray:
    """`(1, D)`: the class's unit mean direction — the global prototype."""
    return _unit(features.mean(axis=0, keepdims=True))


def lse_probability(
    query: np.ndarray, foreground: np.ndarray, background: np.ndarray, temperature: float
) -> np.ndarray:
    """`(P,)` foreground probability as the foreground's share of a softmax over all prototypes.

    Each side scores a patch by the log-sum-exp of its cosine similarities over the
    temperature — a soft maximum over that side's prototypes — and the probability is the
    two-way softmax of those scores. Computed stably, relative to each patch's best match.

    The sum makes a side's prototype count part of its score, and as the temperature rises
    the probability drifts toward the ratio of the counts; at a working temperature the best
    match dominates. `proto_seg` gives both sides the same number of prototypes.
    """
    logits = np.concatenate([query @ foreground.T, query @ background.T], axis=1) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    share = weights[:, : len(foreground)].sum(axis=1) / weights.sum(axis=1)
    return np.asarray(share, dtype=np.float32)


def fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> tuple[np.ndarray, float]:
    """A class-balanced, L2-regularised logistic regression by full-batch gradient descent.

    Deterministic — zero initialisation, the whole batch every step — so it needs no seed.
    Balanced because a reference is mostly background, and an unweighted probe learns that.
    """
    x = features.astype(np.float64)
    y = labels.astype(np.float64)
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0, 0.5 / positives, 0.5 / negatives)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(iterations):
        predicted = 1.0 / (1.0 + np.exp(-np.clip(x @ weights + bias, -60, 60)))
        error = (predicted - y) * sample_weight
        weights -= learning_rate * (x.T @ error + l2 * weights)
        bias -= learning_rate * float(error.sum())
    return weights.astype(np.float32), float(bias)


def probe_probability(query: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(query @ weights + bias, -60, 60)
    return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)


def presence_score(patch_probability: np.ndarray, region_patches: int) -> float:
    """The mean of the `region_patches` most confident patches.

    A presence claim has to be carried by a region of at least that size, not by one patch.
    """
    count = max(1, min(region_patches, patch_probability.size))
    top = np.partition(patch_probability.ravel(), -count)[-count:]
    return float(top.mean())
