"""The whole of SubspaceAD that is not a forward pass, in numpy.

Keeping this torch-free is a decision about what the reproduction gate is worth. The
method is a covariance, an eigendecomposition and a subtraction; if those live inside the
same module as the encoder, they can only be exercised on a machine with the optional `dl`
extra installed, which is neither CI's torch-free job nor most checkouts. Here they are
ordinary array code with ordinary tests, and the only thing the encoder contributes is the
array it hands over.

**Two identities do the work, and both are exact rather than approximations.**

The paper scores a patch by the squared length of the part of it that the subspace cannot
reconstruct, `‖(x-µ) - CCᵀ(x-µ)‖²`. Because `C`'s columns are orthonormal, that equals

    S_r(x) = ‖x-µ‖² - Σ_{i≤r} a_i²,     a_i = v_iᵀ(x-µ)

so the coefficients `a_i` are computed **once**, up to the largest rank any threshold
selects, and a running `cumsum` reads off the score at every smaller rank. Sweeping τ
therefore costs one projection rather than one per value, and the numbers it produces are
bit-identical to the ones separate runs would produce. `ResidualBasis` is that object.

The image score is the mean of the top rho% of patch scores, which is a prefix mean of the
sorted map — so sweeping rho costs one sort. `tail_value_at_risk` takes every rho at once for
the same reason.

Only the encoder forward and the input resolution genuinely cost anything. That is the
finding the campaign's budget is built on, and it lives here because it is a property of
this arithmetic, not of the schedule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CovarianceAccumulator:
    """Streams patch features into the three sums a covariance is made of.

    Nothing is retained. A 4-shot fit at 672 px is 285k patches of 1024 floats — over a
    gigabyte held at once, and the campaign fits dozens of these per category — so the
    features are folded in a batch at a time and dropped.

    **The sums are float64 even though the features are float32, and that is not
    superstition.** The covariance is recovered as `S - s sᵀ/n`, a difference between two
    quantities of similar size, and every digit lost to cancellation lands in the small
    eigenvalues — which is exactly where τ=0.99 draws its line. The features themselves
    stay float32: they arrived that way from the encoder, and promoting them would buy
    precision the measurement never had.
    """

    dimension: int
    count: int = 0
    total: np.ndarray = field(init=False)
    scatter: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.total = np.zeros(self.dimension, dtype=np.float64)
        self.scatter = np.zeros((self.dimension, self.dimension), dtype=np.float64)

    def add(self, features: np.ndarray) -> None:
        """Fold in one `(n, D)` block of patch features."""
        if features.ndim != 2 or features.shape[1] != self.dimension:
            msg = f"expected (n, {self.dimension}) features; got shape {features.shape}"
            raise ValueError(msg)
        block = np.asarray(features, dtype=np.float64)
        self.count += block.shape[0]
        self.total += block.sum(axis=0)
        self.scatter += block.T @ block

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, int]:
        """`(mean, covariance, count)` as they stand, without consuming the accumulator.

        Called repeatedly on purpose. The k-shot arms are nested — the 4-shot fit set is
        the 1-shot one plus three more images — so one pass over the drawn normals yields
        every k by snapshotting as it crosses each one, and the encoder never runs twice
        over the same picture.
        """
        if self.count < 2:
            msg = f"a covariance needs at least two samples; {self.count} were added"
            raise ValueError(msg)
        mean = self.total / self.count
        covariance = (self.scatter - self.count * np.outer(mean, mean)) / (self.count - 1)
        # Symmetry is algebraically guaranteed and numerically not: the two triangles of
        # `scatter` accumulate in a different order, and `eigh` reads only one of them.
        # Averaging costs nothing and removes the asymmetry from the eigenvalues.
        return mean, (covariance + covariance.T) / 2.0, self.count


@dataclass(frozen=True)
class SubspaceFit:
    """A fitted normal subspace: the mean, an orthonormal basis, and the spectrum.

    `eigenvalues` holds **all** D of them, not just the retained ones, because τ is a
    fraction of the total variance — truncating the spectrum first would make every
    threshold compare against a denominator that had already been thresholded.
    """

    mean: np.ndarray
    components: np.ndarray
    """`(R, D)`, one unit eigenvector per row, descending by eigenvalue. Stored float32:
    the projection it feeds is a float32 matmul, and a float64 basis would only pay for
    precision the features do not carry."""
    eigenvalues: np.ndarray
    sample_count: int

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    @property
    def available_rank(self) -> int:
        return int(self.components.shape[0])

    def rank_for(self, tau: float) -> int:
        """The smallest r whose eigenvalues carry at least a τ fraction of the variance.

        τ=1.0 asks for the whole spectrum, and the degenerate arm that produces is the
        point of asking: with every direction retained the residual is the difference
        between a number and itself, so the score carries no signal at all. The paper
        measures that collapse (Table 4) rather than guarding against it, and so does this.
        """
        if not 0.0 < tau <= 1.0:
            msg = f"tau must be in (0, 1]; got {tau}"
            raise ValueError(msg)
        spectrum = np.clip(self.eigenvalues, 0.0, None)
        total = float(spectrum.sum())
        if total <= 0.0:
            return 1
        cumulative = np.cumsum(spectrum) / total
        # `searchsorted` on the cumulative fraction is the same answer as a scan, and its
        # boundary behaviour is the one that is wanted: `side="left"` returns the first
        # index at or above the threshold rather than the first strictly above it.
        #
        # The clamp is not defensive padding. `cumulative[-1]` is 1.0 only up to rounding,
        # and on a real covariance -- whose tail eigenvalues are small negatives that the
        # clip above flattens to zero -- it lands a few ulps low often enough. tau=1.0 then
        # asks for one component more than the space has, and the fit refuses to score at
        # all. A threshold can never need more directions than exist.
        rank = int(np.searchsorted(cumulative, tau, side="left")) + 1
        return int(min(rank, spectrum.size))


def fit_subspace(
    accumulator: CovarianceAccumulator,
    *,
    max_rank: int | None = None,
) -> SubspaceFit:
    """Eigendecompose one accumulated covariance into a normal subspace.

    `max_rank` bounds only what is *kept*, never what is *measured*: the full spectrum is
    returned regardless, so a rank chosen for τ is still a fraction of the true total
    variance. A fit asked to score at a rank it did not keep says so rather than quietly
    scoring at a lower one.
    """
    mean, covariance, count = accumulator.snapshot()
    # `eigh` exploits symmetry, returns a genuinely orthonormal basis, and — unlike `eig` —
    # cannot return complex values for a matrix that rounding has nudged off symmetric.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order].T
    if max_rank is not None:
        components = components[:max_rank]
    return SubspaceFit(
        mean=mean.astype(np.float32),
        components=np.ascontiguousarray(components, dtype=np.float32),
        eigenvalues=eigenvalues,
        sample_count=count,
    )


@dataclass(frozen=True)
class ResidualBasis:
    """Every rank's patch scores for one image, held as a total and a running sum.

    This is the object that makes τ free. `total` is `‖x-µ‖²` per patch and `cumulative`
    is the running `Σ a_i²`, so the score at any rank is one subtraction against a column
    that is already there.
    """

    total: np.ndarray
    cumulative: np.ndarray

    def at_rank(self, rank: int) -> np.ndarray:
        """Patch scores with `rank` components retained.

        **Not clipped at zero**, though a squared residual cannot be negative. At full rank
        the subtraction is a number minus itself and what survives is float noise of either
        sign; clipping would turn that into a field of exact ties, which reads on a results
        page as a metric that could not be computed rather than as a configuration that
        destroyed its own signal.
        """
        if rank < 0 or rank > self.cumulative.shape[1]:
            msg = (
                f"rank {rank} is outside the {self.cumulative.shape[1]} components this "
                "basis was projected onto"
            )
            raise ValueError(msg)
        if rank == 0:
            return self.total.copy()
        scores: np.ndarray = self.total - self.cumulative[:, rank - 1]
        return scores


def residual_basis(fit: SubspaceFit, features: np.ndarray, *, rank: int) -> ResidualBasis:
    """Project `(n, D)` patch features onto the leading `rank` components, once.

    The cost of the whole τ axis is this one matmul, sized by the largest rank any
    threshold in the sweep selects.
    """
    if rank > fit.available_rank:
        msg = (
            f"this fit kept {fit.available_rank} components and was asked to score at rank "
            f"{rank}; refit with a larger max_rank"
        )
        raise ValueError(msg)
    centred = np.asarray(features, dtype=np.float32) - fit.mean
    total = np.einsum("nd,nd->n", centred, centred, dtype=np.float32).astype(np.float64)
    coefficients = centred @ fit.components[:rank].T
    cumulative = np.cumsum(np.square(coefficients, dtype=np.float32), axis=1, dtype=np.float64)
    return ResidualBasis(total=total, cumulative=cumulative)


def tail_value_at_risk(scores: np.ndarray, fractions: Sequence[float]) -> dict[float, float]:
    """The mean of the top rho of patch scores, for every rho at once.

    One descending sort, then a prefix mean — so the rho axis costs a sort rather than a pass
    per value. At least one patch is always taken, which is what keeps a small rho on a small
    grid from averaging an empty set.
    """
    flat = np.sort(np.asarray(scores, dtype=np.float64).ravel())[::-1]
    if flat.size == 0:
        msg = "an image-level score needs at least one patch score"
        raise ValueError(msg)
    running = np.cumsum(flat) / np.arange(1, flat.size + 1)
    result: dict[float, float] = {}
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            msg = f"the TVaR fraction must be in (0, 1]; got {fraction}"
            raise ValueError(msg)
        taken = max(1, round(fraction * flat.size))
        result[fraction] = float(running[taken - 1])
    return result
