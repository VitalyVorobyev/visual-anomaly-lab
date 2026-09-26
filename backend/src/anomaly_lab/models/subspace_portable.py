"""`subspace_ad`'s exported graph: prepared pixels in, the stored map and the image score out.

Imported only by the plugin's export path, so torch at module scope costs nothing to anyone
who opens the method picker (ADR-0007).

Every step after the encoder is a constant of the fit — the mean, the retained components,
the token grid, the tail count and the two map operators — so the graph is static: no
data-dependent control flow, and no channel branch, because one graph carries one channel's
subspace. The arithmetic mirrors `subspace.residual_basis`, `tail_value_at_risk` and
`score_map.pixel_map` operation for operation, including where they widen to float64:
the residual is a difference of two sums of similar size, and the score is read from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


class PortableSubspace(nn.Module):
    """Encoder, centre, project, residual, tail mean and the separable map operators."""

    def __init__(
        self,
        encoder: Any,
        *,
        indices: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        centre: np.ndarray,
        components: np.ndarray,
        grid: tuple[int, int],
        tail_count: int,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.indices = [int(index) for index in indices]
        self.grid = grid
        self.tail_count = tail_count
        self.register_buffer("pixel_mean", torch.tensor(list(mean)).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(list(std)).view(1, -1, 1, 1))
        self.register_buffer("centre", torch.from_numpy(np.asarray(centre, dtype=np.float32)))
        # `(D, r)` float64: the float32 basis, widened exactly as `residual_basis` widens it.
        self.register_buffer(
            "components_t",
            torch.from_numpy(
                np.ascontiguousarray(np.asarray(components, dtype=np.float32).T, np.float64)
            ),
        )
        self.register_buffer("rows", torch.from_numpy(np.ascontiguousarray(rows, np.float32)))
        self.register_buffer(
            "columns_t", torch.from_numpy(np.ascontiguousarray(columns.T, np.float32))
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.get_buffer("pixel_mean")
        std = self.get_buffer("pixel_std")
        if image.shape[1] != mean.shape[1]:
            # `expand_planes`: a grey frame is its one plane replicated, as the fit saw it.
            image = image.expand(-1, int(mean.shape[1]), -1, -1)
        maps = self.encoder.forward_intermediates(
            (image - mean) / std,
            indices=self.indices,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        pooled = torch.stack(maps, dim=1).flatten(3).mean(dim=1)  # (1, D, P)
        features = pooled[0].transpose(0, 1)  # (P, D)

        centred = (features - self.get_buffer("centre")).to(torch.float64)
        total = (centred * centred).sum(dim=1)
        coefficients = centred @ self.get_buffer("components_t")
        explained = (coefficients * coefficients).sum(dim=1)
        residual = total - explained  # (P,), float64

        top, _ = torch.topk(residual, self.tail_count)
        score = top.mean().to(torch.float32).reshape(1)

        tokens = residual.to(torch.float32).reshape(self.grid)
        anomaly_map = self.get_buffer("rows") @ tokens @ self.get_buffer("columns_t")
        return anomaly_map.reshape(1, 1, *anomaly_map.shape), score
