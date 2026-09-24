"""Experiments: create, train, score, and read the results.

The route surface deliberately mentions no method by name. It serves the registry's
descriptions — including each method's JSON Schema — and the frontend builds its form
from them, which is what makes "add a method without touching the rest of the app" true
in practice rather than aspirational (ADR-0007).

One prefix, four cohesive modules, included here in the order their routes have always
been declared (the order is visible in the OpenAPI document and so in the generated
client):

- `crud` — the method catalogue, create, search, read and delete;
- `runs` — queue training, scoring and export, and re-read stored scores;
- `results` — rankings, thresholds, curves, previews and the artifact listing;
- `diagnostics` — the recorded index, on-demand diagnosis, pruning and raw values.

`views` holds the read models they share. What the routes decide is in
`anomaly_lab.experiments.service`.
"""

from __future__ import annotations

from fastapi import APIRouter

from anomaly_lab.api.routers.experiments import crud, diagnostics, results, runs

router = APIRouter()
for _part in (crud, runs, results, diagnostics):
    router.include_router(_part.router)
