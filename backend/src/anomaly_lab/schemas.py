"""Shared configuration for models that cross the API boundary.

At the package root rather than under `api/`, because the domain entities use it too and
`domain/` must not depend on the API layer (§3).

Pydantic marks a field with a default as *not required* in JSON Schema, which is right
for a request — a client may omit it — and wrong for a response, where the server always
emits it. Left alone, that makes every list field optional in the generated TypeScript,
and every screen litters itself with `?? []` for a case that cannot happen.

`json_schema_serialization_defaults_required` fixes it at the source: the serialization
schema says these fields are always present, because they are. FastAPI generates separate
input and output schemas for a model used in both directions, so a request body stays as
permissive as it was.
"""

from __future__ import annotations

from pydantic import ConfigDict

API_MODEL_CONFIG = ConfigDict(frozen=True, json_schema_serialization_defaults_required=True)
