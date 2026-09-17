"""Research code: measured campaigns that decide what the application should ship.

Not part of `anomaly_lab` and never imported by it (ADR-0038). The wheel packages
`src/anomaly_lab` alone, so nothing here can reach a user's installation; what reaches
them is a verdict — a default in a plugin, a row in `docs/measurements.md`.

It lives inside `backend/` rather than at the repository root for one practical reason:
these campaigns must be linted, type-checked and tested by the same commands as the
application, and `pyproject.toml` is what defines those commands.
"""
