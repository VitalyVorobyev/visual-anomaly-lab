# Development

Everything a contributor needs that a user does not. Start from the [README](../README.md) for what
the application is and how to run it.

## Toolchain

`uv` for Python, `bun` for the frontend, `cargo` for the Tauri shell. Never `pip`, `npm`, `yarn` or
`pnpm` — lockfiles are committed and the mixed-tool failure mode is expensive to unpick.

```bash
./scripts/setup-hooks.sh                 # installs the pre-commit guard
uv sync --directory backend --extra dl   # the dl extra is optional
cd frontend && bun install
```

The deep-learning dependencies live behind the optional `dl` extra. `pixel_reference`, the whole
evaluation layer and every test but the EfficientAD ones work without torch installed, and CI
installs without it — so anything added to the evaluation path must stay torch-free.

## Checks

```bash
uv run --directory backend pytest
uv run --directory backend ruff check .
uv run --directory backend ruff format --check .
uv run --directory backend mypy            # strict
cd frontend && bun run test && bun run typecheck
./scripts/check-repo-safety.sh             # must exit 0 before any commit
```

Before trusting the accelerator, and before writing wrapper code against a new library:

```bash
./scripts/mps-smoke-test.py
```

After changing any API route or response model, regenerate the typed client and commit the result.
CI fails on a stale file, and the diff *is* the API contract changing:

```bash
./scripts/gen-api-types.sh
```

## Private source images

The project is developed partly against a private industrial inspection dataset, mounted read-only
at `privatedata/` and referenced in place. It is never read into the repository, never copied into a
tracked path, and never leaves the machine.

Three layers keep it that way, and they are load-bearing rather than decorative:

- `.gitignore` excludes `privatedata/` and all `*.bmp` files.
- `scripts/check-repo-safety.sh` fails if anything private, anything under `datasets/`, or any
  oversized file is staged or tracked. **Run it before every commit and push.**
- Stage explicit paths. Never `git add -A` or `git add .`.

Test fixtures are small synthetic PNGs generated in code — never a real dataset file. A handful of
tests assert the composition of the private tree; they are skipped unless pointed at it, and CI
never is:

```bash
ANOMALY_LAB_SHOWCASE_ROOT=/path/to/tree uv run --directory backend pytest -k showcase
```

Public reference datasets live under `/datasets/` and are also uncommitted — for size, not secrecy.
Note the leading slash: an unanchored pattern would also match `backend/src/anomaly_lab/datasets/`,
which is the adapter package.

## Adding a method

A method is one module implementing `AnomalyModel` and one entry in `models/registry.py`. It must
not need a route, a schema, or a line of TypeScript: the method picker and every configuration form
are generated from the plugin's own JSON Schema, and the visualisation views render whatever the
plugin declares rather than branching on its name.

Two rules that are easy to break:

- **Keep heavy imports inside the plugin's functions.** The registry is lazy so that opening the
  method picker does not cost three seconds of torch, and that only holds if every module cooperates.
- **Load pixels through `models/preprocessing.load_array`.** Preprocessing is configuration of the
  *experiment*, not of the model. A method that decodes images its own way makes every comparison
  against it partly a measurement of its resize.

If a change for a new method leaks into the jobs, evaluation, results or UI layers, the plugin
boundary is wrong — fix the boundary, not the caller.

## Conventions worth knowing

- **A metric that could not be computed is `None` and renders as a dash.** Never 0.0.
- **Bound anything whose cost is linear in the dataset and whose value is not, and say what was
  dropped.** Use `models.base.evenly_spaced`, never the first N. A silent truncation reads as "this
  is all there was".
- **Channel count is data, never schema.** No constant, column or layout may encode how many
  acquisition channels a dataset has.
- **Colour comes from the tokens in `frontend/src/styles.css`** — components name `surface`, `line`,
  `fg-muted`, `signal`, `normal`, `defect`, `warn`, never a raw Tailwind ramp step. A raw colour
  compiles, looks nearly right, and quietly ignores the theme.
- **Controls come from `frontend/src/components/ui`.** A raw `<details>` in particular renders with
  no caret, because the base layer drops the UA marker.
- **An empty control means unset.** A field nobody touched sends nothing, so defaults are defined in
  Python alone. Do not "fix" this by pre-filling.

## Documentation

| Document | Contents |
| --- | --- |
| [`system-design.md`](system-design.md) | Architecture, domain model, canonical entity names, API surface, job and evaluation protocols |
| [`roadmap.md`](roadmap.md) | What is built, what is next, and what each stage has to satisfy |
| [`backlog.md`](backlog.md) | Task-level breakdown |
| [`adr/`](adr/) | Architecture decision records — what was decided, why, and what it costs |

**ADRs are immutable once accepted.** A significant new decision gets a new numbered record that
explicitly supersedes the old one; never edit an accepted one in place, and never silently
contradict it.
