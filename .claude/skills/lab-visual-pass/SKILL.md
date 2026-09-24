---
name: lab-visual-pass
description: Run visual-anomaly-lab against public data on a scratch catalogue, screenshot every screen in two viewports and both themes, and audit it against this repo's UI rules. Use after any frontend change that alters a screen, before closing a UI backlog item, for the roadmap's visual QA pass, or when the user asks to "look at the app", "check the UI", "screenshot the screens" or "do a visual pass".
---

# Lab visual pass

A picture of the running app, on data anyone may see, checked against the rules the code review
cannot see. Tests prove behaviour; this proves what a person meets.

## Rules of engagement

- **Public data only.** VisA (and GKN) under the repo's `datasets/`. Never the private showcase
  images, not even one "to check the format" — they live outside the tree and stay there.
- **Scratch catalogue only.** The backend runs with `ANOMALY_LAB_DATA_DIR` pointed into the session
  scratchpad, so the user's own `data/` is never read or written. Reuse an existing scratch
  `labdata/` if the session has one; seeding is idempotent.
- **Screenshots stay in the scratchpad.** They are never staged. (`book/src/images/*.jpg` is the one
  committed home for screenshots of the app, and only when the user asks for one there.)
- Stop both servers when done.

## 1. Start a scratch lab

Pick free ports (8010/5174 unless taken). `$S` is the session scratchpad.

```bash
(ANOMALY_LAB_DATA_DIR=$S/labdata nohup uv run --directory backend uvicorn \
   anomaly_lab.api.app:create_app --factory --port 8010 > $S/backend.log 2>&1 &)
(cd frontend && VITE_API_BASE_URL=http://127.0.0.1:8010 nohup bun run dev --port 5174 \
   --strictPort > $S/frontend.log 2>&1 &)
for i in $(seq 1 40); do curl -sf http://127.0.0.1:8010/api/health >/dev/null && break; sleep 1; done
```

## 2. Seed it

```bash
python3 .claude/skills/lab-visual-pass/scripts/seed.py --api http://127.0.0.1:8010 \
  --visa-root datasets/VisA_20220922 --category candle > $S/ids.json
```

One VisA class, its published split, a full-frame region profile, and two trained and scored
`pixel_reference` runs (the only method that needs no torch, and fast). `--few-shot` adds a 5-shot
`defect` reference split and a scored `color_prototype` run, and the shots then include screens
19–22, a few-shot run's Overview, Samples, Benchmark and sample page. It prints the ids the next step
needs. If VisA is absent, say so and stop — do not substitute any other data.

## 3. Shoot and audit

```bash
uv run --with playwright python .claude/skills/lab-visual-pass/scripts/shots.py \
  --ids $S/ids.json --out $S/shots            # all 19 screens (23 with --few-shot) × {light,dark} × {1440,1024}
# --quick for light 1440 only; --only 11-exp-overview 15-exp-sample to re-shoot a few
```

(First run in a fresh environment: `uv run --with playwright playwright install chromium`.)

Each line is a screen and what the DOM audit found — `scroll` (an unmarked nested scroller),
`nest` (a control inside a link), `unnamed` (a control with no accessible name), `errors` (console
errors and page crashes). A crash screen in a later shot can be the tail of an earlier page's
crash in the same context — re-shoot the screen alone before believing it.

## 4. Read the pictures

Open each screenshot with the Read tool and judge what the audit cannot. The checklist, in the order
problems usually show up:

1. **The job of the screen is obvious.** One primary action, visibly primary; the page title names
   the thing (a dataset, a run, `group/external_id`) — never an internal database id.
2. **The next step is reachable.** From every screen, the way back and the way on are links. A
   disabled control says *why* in visible text, not only in a `title=` tooltip.
3. **States are designed.** Loading shows `Skeleton`, empty shows `Empty` with the action that fills
   it, errors show `ErrorBox`/`Callout` with the message — none of them a blank area.
4. **Tokens, not ramp steps.** Colours come from `surface`, `line`, `fg-muted`, `signal`,
   `normal`, `defect`, `warn`. Something that looks almost right in light and wrong in dark is
   usually a raw `slate-500` or `#hex`.
5. **One scroller per screen**, owned by the layout (`data-layout`/`data-scroll`); a peer rail is
   the only exception.
6. **Density and hierarchy** at 1024×768: nothing clipped, nothing that needs horizontal scroll,
   readouts aligned, numbers in tabular figures.
7. **Focus is visible** on every control you can tab to (check one screen by pressing Tab).
8. **Vocabulary is consistent** — the same thing has the same name on every screen, and a method
   shows its title, not its registry key.

## 5. Report

Group findings by screen, most severe first, each with the screenshot filename, what is wrong and the
file that renders it. Fixes that belong in a shared primitive go upstream to `@vitavision/lab-ui`,
never into a local copy. Open items that are not fixed now go into `docs/backlog.md` under
Interface.
