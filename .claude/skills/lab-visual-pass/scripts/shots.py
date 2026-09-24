"""Screenshot every screen of a seeded scratch lab and audit what a picture cannot show.

    uv run --with playwright python shots.py --ids ids.json --out <scratchpad>/shots \
        [--app http://localhost:5174] [--only 11-exp-overview 15-exp-sample] [--quick] [--states]

`ids.json` is what seed.py printed. `--quick` takes light 1440 only. The first run needs
`uv run --with playwright playwright install chromium`.

Per screen and viewport it prints one line of findings, the machine-checkable half of the
checklist in SKILL.md:
  scroll   an overflowing vertical scroller that is not a marked layout scroller
           (`data-layout`, `data-scroll`) — the nested-scroll bug
  nest     an interactive control inside an <a>
  unnamed  a button or link with no accessible name
  errors   console errors, including React warnings
The screenshots are for the other half — hierarchy, density, states — which only a reader can judge.

`--states` shoots the transient states a resting screenshot never shows, light 1440 only:
  <name>-pending.png  every API request held open, so the screen's loading state is what renders
  <name>-error.png    every API request answered 500, so its error state is what renders
and walks Tab through the resting screen, reporting
  focus    a stop with no visible focus indicator (no outline and no ring shadow)
  reason   a disabled button whose only explanation is a `title` tooltip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

AUDIT = """() => {
  const label = e => e.tagName.toLowerCase() + (e.dataset.scroll ? `[data-scroll=${e.dataset.scroll}]`
    : e.id ? '#' + e.id : '.' + String(e.className).split(' ').slice(0, 3).join('.'));
  const scroll = [...document.querySelectorAll('*')].filter(e => {
    const s = getComputedStyle(e);
    return /(auto|scroll)/.test(s.overflowY) && e.scrollHeight > e.clientHeight + 2
      && !e.hasAttribute('data-scroll') && !e.hasAttribute('data-layout') && e.tagName !== 'HTML' && e.tagName !== 'BODY';
  }).map(label);
  const nest = [...document.querySelectorAll('a button, a input, a select, a [role=checkbox], a [role=switch]')]
    .map(e => label(e.closest('a')) + ' > ' + label(e));
  const unnamed = [...document.querySelectorAll('button, a[href]')].filter(e =>
    !(e.getAttribute('aria-label') || e.getAttribute('aria-labelledby') || e.textContent.trim() || e.title || (e.labels && e.labels.length))
  ).map(label);
  return { scroll, nest, unnamed };
}"""


def screens(ids: dict) -> dict[str, str]:
    d, (a, b) = ids["dataset_id"], ids["experiment_ids"][:2]
    s, i = ids["defect_sample_id"], ids["defect_image_id"]
    return {
        "01-catalogue": "/",
        "02-import": "/import",
        "03-browse": f"/datasets/{d}",
        "04-sample": f"/datasets/{d}/samples/{s}",
        "05-annot-queue": f"/datasets/{d}/annotate",
        "06-annot-editor": f"/datasets/{d}/annotate/{s}/{i}",
        "07-prepare": f"/datasets/{d}/prepare",
        "08-splits": f"/datasets/{d}/splits",
        "09-ds-experiments": f"/datasets/{d}/experiments",
        "10-new-experiment": f"/datasets/{d}/experiments/new",
        "11-exp-overview": f"/experiments/{a}?tab=overview",
        "12-exp-samples": f"/experiments/{a}?tab=samples",
        "13-exp-benchmark": f"/experiments/{a}?tab=benchmark",
        "14-exp-training": f"/experiments/{a}?tab=training",
        "15-exp-sample": f"/experiments/{a}/samples/{s}",
        "16-compare": f"/compare?ids={a},{b}",
        "17-compare-sample": f"/compare/samples/{s}?ids={a},{b}",
        "18-experiments": "/experiments",
        "23-studio": f"/datasets/{d}/studio/defect?refs={s}&focus={s}&method=color_prototype",
        # A few-shot segmentation run (ADR-0040), when the seed made one.
        **(
            {
                "19-fs-overview": f"/experiments/{f}?tab=overview",
                "20-fs-samples": f"/experiments/{f}?tab=samples",
                "21-fs-benchmark": f"/experiments/{f}?tab=benchmark",
                "22-fs-sample": f"/experiments/{f}/samples/{s}",
            }
            if (f := ids.get("few_shot_run")) is not None
            else {}
        ),
    }


FOCUS = """() => {
  const e = document.activeElement;
  if (!e || e === document.body) return null;
  const s = getComputedStyle(e);
  const visible = (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0)
    || (s.boxShadow && s.boxShadow !== 'none');
  const name = e.getAttribute('aria-label') || e.textContent.trim().slice(0, 40) || e.tagName.toLowerCase();
  return { visible, name };
}"""

REASONS = """() => [...document.querySelectorAll('button[disabled], [role=button][aria-disabled=true]')]
  .filter(e => e.getAttribute('title') && !e.getAttribute('aria-describedby'))
  .map(e => (e.getAttribute('aria-label') || e.textContent.trim()).slice(0, 40))"""

HOLD_SECONDS = 1.5
# Past the one retry a failed read gets (`frontend/src/api/retry.ts`), so the error is what renders.
ERROR_SECONDS = 4.0
TAB_STOPS = 40


def _fresh(page, url: str) -> None:
    """A new document, not a hash change: every screen is a hash route, and a same-document
    navigation keeps the query cache, which would draw cached data over the state under test."""
    page.goto("about:blank")
    page.goto(url)


def _states(page, args: argparse.Namespace, name: str, path: str) -> dict[str, list[str]]:
    """Pending, error and focus for one screen; the screen is re-opened for each."""
    held: list = []
    backend = f"{args.api.rstrip('/')}/api/**"
    page.route(backend, lambda route: held.append(route))
    _fresh(page, f"{args.app}/#{path}")
    page.wait_for_timeout(HOLD_SECONDS * 1000)
    page.screenshot(path=str(args.out / f"{name}-pending.png"))
    page.unroute(backend)
    for route in held:
        try:
            route.abort()
        except Exception:
            pass

    page.route(
        backend,
        lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body='{"detail":"stubbed failure for the visual pass"}',
        ),
    )
    _fresh(page, f"{args.app}/#{path}")
    page.wait_for_timeout(ERROR_SECONDS * 1000)
    page.screenshot(path=str(args.out / f"{name}-error.png"))
    page.unroute(backend)

    _fresh(page, f"{args.app}/#{path}")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    unfocused: list[str] = []
    seen: set[str] = set()
    for _ in range(TAB_STOPS):
        page.keyboard.press("Tab")
        stop = page.evaluate(FOCUS)
        if stop is None:
            continue
        if not stop["visible"] and stop["name"] not in seen:
            unfocused.append(stop["name"])
        seen.add(stop["name"])
    return {"focus": unfocused, "reason": page.evaluate(REASONS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--app", default="http://localhost:5174")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--states", action="store_true")
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8010",
        help="The backend's origin; --states holds or fails only its requests, never the app's own.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = screens(json.loads(args.ids.read_text()))
    passes = (
        [("light", 1440, 900)]
        if args.quick or args.states
        else [("light", 1440, 900), ("light", 1024, 768), ("dark", 1440, 900), ("dark", 1024, 768)]
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for theme, width, height in passes:
            context = browser.new_context(viewport={"width": width, "height": height})
            context.add_init_script(f"localStorage.setItem('anomaly-lab-theme', '{theme}')")
            page = context.new_page()
            errors: list[str] = []
            page.on("console", lambda m: m.type == "error" and errors.append(m.text[:160]))
            page.on("pageerror", lambda e: errors.append(f"pageerror: {str(e)[:160]}"))
            for name, path in table.items():
                if args.only and name not in args.only:
                    continue
                if args.states:
                    problems = {k: v for k, v in _states(page, args, name, path).items() if v}
                    print(f"states {name:18} {problems or 'ok'}")
                    errors.clear()
                    continue
                page.goto(f"{args.app}/#{path}")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(800)
                page.screenshot(path=str(args.out / f"{name}-{theme}-{width}.png"))
                found = page.evaluate(AUDIT)
                found["errors"] = errors[:3]
                problems = {k: v for k, v in found.items() if v}
                print(f"{theme:5} {width} {name:18} {problems or 'ok'}")
                errors.clear()
            context.close()
        browser.close()


if __name__ == "__main__":
    main()
