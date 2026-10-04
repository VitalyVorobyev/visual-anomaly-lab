/**
 * The design tokens a mask raster is tinted with, resolved to literal colours.
 *
 * ADR-0021 puts colour in the tokens and nowhere else. The stage's vector layers are SVG and
 * name a token directly (`var(--defect)`), but a painted region, the imported base and a
 * MobileSAM suggestion are rasters: their pixels are tinted on a canvas, and a canvas takes a
 * colour string, not a custom property. So the few tokens a raster needs are read at runtime
 * and re-read when they change, rather than copied out of one theme — which is how a canvas
 * ends up ignoring the light theme.
 *
 * Resolution happens against `document.documentElement`, where `theme.ts` puts the `dark` class,
 * and is re-read on both signals that can move it: the class itself, and the OS preference while
 * the choice is "system".
 */

import { useEffect, useState } from "react";

/** Every colour a mask raster is tinted with, resolved for the theme on screen. */
export interface ScenePalette {
  /** A region that removes from the mask, rather than adding to it. */
  cut: string;
  /** A region whose class is not in the taxonomy, and the imported base of a classless dataset. */
  unknownLabel: string;
  /** A suggested, not-yet-accepted region. */
  suggestion: string;
}

const TOKENS: Record<keyof ScenePalette, string> = {
  cut: "--defect",
  unknownLabel: "--signal",
  suggestion: "--warn",
};

/**
 * What a token paints as before a stylesheet defines it — a non-DOM test environment, or a
 * document whose stylesheet failed to load: nothing, rather than a copy of one theme's values
 * that would quietly go stale. `styles.css` is imported before the first render, so the app
 * itself never sees it.
 */
const UNRESOLVED = "transparent";

function read(): ScenePalette {
  const style =
    typeof globalThis.getComputedStyle === "function"
      ? globalThis.getComputedStyle(globalThis.document.documentElement)
      : null;
  const entries = Object.entries(TOKENS).map(([key, token]) => [
    key,
    style?.getPropertyValue(token).trim() || UNRESOLVED,
  ]);
  return Object.fromEntries(entries) as ScenePalette;
}

export function useScenePalette(): ScenePalette {
  const [palette, setPalette] = useState<ScenePalette>(read);

  useEffect(() => {
    const refresh = () => setPalette(read());
    refresh();

    // The class is what `theme.ts` toggles for an explicit choice…
    const observer = new globalThis.MutationObserver(refresh);
    observer.observe(globalThis.document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    // …and while the choice is "system" nothing is toggled at all, so the OS is the signal.
    const media =
      typeof globalThis.matchMedia === "function"
        ? globalThis.matchMedia("(prefers-color-scheme: dark)")
        : null;
    media?.addEventListener("change", refresh);

    return () => {
      observer.disconnect();
      media?.removeEventListener("change", refresh);
    };
  }, []);

  return palette;
}
