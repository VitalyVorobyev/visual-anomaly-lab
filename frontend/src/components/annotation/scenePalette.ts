/**
 * The design tokens, as literal colours a canvas can paint with.
 *
 * ADR-0021 puts colour in `styles.css` and nowhere else, and every component names a token
 * rather than a ramp step. Konva is the one surface that cannot honour that directly: it paints
 * into a canvas and takes colour strings, not class names. The answer is not to give up and
 * hardcode — which is what the scene did, with fifteen literals copied out of the *dark* palette,
 * so the whole editor ignored the light theme — but to read the tokens at runtime and repaint
 * when they change.
 *
 * Resolution happens against `document.documentElement`, where `theme.ts` puts the `dark` class,
 * and is re-read on both signals that can move it: the class itself, and the OS preference while
 * the choice is "system".
 */

import { useEffect, useState } from "react";

/** Every colour the annotation scene paints with, resolved for the theme on screen. */
export interface ScenePalette {
  /** Behind the source image — the "paper" the photograph sits on. */
  canvas: string;
  /** The 1px outline marking the source frame's true extent. */
  frame: string;
  /** Interaction accent: selection outlines, pending geometry, the keyboard cursor. */
  signal: string;
  /** A region that removes from the mask, rather than adding to it. */
  cut: string;
  /** Fallback for a shape whose label key is not in the taxonomy. */
  unknownLabel: string;
  /** Assist prompts: include and exclude. */
  positive: string;
  negative: string;
  /** A suggested, not-yet-accepted region. */
  suggestion: string;
}

const TOKENS: Record<keyof ScenePalette, string> = {
  canvas: "--canvas",
  frame: "--line-strong",
  signal: "--signal",
  cut: "--defect",
  unknownLabel: "--signal",
  positive: "--normal",
  negative: "--defect",
  suggestion: "--warn",
};

/** What to paint before the stylesheet has resolved, and in a non-DOM test environment. */
const FALLBACK: ScenePalette = {
  canvas: "#08090a",
  frame: "#2a2f36",
  signal: "#3bc9db",
  cut: "#f87171",
  unknownLabel: "#3bc9db",
  positive: "#4ade80",
  negative: "#f87171",
  suggestion: "#fbbf24",
};

function read(): ScenePalette {
  if (typeof globalThis.getComputedStyle !== "function") return FALLBACK;
  const style = globalThis.getComputedStyle(globalThis.document.documentElement);
  const entries = Object.entries(TOKENS).map(([key, token]) => {
    const value = style.getPropertyValue(token).trim();
    return [key, value || FALLBACK[key as keyof ScenePalette]];
  });
  return Object.fromEntries(entries) as unknown as ScenePalette;
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

/**
 * `#rrggbb` plus an alpha byte, for a fill that should tint rather than cover.
 *
 * Tolerates a token that resolved to something other than six hex digits — a themed value can
 * legitimately be `oklch(...)` — by returning it unchanged rather than producing a colour string
 * the canvas will silently ignore.
 */
export function withAlpha(color: string, alpha: number): string {
  if (!/^#[0-9a-fA-F]{6}$/.test(color)) return color;
  const byte = Math.round(Math.min(1, Math.max(0, alpha)) * 255);
  return `${color}${byte.toString(16).padStart(2, "0")}`;
}
