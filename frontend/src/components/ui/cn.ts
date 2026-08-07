/**
 * Class composition, and the shared focus treatment.
 *
 * The old primitives *appended* the caller's `className` to their own base string, so
 * `<Button className="px-6">` produced `px-3 … px-6` and won or lost on source order rather
 * than on intent. `twMerge` resolves that by conflict group: the caller's `px-6` replaces
 * the base `px-3` and leaves everything else alone.
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * One focus treatment for every interactive element in the application.
 *
 * There was none before: `inputClasses` set `focus:outline-none` and replaced it with a
 * border-colour change, and `Button` had nothing at all, so keyboard navigation was
 * invisible. `outline` rather than `ring` because an outline follows the element's own
 * border-radius and needs no offset *colour* -- a ring offset has to know which surface it
 * sits on, and these primitives are used on `ground`, `surface` and `raised` alike.
 *
 * `focus-visible` rather than `focus`, so a mouse click does not leave a ring behind.
 */
export const focusRing =
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal";

/** For controls whose own box is the hit target, where an offset ring would clip. */
export const focusRingInset = "focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-signal";
