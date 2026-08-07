/**
 * Which of the two palettes is on screen.
 *
 * Until now dark mode was `prefers-color-scheme` and nothing else: every component carried
 * `dark:` variants, and no one could override the OS. That is the wrong default for this
 * application specifically -- judging an anomaly map against a bright room and against a
 * dark one are different tasks, and a researcher switches between them within a session.
 *
 * Three states, not two. "system" is a real choice and has to survive a reload as itself,
 * otherwise the first time the OS flips at sunset the app disagrees with everything else on
 * the machine and there is no way back to following it.
 *
 * The class is applied to <html> rather than <body> so the page background painted before
 * React mounts is already correct. `applyTheme` is duplicated as an inline script in
 * index.html for exactly that reason -- see the comment there before changing this shape.
 */

export type ThemeChoice = "light" | "dark" | "system";

const STORAGE_KEY = "anomaly-lab-theme";

const query = () =>
  typeof window.matchMedia === "function" ? window.matchMedia("(prefers-color-scheme: dark)") : null;

export function readThemeChoice(): ThemeChoice {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    // Private browsing, or a WebView with storage disabled. Following the OS is a fine
    // answer to "I could not read your preference", and better than refusing to render.
  }
  return "system";
}

/** What `choice` actually resolves to right now. */
export function resolveTheme(choice: ThemeChoice): "light" | "dark" {
  if (choice !== "system") return choice;
  return query()?.matches ? "dark" : "light";
}

function paint(choice: ThemeChoice): void {
  document.documentElement.classList.toggle("dark", resolveTheme(choice) === "dark");
}

export function setThemeChoice(choice: ThemeChoice): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // Not worth failing the click over; the choice still applies for this session.
  }
  paint(choice);
}

/**
 * Paint the stored choice and keep following the OS while the choice is "system".
 *
 * Returns an unsubscribe so a test can tear the listener down; the app never does.
 */
export function initTheme(): () => void {
  paint(readThemeChoice());

  const media = query();
  if (media === null) return () => {};

  const onSystemChange = () => {
    if (readThemeChoice() === "system") paint("system");
  };
  media.addEventListener("change", onSystemChange);
  return () => media.removeEventListener("change", onSystemChange);
}
