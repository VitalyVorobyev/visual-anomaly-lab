/**
 * This app's own `localStorage` key for its theme choice.
 *
 * `@vitavision/lab-ui`'s `initTheme`/`readThemeChoice`/`setThemeChoice` take the key as an
 * argument rather than assuming one, precisely so a consumer that already had a stored
 * preference under its own key does not lose it when it starts sharing the theme module.
 * Kept in agreement with the inline no-flash script in `index.html`, which cannot import
 * this constant (it runs before any module is fetched) and so repeats the literal instead.
 */
export const THEME_STORAGE_KEY = "anomaly-lab-theme";
