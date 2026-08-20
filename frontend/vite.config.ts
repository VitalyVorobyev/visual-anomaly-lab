import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// @vitavision/lab-ui is a published npm package, so its peer dependencies
// (react, react-dom, react-router, lucide-react) resolve to this project's own
// copies like any other package's. While it was a `file:` dependency it carried
// its own node_modules, which meant two React instances in one page -- every
// hook its components called threw "Invalid hook call" -- and both a
// `resolve.alias` and a Vitest `server.deps.inline` entry existed here to force
// a single copy. Both are gone with the cause.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Fixed, because the Tauri shell points the WebView at this URL in development.
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
  },
  test: {
    // M1 tested only pure functions and needed no DOM. M2 has screens whose logic --
    // which warnings block a commit, how a two-channel sample renders -- is only
    // meaningful once rendered, so the suite gets a document. happy-dom rather than
    // jsdom: same API surface for what is used here, considerably faster to start.
    environment: "happy-dom",
    globals: false,
    setupFiles: ["./src/test-setup.ts"],
  },
});
