import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

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
