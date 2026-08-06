import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

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
});
