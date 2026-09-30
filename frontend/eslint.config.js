// @ts-check
import { recommended, tokensOnly } from "@vitavision/config-eslint";

export default [
  { ignores: ["src-tauri/**", "e2e/.state/**", "test-results/**", "playwright-report/**"] },
  ...recommended({ tsconfigRootDir: import.meta.dirname }),
  {
    // Exception from the toolchain upgrade (lab-ui PLAN L2-1), removed screen by screen in L3.
    // These rules come with the React Compiler and find setState called inside an effect and
    // refs read or written during render. Every fix changes when a screen renders, and L2 changes no
    // UI, so they report as warnings until the screen that holds each one migrates.
    rules: {
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/refs": "warn",
      "react-hooks/immutability": "warn",
    },
  },
  // Gate G5.1 (lab-ui PLAN §5): in src/, colour comes from the @vitavision/ui design tokens —
  // no raw Tailwind palette classes, no hex literals (tests and stories are exempt by the rule).
  tokensOnly(["src/**"]),
  {
    // Literal colours on purpose, one reason each:
    files: [
      // Class colours are data: stored per class by the backend and edited with a colour picker.
      "src/api/classPalette.ts",
      // Draws when the app failed to start — possibly without its stylesheet, so no tokens.
      "src/components/CrashScreen.tsx",
    ],
    rules: { "vitavision/tokens-only": "off" },
  },
];
