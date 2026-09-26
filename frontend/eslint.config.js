// @ts-check
import { recommended } from "@vitavision/config-eslint";

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
  // Gate G5.1 (`tokensOnly` from @vitavision/config-eslint) is enabled per directory as the
  // screens migrate to the shared visual language, from L3 on.
];
