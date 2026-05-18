import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright smoke tests for PIEmaker frontend (V.4 Wave 5).
 *
 * Runs against the Next.js dev server in demo mode so no backend is
 * required. CI uses the same config — see .github/workflows/ci.yml.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",

  use: {
    baseURL: "http://localhost:3765",
    trace: "retain-on-failure",
    // Demo mode is the safest test surface: every page renders without a
    // backend, no upload IDs needed, no state to manage.
    storageState: undefined,
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: {
    command: "npm run dev",
    url: "http://localhost:3765",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      NEXT_PUBLIC_FORCE_DEMO: "1",
    },
  },
});
