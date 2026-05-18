import { test } from "@playwright/test";

/**
 * Capture real screenshots of the running dev server for README assets.
 * Each shot is saved as a PNG under PIEmaker/assets/.
 *
 * Run with:  npx playwright test screenshots.spec.ts --project=chromium
 * Skipped in CI (no `screenshots: true` config); manual local run only.
 */

const shots = [
  { route: "/", name: "screenshot-landing.png", waitFor: "PIEmaker" },
  { route: "/dashboard?demo=1", name: "screenshot-dashboard.png", waitFor: "Donor pool" },
  { route: "/diagnostics?demo=1", name: "screenshot-diagnostics.png", waitFor: "Feature ablation" },
  { route: "/decisions?demo=1", name: "screenshot-decisions.png", waitFor: "Rank a portfolio" },
  { route: "/simulator?demo=1", name: "screenshot-simulator.png", waitFor: "Reallocate the budget" },
  { route: "/models?demo=1", name: "screenshot-models.png", waitFor: "Train, evaluate, promote" },
] as const;

test.describe.configure({ mode: "serial" });

for (const { route, name, waitFor } of shots) {
  test(`screenshot ${name}`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(route, { waitUntil: "networkidle" });
    await page.waitForFunction(
      (substr) => document.body.textContent?.includes(substr),
      waitFor,
      { timeout: 20_000 },
    );
    // Let Recharts settle (it animates).
    await page.waitForTimeout(800);
    await page.screenshot({
      path: `../assets/${name}`,
      fullPage: true,
    });
  });
}
