import { expect, test } from "@playwright/test";

/**
 * Smoke walk-through of the V.4 dashboard in demo mode.
 *
 * Covers the click-paths a non-technical reviewer would take:
 *   1. Land on /, see hero + "Built by Prima Hanura Akbar"
 *   2. Activate demo mode, navigate to /dashboard
 *   3. Visit each workbench surface in turn; assert each page renders
 *      with its title heading (proves the route loads + dispatcher
 *      returns mock data without throwing)
 */

test("landing page renders with attribution + demo CTA", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /predict campaign incrementality/i })).toBeVisible();
  await expect(page.getByText(/Prima Hanura Akbar/i)).toBeVisible();
  await expect(page.getByRole("link", { name: /try demo mode/i })).toBeVisible();
});

test("dashboard renders after demo flag is set", async ({ page }) => {
  await page.goto("/dashboard?demo=1");
  await expect(page.getByRole("heading", { name: /piemaker workbench|empty workbench/i })).toBeVisible();
});

test("every workbench surface renders in demo mode", async ({ page }) => {
  const routes = [
    { path: "/dashboard?demo=1", heading: /piemaker workbench|empty workbench/i },
    { path: "/upload?demo=1", heading: /upload/i },
    { path: "/cleaning?demo=1", heading: /cleaning/i },
    { path: "/donor-pool?demo=1", heading: /donor pool/i },
    { path: "/labels?demo=1", heading: /labels|att/i },
    { path: "/features?demo=1", heading: /feature/i },
    { path: "/models?demo=1", heading: /model/i },
    { path: "/predict?demo=1", heading: /predict|forecast/i },
    { path: "/portfolio?demo=1", heading: /portfolio|score a media plan/i },
    { path: "/decisions?demo=1", heading: /decision/i },
    { path: "/drift?demo=1", heading: /drift/i },
    { path: "/simulator?demo=1", heading: /simulator|reallocate/i },
    { path: "/diagnostics?demo=1", heading: /diagnostics|trust the forecast/i },
    { path: "/settings", heading: /settings/i },
    { path: "/faq", heading: /faq|reference/i },
  ];

  for (const { path, heading } of routes) {
    await page.goto(path);
    await expect(
      page.getByRole("heading", { name: heading }).first(),
    ).toBeVisible({ timeout: 10_000 });
  }
});

test("decision-disagreement curves chart renders on /decisions", async ({ page }) => {
  await page.goto("/decisions?demo=1");
  // Wave 3 chart section title
  await expect(
    page.getByText(/decision-disagreement curves/i),
  ).toBeVisible({ timeout: 10_000 });
});
