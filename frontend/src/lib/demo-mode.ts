// Demo-mode toggle: makes the dashboard render mock data without a backend.
// Activated via ?demo=1 query (sticks via localStorage) or programmatic call.

const KEY = "pie:demo";

export function isDemoMode(): boolean {
  if (typeof window === "undefined") return false;
  // Build-time forced demo (e.g. Vercel deploys without a backend).
  if (process.env.NEXT_PUBLIC_FORCE_DEMO === "1") return true;
  // Honour ?demo=1 in the current URL even before localStorage is synced —
  // covers the first-paint API calls that fire before any banner mounts.
  // Persisting to localStorage keeps subsequent navigation in demo mode
  // even if the URL no longer carries the flag.
  try {
    const params = new URLSearchParams(window.location.search);
    const v = params.get("demo");
    if (v === "1") {
      window.localStorage.setItem(KEY, "1");
      return true;
    }
    if (v === "0") {
      window.localStorage.removeItem(KEY);
      return false;
    }
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function setDemoMode(on: boolean): void {
  if (typeof window === "undefined") return;
  try {
    if (on) window.localStorage.setItem(KEY, "1");
    else window.localStorage.removeItem(KEY);
  } catch {
    // localStorage unavailable (private browsing, embed) — no-op
  }
}

/** Read ?demo= from the current URL and persist into localStorage. */
export function syncFromUrl(): void {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  const v = params.get("demo");
  if (v === "1") setDemoMode(true);
  else if (v === "0") setDemoMode(false);
}
