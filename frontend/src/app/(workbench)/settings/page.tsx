"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Github, Server, Sparkles, Trash2 } from "lucide-react";

import { isDemoMode, setDemoMode } from "@/lib/demo-mode";
import { SummaryCard } from "@/components/summary-card";
import { cn } from "@/lib/utils";

export default function SettingsPage() {
  const router = useRouter();
  const [demoActive, setDemoActive] = useState(false);
  const [backendUrl, setBackendUrl] = useState<string>("");
  const [storageKeys, setStorageKeys] = useState<string[]>([]);

  useEffect(() => {
    setDemoActive(isDemoMode());
    setBackendUrl(
      (process.env.NEXT_PUBLIC_BACKEND_URL as string | undefined) ??
        "http://localhost:8000",
    );
    if (typeof window !== "undefined") {
      try {
        const keys: string[] = [];
        for (let i = 0; i < window.localStorage.length; i++) {
          const k = window.localStorage.key(i);
          if (k && k.startsWith("pie:")) keys.push(k);
        }
        setStorageKeys(keys);
      } catch {
        // localStorage unavailable
      }
    }
  }, []);

  const toggleDemo = () => {
    const next = !demoActive;
    setDemoMode(next);
    setDemoActive(next);
    router.refresh();
  };

  const clearStorage = () => {
    if (typeof window === "undefined") return;
    try {
      for (const k of storageKeys) window.localStorage.removeItem(k);
      setStorageKeys([]);
      setDemoActive(false);
      router.refresh();
    } catch {
      // no-op
    }
  };

  return (
    <main className="container py-12">
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Configuration
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Settings</h1>
      </header>

      <SummaryCard
        title="Configure how the dashboard runs"
        body={
          <>
            PIEmaker is a thin client over a FastAPI backend. The two knobs
            that matter day-to-day are <strong>demo mode</strong> (route all
            API calls to canned mock data — no backend needed) and the{" "}
            <strong>backend URL</strong> (set at deploy time via{" "}
            <code>NEXT_PUBLIC_BACKEND_URL</code>).
          </>
        }
        recommendations={[
          "Use demo mode for offline previews, screenshots, or Vercel deployments where the backend isn't reachable.",
          "Local dev: keep demo off and run the FastAPI server on :8000.",
        ]}
      />

      <section className="mb-6 rounded-lg border bg-card p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <span
            className={cn(
              "grid h-10 w-10 shrink-0 place-items-center rounded-md",
              demoActive
                ? "bg-amber-200/60 text-amber-800"
                : "bg-secondary text-foreground/70",
            )}
          >
            <Sparkles className="h-5 w-5" />
          </span>
          <div className="flex-1">
            <p className="font-medium">Demo mode</p>
            <p className="mt-1 text-sm text-muted-foreground">
              {demoActive
                ? "Active — every API call returns realistic mock data. No backend traffic."
                : "Inactive — API calls hit the live FastAPI backend at the URL below."}
            </p>
          </div>
          <button
            type="button"
            onClick={toggleDemo}
            className={cn(
              "rounded-md border px-4 py-2 text-sm font-medium",
              demoActive
                ? "border-amber-400/60 bg-amber-400/10 text-amber-900 hover:bg-amber-400/20"
                : "bg-primary text-primary-foreground hover:opacity-90",
            )}
          >
            {demoActive ? "Disable demo mode" : "Enable demo mode"}
          </button>
        </div>
      </section>

      <section className="mb-6 rounded-lg border bg-card p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-secondary text-foreground/70">
            <Server className="h-5 w-5" />
          </span>
          <div className="flex-1">
            <p className="font-medium">Backend URL</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Set at build time via{" "}
              <code className="rounded bg-secondary px-1 py-0.5 font-mono text-xs">
                NEXT_PUBLIC_BACKEND_URL
              </code>
              . Next.js rewrites <code>/api/backend/*</code> to this origin.
            </p>
            <p className="mt-3 rounded-md border bg-secondary/50 p-3 font-mono text-xs">
              {backendUrl}
            </p>
          </div>
        </div>
      </section>

      <section className="mb-6 rounded-lg border bg-card p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-secondary text-foreground/70">
            <Trash2 className="h-5 w-5" />
          </span>
          <div className="flex-1">
            <p className="font-medium">Local storage</p>
            <p className="mt-1 text-sm text-muted-foreground">
              {storageKeys.length === 0
                ? "No PIEmaker keys in localStorage."
                : `${storageKeys.length} key(s): ${storageKeys.join(", ")}`}
            </p>
          </div>
          <button
            type="button"
            onClick={clearStorage}
            disabled={storageKeys.length === 0}
            className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-secondary disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </section>

      <section className="mb-6 rounded-lg border bg-card p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-secondary text-foreground/70">
            <Github className="h-5 w-5" />
          </span>
          <div className="flex-1">
            <p className="font-medium">Source &amp; documentation</p>
            <p className="mt-1 text-sm text-muted-foreground">
              All formulas, validation rules, and trust diagnostics are open
              source. Built by Prima Hanura Akbar.
            </p>
            <a
              href="https://github.com/matiyashu/PIEmaker-predicted-incrementality-by-experiment-"
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-secondary"
            >
              <Github className="h-3.5 w-3.5" />
              github.com/matiyashu/PIEmaker
            </a>
          </div>
        </div>
      </section>
    </main>
  );
}
