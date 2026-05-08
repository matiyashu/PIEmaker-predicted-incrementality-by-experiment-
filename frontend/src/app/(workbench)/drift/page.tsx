"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";

import {
  type DriftResponse,
  type DriftSeverity,
  type DriftVerdict,
  checkDrift,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const SEVERITY_BADGE: Record<DriftSeverity, string> = {
  stable: "bg-emerald-500 text-white",
  moderate: "bg-amber-400 text-black",
  severe: "bg-destructive text-white",
};

const VERDICT_BADGE: Record<DriftVerdict, string> = {
  stable: "bg-emerald-500 text-white",
  watch: "bg-amber-400 text-black",
  retrain_recommended: "bg-destructive text-white",
};

function DriftInner() {
  const params = useSearchParams();
  const [uploadId, setUploadId] = useState(params.get("upload_id") ?? "");
  const [onlyNonRct, setOnlyNonRct] = useState(true);
  const [data, setData] = useState<DriftResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const onRun = async () => {
    if (!uploadId.trim()) {
      setError("Enter an upload_id");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await checkDrift({ uploadId: uploadId.trim(), onlyNonRct });
      setData(r);
      setExpanded(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const maxPsiDisplay = data ? Math.max(0.5, data.max_psi) : 0.5;

  return (
    <main className="container py-12">
      <header className="mb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 4 · Drift Monitoring
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Feature distribution drift</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Population Stability Index (PSI) per feature: scoring distribution
          vs. training feature_store. Standard bands: <strong>stable</strong>{" "}
          (PSI &lt; 0.10), <strong>moderate</strong> (0.10–0.25),{" "}
          <strong>severe</strong> (≥ 0.25). Verdict: stable, watch (3+ moderate
          drifters), or retrain_recommended (any severe drifter).
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      <section className="mb-8 grid gap-3 rounded-md border p-5 md:grid-cols-3">
        <label className="flex flex-col gap-1 text-xs md:col-span-2">
          <span className="font-mono text-muted-foreground">upload_id</span>
          <input
            value={uploadId}
            onChange={(e) => setUploadId(e.target.value)}
            placeholder="paste an upload_id from /upload"
            className="w-full rounded-md border bg-background px-2 py-1 text-sm"
          />
        </label>
        <div className="flex items-end gap-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={onlyNonRct}
              onChange={(e) => setOnlyNonRct(e.target.checked)}
            />
            non-RCT only
          </label>
          <button
            type="button"
            onClick={() => void onRun()}
            disabled={busy}
            className="rounded-md border px-4 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          >
            {busy ? "Computing…" : "Check drift"}
          </button>
        </div>
      </section>

      {data && (
        <>
          <section className="mb-6 rounded-md border p-5">
            <div className="flex flex-wrap items-baseline gap-3">
              <p className="text-sm uppercase tracking-wider text-muted-foreground">
                Verdict
              </p>
              <span
                className={cn(
                  "rounded-full px-3 py-1 text-sm font-medium",
                  VERDICT_BADGE[data.verdict],
                )}
              >
                {data.verdict.replace("_", " ")}
              </span>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">
              {data.rationale}
            </p>
          </section>

          <section className="mb-6 grid gap-4 md:grid-cols-4">
            <Card label="Max PSI" value={data.max_psi.toFixed(3)} />
            <Card label="Mean PSI" value={data.mean_psi.toFixed(3)} />
            <Card
              label="Training rows"
              value={String(data.n_training_rows)}
              hint={`feature_set_version: ${data.feature_set_version}`}
            />
            <Card label="Scoring rows" value={String(data.n_scoring_rows)} />
          </section>

          <section className="mb-6 rounded-md border p-4">
            <p className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">
              Severity distribution
            </p>
            <div className="flex flex-wrap gap-2">
              {(Object.entries(data.severity_counts) as [
                DriftSeverity,
                number,
              ][]).map(([severity, count]) => (
                <span
                  key={severity}
                  className={cn(
                    "rounded-full px-3 py-1 text-xs font-medium",
                    SEVERITY_BADGE[severity],
                  )}
                >
                  {severity}: {count}
                </span>
              ))}
            </div>
          </section>

          <section>
            <h2 className="mb-3 text-lg font-medium">Per-feature PSI</h2>
            <ul className="space-y-2">
              {data.drifts.map((d) => {
                const pct = Math.min(100, (d.psi / maxPsiDisplay) * 100);
                const isOpen = expanded === d.feature;
                return (
                  <li
                    key={d.feature}
                    className="rounded-md border p-3 text-sm"
                  >
                    <button
                      type="button"
                      onClick={() => setExpanded(isOpen ? null : d.feature)}
                      className="flex w-full items-center justify-between gap-3 text-left"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-xs">{d.feature}</span>
                        <span className="text-xs text-muted-foreground">
                          {d.kind}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="tabular-nums">{d.psi.toFixed(4)}</span>
                        <span
                          className={cn(
                            "rounded-full px-2 py-0.5 text-xs",
                            SEVERITY_BADGE[d.severity],
                          )}
                        >
                          {d.severity}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {isOpen ? "▾" : "▸"}
                        </span>
                      </div>
                    </button>
                    <div className="mt-2 h-2 w-full rounded-full bg-secondary">
                      <div
                        className={cn(
                          "h-2 rounded-full",
                          d.severity === "severe"
                            ? "bg-destructive"
                            : d.severity === "moderate"
                              ? "bg-amber-400"
                              : "bg-emerald-500",
                        )}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    {isOpen && d.bins.length > 0 && (
                      <div className="mt-3 overflow-x-auto rounded-md border">
                        <table className="w-full text-xs">
                          <thead className="bg-secondary">
                            <tr>
                              <th className="px-2 py-1 text-left">Bin</th>
                              <th className="px-2 py-1 text-right">
                                Expected (train)
                              </th>
                              <th className="px-2 py-1 text-right">
                                Actual (score)
                              </th>
                              <th className="px-2 py-1 text-right">Δ</th>
                            </tr>
                          </thead>
                          <tbody>
                            {d.bins.map((b, i) => (
                              <tr key={i} className="border-t">
                                <td className="px-2 py-1 font-mono">{b.bin}</td>
                                <td className="px-2 py-1 text-right tabular-nums">
                                  {b.expected.toFixed(3)}
                                </td>
                                <td className="px-2 py-1 text-right tabular-nums">
                                  {b.actual.toFixed(3)}
                                </td>
                                <td
                                  className={cn(
                                    "px-2 py-1 text-right tabular-nums",
                                    b.delta > 0
                                      ? "text-emerald-700"
                                      : b.delta < 0
                                        ? "text-destructive"
                                        : "",
                                  )}
                                >
                                  {b.delta > 0 ? "+" : ""}
                                  {b.delta.toFixed(3)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      )}
    </main>
  );
}

function Card({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-md border p-4">
      <p className="text-xs uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export default function DriftPage() {
  return (
    <Suspense
      fallback={
        <main className="container py-12">
          <p className="text-muted-foreground">Loading drift…</p>
        </main>
      }
    >
      <DriftInner />
    </Suspense>
  );
}
