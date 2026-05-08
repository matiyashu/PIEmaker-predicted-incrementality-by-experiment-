"use client";

import { Suspense, useCallback, useEffect, useState } from "react";

import {
  type DriftResponse,
  type DriftSeverity,
  type DriftVerdict,
  checkDrift,
} from "@/lib/api";
import { isDemoMode } from "@/lib/demo-mode";
import { useUploadId } from "@/lib/use-upload-id";
import { cn } from "@/lib/utils";
import { SummaryCard } from "@/components/summary-card";

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
  const { uploadId: resolvedUploadId } = useUploadId();
  const [uploadId, setUploadId] = useState(resolvedUploadId ?? "");
  const [onlyNonRct, setOnlyNonRct] = useState(true);
  const [data, setData] = useState<DriftResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const onRun = useCallback(async () => {
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
  }, [uploadId, onlyNonRct]);

  useEffect(() => {
    if (isDemoMode() && uploadId && !data && !busy) void onRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId]);

  const maxPsiDisplay = data ? Math.max(0.5, data.max_psi) : 0.5;

  return (
    <main className="container py-12">
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 4 · Drift Monitoring
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Feature distribution drift</h1>
      </header>

      <SummaryCard
        title="What you're seeing"
        body={
          <>
            Population Stability Index (PSI) per feature, comparing the
            scoring distribution against the training set the model was
            built on. Bands: <strong>stable &lt;0.10</strong>,{" "}
            <strong>moderate 0.10–0.25</strong>,{" "}
            <strong>severe ≥0.25</strong>. The verdict aggregates: any severe
            → <em>retrain_recommended</em>; ≥3 moderate (no severe) →{" "}
            <em>watch</em>; otherwise <em>stable</em>.
          </>
        }
        recommendations={[
          "Retrain when 'retrain_recommended' fires — your incoming traffic now differs from what the model learned.",
          "Click any feature row to see the bin-level expected vs actual share — pinpoints exactly where the distribution shifted.",
          "Categorical drift on 'vertical' or 'audience_type' often signals a new advertiser segment; flag for shadow-RCT.",
        ]}
        tone={data?.verdict === "retrain_recommended" ? "warning" : "info"}
      />

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
