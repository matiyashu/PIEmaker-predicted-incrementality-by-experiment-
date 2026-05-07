"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import {
  type ModelRecord,
  type PortfolioResponse,
  type PredictionRun,
  listModels,
  scorePortfolio,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const RISK_BADGE: Record<string, string> = {
  severe: "bg-destructive text-white",
  high: "bg-destructive/80 text-white",
  medium: "bg-amber-400 text-black",
  low: "bg-emerald-500 text-white",
  unknown: "bg-muted",
};

type SortKey = "icpd_desc" | "icpd_asc" | "risk_desc";

function PortfolioInner() {
  const params = useSearchParams();
  const initialUploadId = params.get("upload_id") ?? "";

  const [uploadId, setUploadId] = useState(initialUploadId);
  const [modelId, setModelId] = useState("");
  const [onlyNonRct, setOnlyNonRct] = useState(true);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("icpd_desc");
  const [riskFilter, setRiskFilter] = useState<string>("");

  useEffect(() => {
    void listModels().then((m) => setModels(m.models));
  }, []);

  const onRun = async () => {
    if (!uploadId.trim()) {
      setError("Enter an upload_id");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await scorePortfolio({
        uploadId: uploadId.trim(),
        modelId: modelId || undefined,
        onlyNonRct,
      });
      setData(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const sortedRuns = useMemo<PredictionRun[]>(() => {
    if (!data) return [];
    const filtered = riskFilter
      ? data.runs.filter(
          (r) => (r.worst_segment_risk?.risk ?? "unknown") === riskFilter,
        )
      : data.runs;
    const order = [...filtered];
    order.sort((a, b) => {
      if (sortKey === "icpd_desc") return b.predicted_icpd - a.predicted_icpd;
      if (sortKey === "icpd_asc") return a.predicted_icpd - b.predicted_icpd;
      const rank = { severe: 4, high: 3, medium: 2, low: 1, unknown: 0 } as const;
      const ra = rank[a.worst_segment_risk?.risk ?? "unknown"];
      const rb = rank[b.worst_segment_risk?.risk ?? "unknown"];
      return rb - ra;
    });
    return order;
  }, [data, sortKey, riskFilter]);

  return (
    <main className="container py-12">
      <header className="mb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 3 · Portfolio Scoring
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Score a media plan</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Bulk-score every non-RCT row in an upload against a registered
          model. One model selection per call; aggregates surface mean,
          median, p10/p90, and segment-risk counts so you can see the shape
          of the portfolio at a glance.
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
        <label className="flex flex-col gap-1 text-xs">
          <span className="font-mono text-muted-foreground">model</span>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full rounded-md border bg-background px-2 py-1 text-sm"
          >
            <option value="">Auto (latest production, else research)</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} · {m.status} ({m.id.slice(0, 8)}…)
              </option>
            ))}
          </select>
        </label>
        <div className="md:col-span-3 flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={onlyNonRct}
              onChange={(e) => setOnlyNonRct(e.target.checked)}
            />
            Score non-RCT rows only (skip is_rct=1)
          </label>
          <button
            type="button"
            onClick={() => void onRun()}
            disabled={busy}
            className="rounded-md border px-4 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          >
            {busy ? "Scoring portfolio…" : "Score portfolio"}
          </button>
        </div>
      </section>

      {data && (
        <>
          {data.watermark && (
            <div className="mb-4 rounded-md border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-800">
              ⚠ {data.watermark}
            </div>
          )}

          <section className="mb-6 grid gap-4 md:grid-cols-4">
            <Card label="Campaigns scored" value={String(data.aggregates.n)} />
            <Card
              label="Mean ICPD"
              value={data.aggregates.mean_icpd.toFixed(4)}
              hint={`σ ${data.aggregates.stdev_icpd.toFixed(4)}`}
            />
            <Card
              label="Median ICPD"
              value={data.aggregates.median_icpd.toFixed(4)}
              hint={`p10 ${data.aggregates.p10_icpd.toFixed(3)} · p90 ${data.aggregates.p90_icpd.toFixed(3)}`}
            />
            <Card
              label="Worst segment"
              value={data.worst_segment_risk?.risk ?? "—"}
              hint={
                data.worst_segment_risk
                  ? `${data.worst_segment_risk.segmentation_var} · ${data.worst_segment_risk.level} · ${data.worst_segment_risk.penalty_pp?.toFixed(1) ?? ""}pp`
                  : "no hold-out results"
              }
              valueClass={
                RISK_BADGE[data.worst_segment_risk?.risk ?? "unknown"] ?? ""
              }
            />
          </section>

          <section className="mb-6 rounded-md border p-4">
            <p className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">
              Risk distribution
            </p>
            <div className="flex flex-wrap gap-2">
              {(Object.entries(data.aggregates.risk_counts) as [
                keyof typeof RISK_BADGE,
                number,
              ][]).map(([risk, count]) => (
                <span
                  key={risk}
                  className={cn(
                    "rounded-full px-3 py-1 text-xs font-medium",
                    RISK_BADGE[risk] ?? "bg-muted",
                  )}
                >
                  {risk}: {count}
                </span>
              ))}
            </div>
          </section>

          <section>
            <div className="mb-3 flex flex-wrap items-center gap-3">
              <h2 className="text-lg font-medium">Per-campaign predictions</h2>
              <label className="flex items-center gap-2 text-xs">
                Sort
                <select
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                  className="rounded-md border bg-background px-2 py-1"
                >
                  <option value="icpd_desc">ICPD (high → low)</option>
                  <option value="icpd_asc">ICPD (low → high)</option>
                  <option value="risk_desc">Risk (worst first)</option>
                </select>
              </label>
              <label className="flex items-center gap-2 text-xs">
                Risk filter
                <select
                  value={riskFilter}
                  onChange={(e) => setRiskFilter(e.target.value)}
                  className="rounded-md border bg-background px-2 py-1"
                >
                  <option value="">all</option>
                  <option value="severe">severe</option>
                  <option value="high">high</option>
                  <option value="medium">medium</option>
                  <option value="low">low</option>
                  <option value="unknown">unknown</option>
                </select>
              </label>
              <span className="text-xs text-muted-foreground">
                {sortedRuns.length} of {data.runs.length} shown
              </span>
            </div>

            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-secondary">
                  <tr>
                    <th className="px-3 py-2 text-left">Campaign</th>
                    <th className="px-3 py-2 text-left">Vertical</th>
                    <th className="px-3 py-2 text-left">Audience</th>
                    <th className="px-3 py-2 text-right">Predicted ICPD</th>
                    <th className="px-3 py-2 text-right">95% CI</th>
                    <th className="px-3 py-2 text-left">Worst risk</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedRuns.map((r) => (
                    <tr key={r.id} className="border-t">
                      <td className="px-3 py-2 font-mono text-xs">
                        {r.campaign_id}
                      </td>
                      <td className="px-3 py-2">
                        {String(r.spec.vertical ?? "")}
                      </td>
                      <td className="px-3 py-2">
                        {String(r.spec.audience_type ?? "")}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.predicted_icpd.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {r.ci_lower !== null && r.ci_upper !== null
                          ? `[${r.ci_lower.toFixed(3)}, ${r.ci_upper.toFixed(3)}]`
                          : "—"}
                      </td>
                      <td className="px-3 py-2">
                        {r.worst_segment_risk ? (
                          <span
                            className={cn(
                              "rounded-full px-2 py-0.5 text-xs",
                              RISK_BADGE[r.worst_segment_risk.risk] ??
                                "bg-muted",
                            )}
                          >
                            {r.worst_segment_risk.risk}
                          </span>
                        ) : (
                          <span className="text-xs text-muted-foreground">
                            —
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
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
  valueClass,
}: {
  label: string;
  value: string;
  hint?: string;
  valueClass?: string;
}) {
  return (
    <div className="rounded-md border p-4">
      <p className="text-xs uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "mt-1 text-2xl font-semibold tabular-nums",
          valueClass &&
            "inline-block rounded-md px-2 py-0.5 text-base font-medium",
          valueClass,
        )}
      >
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export default function PortfolioPage() {
  return (
    <Suspense
      fallback={
        <main className="container py-12">
          <p className="text-muted-foreground">Loading portfolio…</p>
        </main>
      }
    >
      <PortfolioInner />
    </Suspense>
  );
}
