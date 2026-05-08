"use client";

import { Suspense, useCallback, useEffect, useState } from "react";

import { isDemoMode } from "@/lib/demo-mode";
import { useUploadId } from "@/lib/use-upload-id";
import { SummaryCard } from "@/components/summary-card";

import {
  type ActionBand,
  type DecisionResponse,
  type ModelRecord,
  listModels,
  recommendDecisions,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const ACTION_BADGE: Record<ActionBand, string> = {
  promote: "bg-emerald-500 text-white",
  hold: "bg-secondary text-foreground",
  deprioritize: "bg-amber-400 text-black",
  block: "bg-destructive text-white",
};

const RISK_BADGE: Record<string, string> = {
  severe: "bg-destructive text-white",
  high: "bg-destructive/80 text-white",
  medium: "bg-amber-400 text-black",
  low: "bg-emerald-500 text-white",
  unknown: "bg-muted",
};

type RiskFloor = "low" | "unknown" | "medium" | "high" | "severe";

function DecisionsInner() {
  const { uploadId: resolvedUploadId } = useUploadId();
  const [uploadId, setUploadId] = useState(resolvedUploadId ?? "");
  const [modelId, setModelId] = useState("");
  const [riskFloor, setRiskFloor] = useState<RiskFloor>("low");
  const [onlyNonRct, setOnlyNonRct] = useState(true);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [data, setData] = useState<DecisionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void listModels().then((m) => setModels(m.models));
  }, []);

  const onRun = useCallback(async () => {
    if (!uploadId.trim()) {
      setError("Enter an upload_id");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await recommendDecisions({
        uploadId: uploadId.trim(),
        modelId: modelId || undefined,
        riskFloor,
        onlyNonRct,
      });
      setData(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [uploadId, modelId, riskFloor, onlyNonRct]);

  // Demo mode: auto-trigger so the page renders populated on first visit.
  useEffect(() => {
    if (isDemoMode() && uploadId && !data && !busy) void onRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId]);

  return (
    <main className="container py-12">
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 3 · Decision Recommendations
        </p>
        <h1 className="mt-2 text-3xl font-semibold">
          Rank a portfolio with risk gates
        </h1>
      </header>

      <SummaryCard
        title="What you're seeing"
        body={
          <>
            Each campaign gets one of four advisory actions, sorted by
            risk-adjusted ICPD (CI lower bound × risk multiplier).
            <strong> promote</strong> = top-tercile risk-adjusted ICPD with
            low/unknown risk. <strong>hold</strong> = below tercile or medium
            risk. <strong>deprioritize</strong> = high risk OR CI-lower ≤ 0.
            <strong> block</strong> = severe extrapolation risk (auto-sinks
            to bottom). The <em>projected lift</em> compares mean ICPD across
            all candidates vs. the campaigns we&rsquo;d keep.
          </>
        }
        recommendations={[
          "Use the risk floor to control how aggressive 'promote' can be — set 'low' for the strictest gate.",
          "Lift figures vanish for research-mode models — Phase 4.2 simulator hard-blocks them anyway.",
          "Read the per-row rationale before acting; the recommendation isn't a command, it's a starting point.",
        ]}
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
        <label className="flex flex-col gap-1 text-xs">
          <span className="font-mono text-muted-foreground">risk floor</span>
          <select
            value={riskFloor}
            onChange={(e) => setRiskFloor(e.target.value as RiskFloor)}
            className="w-full rounded-md border bg-background px-2 py-1 text-sm"
          >
            <option value="low">low — strictest, only low/unknown can promote</option>
            <option value="medium">medium — allow medium-risk promotions</option>
            <option value="high">high — allow high-risk promotions</option>
          </select>
        </label>
        <div className="md:col-span-3 flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={onlyNonRct}
              onChange={(e) => setOnlyNonRct(e.target.checked)}
            />
            Score non-RCT rows only
          </label>
          <button
            type="button"
            onClick={() => void onRun()}
            disabled={busy}
            className="rounded-md border px-4 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          >
            {busy ? "Ranking…" : "Run recommendations"}
          </button>
        </div>
      </section>

      {data && (
        <>
          {data.is_research_model && (
            <div className="mb-4 rounded-md border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-800">
              ⚠ Recommendations from a research-mode model are advisory only.
              Projected portfolio lift is suppressed; the Decision Simulator
              (Phase 4.2) will hard-block research models entirely.
            </div>
          )}

          <section className="mb-6 grid gap-4 md:grid-cols-4">
            {(Object.entries(data.action_counts) as [ActionBand, number][]).map(
              ([action, count]) => (
                <div key={action} className="rounded-md border p-4">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    {action}
                  </p>
                  <p className="mt-1 text-3xl font-semibold tabular-nums">
                    {count}
                  </p>
                  <span
                    className={cn(
                      "mt-2 inline-block rounded-full px-2 py-0.5 text-xs",
                      ACTION_BADGE[action],
                    )}
                  >
                    {action}
                  </span>
                </div>
              ),
            )}
          </section>

          {data.projected_lift && (
            <section className="mb-6 rounded-md border p-5">
              <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                Projected portfolio lift
              </h2>
              <div className="grid gap-4 md:grid-cols-3">
                <Card
                  label="Naive mean ICPD"
                  value={data.projected_lift.naive_portfolio_icpd.toFixed(4)}
                  hint={`across ${data.projected_lift.n_total} candidates`}
                />
                <Card
                  label="Advised mean ICPD"
                  value={data.projected_lift.advised_portfolio_icpd.toFixed(4)}
                  hint={`promote+hold only · ${data.projected_lift.n_followed}/${data.projected_lift.n_total}`}
                />
                <Card
                  label="Lift (percentage points)"
                  value={`${data.projected_lift.lift_pp.toFixed(2)}pp`}
                  valueClass={
                    data.projected_lift.lift_pp >= 0
                      ? "text-emerald-700"
                      : "text-destructive"
                  }
                />
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                {data.projected_lift.rationale}
              </p>
            </section>
          )}

          <section>
            <h2 className="mb-3 text-lg font-medium">Ranked recommendations</h2>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-secondary">
                  <tr>
                    <th className="px-3 py-2 text-right">Rank</th>
                    <th className="px-3 py-2 text-left">Campaign</th>
                    <th className="px-3 py-2 text-right">Predicted ICPD</th>
                    <th className="px-3 py-2 text-right">95% CI</th>
                    <th className="px-3 py-2 text-right">Risk-adj. score</th>
                    <th className="px-3 py-2 text-left">Worst risk</th>
                    <th className="px-3 py-2 text-left">Action</th>
                    <th className="px-3 py-2 text-left">Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recommendations.map((r) => (
                    <tr key={r.run_id} className="border-t">
                      <td className="px-3 py-2 text-right tabular-nums font-mono text-xs">
                        {r.rank}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {r.campaign_id}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.predicted_icpd.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {r.ci_lower !== null && r.ci_upper !== null
                          ? `[${r.ci_lower.toFixed(3)}, ${r.ci_upper.toFixed(3)}]`
                          : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.risk_adjusted_score !== null
                          ? r.risk_adjusted_score.toFixed(4)
                          : "—"}
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={cn(
                            "rounded-full px-2 py-0.5 text-xs",
                            RISK_BADGE[r.worst_risk] ?? "bg-muted",
                          )}
                        >
                          {r.worst_risk}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={cn(
                            "rounded-full px-2 py-0.5 text-xs",
                            ACTION_BADGE[r.action],
                          )}
                        >
                          {r.action}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-muted-foreground">
                        {r.rationale}
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
      <p className={cn("mt-1 text-2xl font-semibold tabular-nums", valueClass)}>
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export default function DecisionsPage() {
  return (
    <Suspense
      fallback={
        <main className="container py-12">
          <p className="text-muted-foreground">Loading decisions…</p>
        </main>
      }
    >
      <DecisionsInner />
    </Suspense>
  );
}
