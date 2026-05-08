"use client";

import { useEffect, useState } from "react";

import { SummaryCard } from "@/components/summary-card";

import {
  type ModelRecord,
  type PredictionRun,
  type SegmentRisk,
  listModels,
  listPredictions,
  scoreCampaign,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const RISK_BADGE: Record<string, string> = {
  severe: "bg-destructive text-white",
  high: "bg-destructive/80 text-white",
  medium: "bg-amber-400 text-black",
  low: "bg-emerald-500 text-white",
  unknown: "bg-muted",
};

type SpecKey = keyof typeof DEFAULT_SPEC;

const DEFAULT_SPEC = {
  campaign_id: "",
  vertical: "ecommerce",
  audience_type: "retargeting",
  funnel_stage: "lower",
  objective: "conversions",
  conversion_optimization: "yes",
  custom_audience: "yes",
  advertiser_platform_experience_months: 18,
  creative_format: "video",
  placement: "feed",
  bid_strategy: "lowest_cost",
  market: "US",
  spend_tier: "high",
  platform: "meta",
  start_date: "2026-06-01",
  end_date: "2026-06-22",
  cost: 75_000,
  test_users: 1_500_000,
  exposed_test_users: 1_200_000,
  clicks: 30_000,
  impressions: 4_000_000,
  conversions: 28_000,
  lcc_7d: 5_000,
} as const;

type SpecInput = Record<string, string | number>;

const NUMERIC_KEYS = new Set([
  "advertiser_platform_experience_months",
  "cost",
  "test_users",
  "exposed_test_users",
  "clicks",
  "impressions",
  "conversions",
  "lcc_7d",
]);

export default function PredictPage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [spec, setSpec] = useState<SpecInput>({ ...DEFAULT_SPEC });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionRun | null>(null);
  const [history, setHistory] = useState<PredictionRun[]>([]);

  useEffect(() => {
    void listModels().then((m) => setModels(m.models));
    void listPredictions(20).then((r) => setHistory(r.runs));
  }, []);

  const set = (key: string, value: string | number) =>
    setSpec((prev) => ({ ...prev, [key]: value }));

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const run = await scoreCampaign(spec, {
        modelId: modelId || undefined,
      });
      setResult(run);
      const fresh = await listPredictions(20);
      setHistory(fresh.runs);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="container py-12">
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 3 · Prediction Workbench
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Forecast a campaign</h1>
      </header>

      <SummaryCard
        title="What you're seeing"
        body={
          <>
            Single-campaign forecast. Fill in the planned spec; the model
            returns predicted ICPD with a 95% confidence band derived from
            the training bootstrap. <strong>Segment risk badges</strong>{" "}
            tell you if this campaign falls in a regime the model has shaky
            evidence for (high penalty in the hold-out-one-level test).
          </>
        }
        recommendations={[
          "If a research-mode watermark appears, treat the prediction as advisory — the donor pool is too small for production.",
          "Wide confidence band → high uncertainty; consider running a real RCT before committing budget.",
          "Severe segment risk → the model has never seen a campaign like this; trust the LCC benchmark or a shadow RCT instead.",
        ]}
      />

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      <form onSubmit={onSubmit} className="mb-8 grid gap-4 md:grid-cols-2">
        <Field label="Model" full>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full rounded-md border bg-background px-2 py-1 text-sm"
          >
            <option value="">Auto (latest production, else research)</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} · {m.version_tag} · {m.status} ({m.id.slice(0, 8)}…)
              </option>
            ))}
          </select>
        </Field>

        {(Object.keys(DEFAULT_SPEC) as SpecKey[]).map((k) => (
          <Field key={k} label={k}>
            <input
              type={NUMERIC_KEYS.has(k) ? "number" : "text"}
              value={String(spec[k] ?? "")}
              onChange={(e) =>
                set(
                  k,
                  NUMERIC_KEYS.has(k)
                    ? Number(e.target.value)
                    : e.target.value,
                )
              }
              className="w-full rounded-md border bg-background px-2 py-1 text-sm"
            />
          </Field>
        ))}

        <div className="md:col-span-2">
          <button
            type="submit"
            disabled={busy}
            className="rounded-md border px-4 py-2 text-sm hover:bg-secondary disabled:opacity-50"
          >
            {busy ? "Scoring…" : "Predict ICPD"}
          </button>
        </div>
      </form>

      {result && <ResultCard run={result} />}

      {history.length > 0 && (
        <section className="mt-10">
          <h2 className="mb-3 text-lg font-medium">Recent predictions</h2>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Run</th>
                  <th className="px-3 py-2 text-left">Campaign</th>
                  <th className="px-3 py-2 text-right">ICPD</th>
                  <th className="px-3 py-2 text-right">95% CI</th>
                  <th className="px-3 py-2 text-left">Worst risk</th>
                  <th className="px-3 py-2 text-left">Status</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r) => (
                  <tr key={r.id} className="border-t">
                    <td className="px-3 py-2 font-mono text-xs">{r.id}</td>
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
                    <td className="px-3 py-2 text-xs">
                      {r.watermark ? (
                        <span className="text-amber-700">research</span>
                      ) : (
                        <span className="text-emerald-700">production</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}

function Field({
  label,
  full,
  children,
}: {
  label: string;
  full?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label
      className={cn(
        "flex flex-col gap-1 text-xs",
        full && "md:col-span-2",
      )}
    >
      <span className="font-mono text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

function ResultCard({ run }: { run: PredictionRun }) {
  const ciWidth =
    run.ci_lower !== null && run.ci_upper !== null
      ? run.ci_upper - run.ci_lower
      : null;

  return (
    <section className="rounded-md border p-6">
      {run.watermark && (
        <div className="mb-4 rounded-md border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-800">
          ⚠ {run.watermark}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-muted-foreground">
            Predicted ICPD
          </p>
          <p className="mt-1 text-4xl font-semibold tabular-nums">
            {run.predicted_icpd.toFixed(4)}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            incremental conversions per dollar
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-muted-foreground">
            95% confidence band
          </p>
          <p className="mt-1 text-2xl font-semibold tabular-nums">
            {run.ci_lower !== null && run.ci_upper !== null
              ? `[${run.ci_lower.toFixed(3)}, ${run.ci_upper.toFixed(3)}]`
              : "—"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {ciWidth !== null
              ? `width ${ciWidth.toFixed(3)} (from training bootstrap)`
              : "no bootstrap CI recorded for this model"}
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-muted-foreground">
            Model
          </p>
          <p className="mt-1 font-mono text-sm">{run.model_version_id}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {run.model_status} · {run.feature_set_version}
          </p>
        </div>
      </div>

      <hr className="my-6" />

      <h3 className="mb-3 text-sm font-medium uppercase tracking-wider text-muted-foreground">
        Segment extrapolation risk
      </h3>
      {run.segment_risks.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No hold-out-one-level results found for this campaign&rsquo;s segment
          levels. Run the extrapolation test on the{" "}
          <a className="underline" href="/models">
            Model Trust
          </a>{" "}
          page to surface risk badges here.
        </p>
      ) : (
        <ul className="grid gap-2 md:grid-cols-2">
          {run.segment_risks.map((r) => (
            <RiskBadge key={`${r.segmentation_var}|${r.level}`} risk={r} />
          ))}
        </ul>
      )}
    </section>
  );
}

function RiskBadge({ risk }: { risk: SegmentRisk }) {
  return (
    <li className="flex items-center justify-between rounded-md border p-3">
      <div>
        <p className="font-mono text-xs text-muted-foreground">
          {risk.segmentation_var}
        </p>
        <p className="text-sm">{risk.level}</p>
      </div>
      <div className="text-right">
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-xs",
            RISK_BADGE[risk.risk] ?? "bg-muted",
          )}
        >
          {risk.risk}
        </span>
        {risk.penalty_pp !== null && Number.isFinite(risk.penalty_pp) && (
          <p className="mt-1 text-xs text-muted-foreground tabular-nums">
            {risk.penalty_pp.toFixed(1)}pp penalty
          </p>
        )}
      </div>
    </li>
  );
}
