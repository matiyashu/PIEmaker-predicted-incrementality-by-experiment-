"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SummaryCard } from "@/components/summary-card";

import {
  type HoldoutResponse,
  type ModelMetric,
  type ModelRecord,
  type TrainResponse,
  SEGMENTATION_VARS,
  fetchModelMetrics,
  listModels,
  promoteModel,
  runHoldout,
  trainModel,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS_BADGE: Record<string, string> = {
  research: "bg-amber-400 text-black",
  production: "bg-emerald-500 text-white",
};

const RISK_BADGE: Record<string, string> = {
  severe: "bg-destructive text-white",
  high: "bg-destructive/80 text-white",
  medium: "bg-amber-400 text-black",
  low: "bg-emerald-500 text-white",
  unknown: "bg-muted",
};

export default function ModelsPage() {
  const [models, setModels] = useState<ModelRecord[] | null>(null);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ModelMetric[] | null>(null);
  const [holdoutVar, setHoldoutVar] =
    useState<(typeof SEGMENTATION_VARS)[number]>("vertical");
  const [holdoutBusy, setHoldoutBusy] = useState(false);
  const [holdout, setHoldout] = useState<HoldoutResponse | null>(null);

  const refresh = useCallback(async () => {
    try {
      const m = await listModels();
      setModels(m.models);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!selectedId) {
      setMetrics(null);
      return;
    }
    fetchModelMetrics(selectedId)
      .then((m) => setMetrics(m.metrics))
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, [selectedId]);

  const onTrain = async () => {
    setTraining(true);
    setError(null);
    try {
      const result = await trainModel({ nBootstrap: 100 });
      setTrainResult(result);
      setSelectedId(result.model.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setTraining(false);
    }
  };

  const onPromote = async (id: string) => {
    setError(null);
    try {
      await promoteModel(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onRunHoldout = async () => {
    setHoldoutBusy(true);
    setError(null);
    try {
      const r = await runHoldout(holdoutVar, { nIterations: 20 });
      setHoldout(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setHoldoutBusy(false);
    }
  };

  return (
    <main className="container py-12">
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 2 · Model Trust Dashboard
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Train, evaluate, promote</h1>
      </header>

      <SummaryCard
        title="What you're seeing"
        body={
          <>
            The trust layer that ships before any prediction UI. Train a
            model, then read the diagnostics: <strong>weighted R²</strong>{" "}
            (cost-weighted goodness of fit), <strong>bootstrap CI</strong>{" "}
            (uncertainty band on R²), <strong>R² ceiling</strong> (theoretical
            upper bound from outcome noise — paper §5.2), the{" "}
            <strong>ablation chart</strong> (paper Figure 2: which features
            carry signal), and <strong>hold-out-one-level</strong>{" "}
            extrapolation per segmentation variable (paper Table 1).
          </>
        }
        recommendations={[
          "Train a model only after the donor pool reaches research-mode (≥200 RCTs); production-mode (≥400) removes the watermark.",
          "Run hold-out-one-level on every segmentation var before promoting to production — flagged as severe = retrain on a refreshed pool.",
          "If R² ≪ R² ceiling, the model is underfit; if R² ≈ R² ceiling, you're near the noise floor and more features won't help.",
        ]}
      />

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      <section className="mb-8 rounded-md border p-5">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-lg font-medium">Train a new model</h2>
          <button
            type="button"
            onClick={() => void onTrain()}
            disabled={training}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          >
            {training ? "Training…" : "Train PIE Random Forest"}
          </button>
          <span className="text-xs text-muted-foreground">
            Requires donor pool ≥ 200 RCTs and built feature_store rows.
          </span>
        </div>

        {trainResult && (
          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <Card label="Weighted R²"
                  value={trainResult.weighted_r_squared.toFixed(3)}
                  hint={`n=${trainResult.n_observations}`} />
            <Card
              label="Bootstrap 95% CI"
              value={`[${trainResult.bootstrap.p025.toFixed(3)}, ${trainResult.bootstrap.p975.toFixed(3)}]`}
              hint={`mean ${trainResult.bootstrap.mean.toFixed(3)}`}
            />
            <Card
              label="R² ceiling"
              value={trainResult.r_squared_ceiling.toFixed(3)}
              hint="upper bound from outcome noise"
            />
            <Card
              label="LCC slope"
              value={fmt(trainResult.lcc_diagnostics.slope)}
              hint="OLS ICPD ~ LCC-7d/$"
            />
            <Card
              label="LCC Spearman ρ"
              value={fmt(trainResult.lcc_diagnostics.spearman_rho)}
              hint="rank correlation"
            />
            <Card
              label="Donor pool"
              value={`${trainResult.donor_pool_status.n_admitted}`}
              hint={trainResult.donor_pool_status.band.replace("_", " ")}
            />
            <div className="md:col-span-3">
              <h3 className="mt-2 mb-2 text-sm font-medium uppercase tracking-wider text-muted-foreground">
                Feature ablation (paper Fig. 2)
              </h3>
              <AblationChart rows={trainResult.ablation} />
            </div>
          </div>
        )}
      </section>

      <section className="mb-8">
        <h2 className="mb-3 text-lg font-medium">Registered models</h2>
        {!models ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : models.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No models yet. Train one above.
          </p>
        ) : (
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Pick</th>
                  <th className="px-3 py-2 text-left">ID</th>
                  <th className="px-3 py-2 text-left">Name</th>
                  <th className="px-3 py-2 text-left">Status</th>
                  <th className="px-3 py-2 text-left">Algorithm</th>
                  <th className="px-3 py-2 text-right">Pool size</th>
                  <th className="px-3 py-2 text-left">Created</th>
                  <th className="px-3 py-2 text-left">Action</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr
                    key={m.id}
                    className={cn(
                      "border-t cursor-pointer hover:bg-secondary/50",
                      selectedId === m.id && "bg-secondary/40",
                    )}
                    onClick={() => setSelectedId(m.id)}
                  >
                    <td className="px-3 py-2">
                      <input
                        type="radio"
                        checked={selectedId === m.id}
                        readOnly
                      />
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{m.id}</td>
                    <td className="px-3 py-2">{m.name}</td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-xs",
                          STATUS_BADGE[m.status] ?? "bg-muted",
                        )}
                      >
                        {m.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{m.algorithm}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {m.training_donor_pool_size}
                    </td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">
                      {m.created_at.slice(0, 19).replace("T", " ")}
                    </td>
                    <td className="px-3 py-2">
                      {m.status === "research" && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            void onPromote(m.id);
                          }}
                          className="text-xs underline"
                        >
                          Promote
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selectedId && metrics && (
        <section className="mb-8">
          <h2 className="mb-3 text-lg font-medium">
            Metrics for{" "}
            <code className="font-mono text-sm">{selectedId}</code>
          </h2>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Metric</th>
                  <th className="px-3 py-2 text-right">Value</th>
                  <th className="px-3 py-2 text-right">95% CI</th>
                  <th className="px-3 py-2 text-left">Segment</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m.id} className="border-t">
                    <td className="px-3 py-2 font-mono text-xs">
                      {m.metric_type}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {m.value.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                      {m.ci_lower !== null && m.ci_upper !== null
                        ? `[${m.ci_lower.toFixed(3)}, ${m.ci_upper.toFixed(3)}]`
                        : ""}
                    </td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">
                      {m.segment ? JSON.stringify(m.segment) : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="mb-8 rounded-md border p-5">
        <h2 className="mb-3 text-lg font-medium">
          Hold-out-one-level extrapolation
        </h2>
        <p className="mb-3 text-sm text-muted-foreground">
          For each level of a segmentation variable, compares within-level R²
          (interpolation) to extrapolation R² trained on out-of-level data.
          Penalty ≥25pp = severe, ≥15pp = high, ≥5pp = medium.
        </p>
        <div className="mb-3 flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            Segmentation variable
            <select
              value={holdoutVar}
              onChange={(e) =>
                setHoldoutVar(
                  e.target.value as (typeof SEGMENTATION_VARS)[number],
                )
              }
              className="rounded-md border bg-background px-2 py-1 text-sm"
            >
              {SEGMENTATION_VARS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void onRunHoldout()}
            disabled={holdoutBusy}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
          >
            {holdoutBusy ? "Running…" : "Run extrapolation test"}
          </button>
        </div>

        {holdout && (
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Level</th>
                  <th className="px-3 py-2 text-right">Within R²</th>
                  <th className="px-3 py-2 text-right">Extrap R²</th>
                  <th className="px-3 py-2 text-right">Penalty (pp)</th>
                  <th className="px-3 py-2 text-right">Iter</th>
                  <th className="px-3 py-2 text-left">Risk</th>
                </tr>
              </thead>
              <tbody>
                {holdout.results.map((r) => (
                  <tr key={r.level} className="border-t">
                    <td className="px-3 py-2 font-mono text-xs">{r.level}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {r.within_r2_median.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {r.extrapolation_r2_median.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {Number.isFinite(r.penalty_pp)
                        ? r.penalty_pp.toFixed(1)
                        : "—"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {r.n_iterations}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-xs",
                          RISK_BADGE[r.risk] ?? "bg-muted",
                        )}
                      >
                        {r.risk}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
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

function AblationChart({
  rows,
}: {
  rows: { spec: string; weighted_r2: number }[];
}) {
  const data = rows.map((r) => ({
    spec: r.spec,
    weighted_r2: Number.isFinite(r.weighted_r2) ? r.weighted_r2 : 0,
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="spec"
          tick={{ fontSize: 10 }}
          interval={0}
          angle={-15}
          textAnchor="end"
          height={60}
        />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(v: number) => v.toFixed(3)}
          labelFormatter={(l) => `Spec: ${l}`}
        />
        <Bar dataKey="weighted_r2" fill="#3b82f6" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function fmt(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return v.toFixed(3);
}
