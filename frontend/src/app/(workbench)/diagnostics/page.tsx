"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, AlertTriangle, GitCompare, LineChart as LineIcon, Microscope } from "lucide-react";

import {
  type AdvertiserBootstrapResponse,
  type AdvertiserCvResponse,
  type CalibrationResponse,
  type HoldoutDistributionsResponse,
  type SampleSizeCurveResponse,
  type TrainResponse,
  fetchAdvertiserBootstrap,
  fetchAdvertiserCv,
  fetchCalibration,
  fetchHoldoutDistributions,
  fetchSampleSizeCurve,
  trainModel,
} from "@/lib/api";
import { SummaryCard } from "@/components/summary-card";
import { cn } from "@/lib/utils";

const RISK_COLOR: Record<string, string> = {
  severe: "#ef4444",
  high: "#f97316",
  medium: "#f59e0b",
  low: "#10b981",
  unknown: "#94a3b8",
};

type SegVar = "vertical" | "audience_type" | "advertiser_size" | "campaign_year";
const SEG_VARS: SegVar[] = [
  "vertical",
  "audience_type",
  "advertiser_size",
  "campaign_year",
];

export default function DiagnosticsPage() {
  const [train, setTrain] = useState<TrainResponse | null>(null);
  const [sampleSize, setSampleSize] = useState<SampleSizeCurveResponse | null>(null);
  const [advCv, setAdvCv] = useState<AdvertiserCvResponse | null>(null);
  const [advBoot, setAdvBoot] = useState<AdvertiserBootstrapResponse | null>(null);
  const [holdoutDists, setHoldoutDists] = useState<HoldoutDistributionsResponse | null>(null);
  const [calSegVar, setCalSegVar] = useState<SegVar>("vertical");
  const [calibration, setCalibration] = useState<CalibrationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [t, ss, ac, ab, hd] = await Promise.all([
        trainModel({ nBootstrap: 100 }),
        fetchSampleSizeCurve(),
        fetchAdvertiserCv(),
        fetchAdvertiserBootstrap(),
        fetchHoldoutDistributions(),
      ]);
      setTrain(t);
      setSampleSize(ss);
      setAdvCv(ac);
      setAdvBoot(ab);
      setHoldoutDists(hd);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    fetchCalibration(calSegVar).then(setCalibration).catch((err) =>
      setError(err instanceof Error ? err.message : String(err)),
    );
  }, [calSegVar]);

  return (
    <main className="container py-12">
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          V.4 · Paper-faithful diagnostics
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Trust the forecast</h1>
      </header>

      <SummaryCard
        title="What's on this page"
        body={
          <>
            Six diagnostics that together answer &ldquo;should I believe this
            model?&rdquo;. Headline R² comes from <strong>out-of-fold</strong>{" "}
            predictions (paper §5.2); the ablation chart has{" "}
            <strong>bootstrap error bars</strong> per spec (Figure 2); the{" "}
            <strong>pool-size curve</strong> tells you how much donor pool
            you need (Figure 6); the <strong>hold-out boxes</strong> show
            where the model extrapolates (Table 1); the{" "}
            <strong>LCC calibration</strong> table shows where the naive
            benchmark fails by segment (§5.4); and the{" "}
            <strong>advertiser CV</strong> isolates cold-start risk (§5.3).
          </>
        }
        recommendations={[
          "If R² is close to the label-noise ceiling, the model is at the noise floor — more features won't help.",
          "If the pool-size curve is still climbing at your current pool, more RCTs will pay off; if it's plateaued, invest in coverage instead.",
          "Any 'severe' hold-out level (≥25pp penalty) → run a shadow RCT in that segment before serving production forecasts.",
        ]}
      />

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      <section className="mb-6 grid gap-4 lg:grid-cols-2">
        <Card
          title="Feature ablation (paper Fig 2)"
          subtitle="Weighted R² per spec on OOF predictions, 95% bootstrap CI"
          icon={Microscope}
        >
          {train?.ablation?.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={train.ablation.map((a) => ({
                  spec: a.spec,
                  weighted_r2: a.weighted_r2,
                  errorLow: Math.max(0, a.weighted_r2 - ((a as unknown as { ci_lower?: number }).ci_lower ?? a.weighted_r2)),
                  errorHigh: Math.max(0, ((a as unknown as { ci_upper?: number }).ci_upper ?? a.weighted_r2) - a.weighted_r2),
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="spec" tick={{ fontSize: 9 }} interval={0} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="weighted_r2" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  <ErrorBar dataKey="errorHigh" width={4} stroke="#1e40af" direction="y" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty msg="Ablation data not available yet." />
          )}
        </Card>

        <Card
          title="Pool-size curve (paper Fig 6)"
          subtitle="Weighted R² median ± 95% CI band vs donor-pool size"
          icon={LineIcon}
        >
          {sampleSize?.points?.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={sampleSize.points}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="pool_size"
                  type="number"
                  scale="log"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 11 }}
                />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="weighted_r2_median" stroke="#3b82f6" name="median R²" />
                <Line type="monotone" dataKey="weighted_r2_p025" stroke="#94a3b8" strokeDasharray="3 3" name="p2.5" />
                <Line type="monotone" dataKey="weighted_r2_p975" stroke="#94a3b8" strokeDasharray="3 3" name="p97.5" />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <Empty msg="Pool-size curve not available yet." />
          )}
        </Card>
      </section>

      <section className="mb-6 grid gap-4 lg:grid-cols-2">
        <Card
          title="Hold-out-one-level (paper Tbl 1)"
          subtitle="p10–p90 R² band per level; gap = extrapolation penalty"
          icon={GitCompare}
        >
          {holdoutDists?.distributions?.length ? (
            <div className="space-y-2 max-h-[260px] overflow-y-auto">
              {holdoutDists.distributions.map((d) => (
                <HoldoutBar key={d.id} d={d} />
              ))}
            </div>
          ) : (
            <Empty msg="No hold-out distributions yet." />
          )}
        </Card>

        <Card
          title="Cold-start advertiser cohort"
          subtitle="Existing-advertiser R² vs new-advertiser R²"
          icon={AlertTriangle}
        >
          {advCv ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                {advCv.cohorts.map((c) => (
                  <div key={c.cohort} className="rounded-md border bg-card p-3">
                    <p className="text-xs uppercase tracking-wider text-muted-foreground">
                      {c.cohort}
                    </p>
                    <p className="mt-1 text-2xl font-semibold tabular-nums">
                      {c.weighted_r2.toFixed(3)}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {c.n} runs · {c.n_advertisers} advertisers
                    </p>
                  </div>
                ))}
              </div>
              <div
                className={cn(
                  "rounded-md border p-3 text-sm",
                  advCv.cohort_gap_pp >= 10
                    ? "border-destructive/40 bg-destructive/5 text-destructive"
                    : "border-amber-300 bg-amber-50/60 text-amber-800",
                )}
              >
                <strong>{advCv.cohort_gap_pp.toFixed(1)}pp</strong>{" "}
                cold-start gap. {advCv.cohort_gap_pp >= 10
                  ? "Material — pre-flag forecasts for new advertisers."
                  : "Tolerable; flag only when ≥10pp."}
              </div>
              {advBoot && (
                <div className="rounded-md border bg-card p-3">
                  <p className="mb-1 text-xs uppercase tracking-wider text-muted-foreground">
                    Cluster bootstrap (over advertisers)
                  </p>
                  <p className="text-sm">
                    <span className="font-semibold tabular-nums">
                      {advBoot.mean.toFixed(3)}
                    </span>{" "}
                    <span className="text-muted-foreground">
                      [{advBoot.p025.toFixed(3)}, {advBoot.p975.toFixed(3)}]
                    </span>{" "}
                    <span className="text-xs text-muted-foreground">
                      · n_draws={advBoot.n_draws}, advertisers={advBoot.n_advertisers}
                    </span>
                  </p>
                </div>
              )}
            </div>
          ) : (
            <Empty msg="Advertiser CV not available yet." />
          )}
        </Card>
      </section>

      <section className="mb-6">
        <Card
          title="LCC calibration by segment (paper §5.4)"
          subtitle="Where the naive LCC-7d benchmark fails — by segment level"
          icon={Activity}
        >
          <div className="mb-3 flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Segmentation:</span>
            {SEG_VARS.map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setCalSegVar(v)}
                className={cn(
                  "rounded-full border px-3 py-1 transition-colors",
                  calSegVar === v
                    ? "border-primary bg-primary text-primary-foreground"
                    : "hover:bg-secondary",
                )}
              >
                {v}
              </button>
            ))}
          </div>
          {calibration?.results?.length ? (
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-secondary text-xs uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left">Level</th>
                    <th className="px-3 py-2 text-right">n</th>
                    <th className="px-3 py-2 text-right">Bias ratio</th>
                    <th className="px-3 py-2 text-right">OLS slope</th>
                    <th className="px-3 py-2 text-right">Spearman ρ</th>
                    <th className="px-3 py-2 text-right">Raw LCC R²</th>
                    <th className="px-3 py-2 text-right">Residual p10 / p90</th>
                  </tr>
                </thead>
                <tbody>
                  {calibration.results.map((r) => (
                    <tr key={r.level} className="border-t">
                      <td className="px-3 py-2 font-medium">{r.level}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{r.n}</td>
                      <td
                        className={cn(
                          "px-3 py-2 text-right tabular-nums",
                          (r.bias_ratio ?? 1) > 1.25 ? "text-destructive" : "",
                        )}
                      >
                        {r.bias_ratio?.toFixed(3) ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.ols_slope?.toFixed(3) ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.spearman_rho?.toFixed(3) ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.raw_lcc_r2?.toFixed(3) ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {r.residual_p10.toFixed(3)} / {r.residual_p90.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty msg="No calibration rows for this segmentation." />
          )}
          <p className="mt-2 text-xs text-muted-foreground">
            Bias ratio &gt; 1 means LCC overstates ICPD in that segment.
            Highlighted rows (&gt;1.25) are the strongest candidates for
            shadow-RCT investment.
          </p>
        </Card>
      </section>

      <section className="mb-6">
        <Card
          title="Random-forest feature importance"
          subtitle="Built-in importance from the trained model (proxy for permutation importance)"
          icon={Microscope}
        >
          {train?.model?.concept_drift_baseline?.feature_importances?.length ? (
            <ImportanceBar
              importances={train.model.concept_drift_baseline.feature_importances}
            />
          ) : (
            <Empty msg="No feature importance data yet — train a model first." />
          )}
        </Card>
      </section>
    </main>
  );
}

// --- helpers ----------------------------------------------------------------

function Card({
  title,
  subtitle,
  icon: Icon,
  children,
}: {
  title: string;
  subtitle?: string;
  icon?: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-card p-5 shadow-sm">
      <div className="mb-3 flex items-start gap-2">
        {Icon && (
          <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-primary/10 text-primary">
            <Icon className="h-3.5 w-3.5" />
          </span>
        )}
        <div className="flex-1">
          <p className="font-medium">{title}</p>
          {subtitle && (
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
      </div>
      {children}
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="grid h-[200px] place-items-center text-sm text-muted-foreground">
      {msg}
    </div>
  );
}

function HoldoutBar({
  d,
}: {
  d: {
    segmentation_var: string;
    level: string;
    within_r2_p10: number;
    within_r2_p90: number;
    extrapolation_r2_p10: number;
    extrapolation_r2_p90: number;
    within_r2_median: number;
    extrapolation_r2_median: number;
    penalty_pp: number;
    risk: string;
  };
}) {
  // Two horizontal bars per row: within (green-ish) and extrap (amber/red).
  const max = 1;
  const withinW = Math.max(0, d.within_r2_median) / max;
  const extrapW = Math.max(0, d.extrapolation_r2_median) / max;
  return (
    <div className="rounded-md border bg-background p-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-mono">
          {d.segmentation_var} = {d.level}
        </span>
        <span
          className="rounded-full px-2 py-0.5 text-[10px] font-medium text-white"
          style={{ backgroundColor: RISK_COLOR[d.risk] ?? "#94a3b8" }}
        >
          {d.risk}
        </span>
      </div>
      <div className="mt-1.5 space-y-1">
        <BarRow
          label="within"
          width={withinW}
          color="#10b981"
          tooltip={`p10 ${d.within_r2_p10.toFixed(2)} – p90 ${d.within_r2_p90.toFixed(2)}`}
        />
        <BarRow
          label="extrap"
          width={extrapW}
          color="#f97316"
          tooltip={`p10 ${d.extrapolation_r2_p10.toFixed(2)} – p90 ${d.extrapolation_r2_p90.toFixed(2)}`}
        />
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground">
        Penalty: <strong>{d.penalty_pp.toFixed(1)}pp</strong>
      </p>
    </div>
  );
}

function BarRow({
  label,
  width,
  color,
  tooltip,
}: {
  label: string;
  width: number;
  color: string;
  tooltip: string;
}) {
  return (
    <div className="flex items-center gap-2" title={tooltip}>
      <span className="w-12 text-[10px] uppercase text-muted-foreground">
        {label}
      </span>
      <div className="flex-1 h-2 rounded-full bg-secondary overflow-hidden">
        <div
          className="h-full"
          style={{
            width: `${Math.max(0, Math.min(100, width * 100))}%`,
            backgroundColor: color,
          }}
        />
      </div>
      <span className="w-12 text-right tabular-nums text-[10px]">
        {(width * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function ImportanceBar({ importances }: { importances: number[] }) {
  const data = importances
    .map((v, i) => ({ feature: `feat_${i}`, importance: v }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 12);
  return (
    <ResponsiveContainer width="100%" height={Math.max(240, data.length * 24)}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis type="number" tick={{ fontSize: 10 }} />
        <YAxis dataKey="feature" type="category" tick={{ fontSize: 10 }} width={150} />
        <Tooltip />
        <Bar dataKey="importance" fill="#6366f1">
          {data.map((_, i) => (
            <Cell key={i} fill={i < 3 ? "#3b82f6" : "#a5b4fc"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
