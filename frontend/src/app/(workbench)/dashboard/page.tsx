"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Brain,
  Database,
  Sliders,
  Sparkles,
  Target,
} from "lucide-react";

import {
  type DashboardSummary,
  type DemoStatus,
  getDashboardSummary,
  getDemoStatus,
  seedDemo,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const RISK_COLORS: Record<string, string> = {
  severe: "#ef4444",
  high: "#f97316",
  medium: "#f59e0b",
  low: "#10b981",
  unknown: "#94a3b8",
};

const BAND_BADGE: Record<string, string> = {
  blocked: "bg-destructive text-white",
  research_mode: "bg-amber-400 text-black",
  production: "bg-emerald-500 text-white",
  production_full: "bg-emerald-700 text-white",
};

const VERDICT_BADGE: Record<string, string> = {
  stable: "bg-emerald-500 text-white",
  watch: "bg-amber-400 text-black",
  retrain_recommended: "bg-destructive text-white",
};

export default function DashboardPage() {
  const [status, setStatus] = useState<DemoStatus | null>(null);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [seeding, setSeeding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [s, sum] = await Promise.all([getDemoStatus(), getDashboardSummary()]);
      setStatus(s);
      setSummary(sum);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onSeed = async () => {
    setSeeding(true);
    setError(null);
    try {
      await seedDemo();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSeeding(false);
    }
  };

  if (!status) {
    return (
      <main className="container py-12">
        <p className="text-muted-foreground">Loading dashboard…</p>
      </main>
    );
  }

  if (!status.seeded) {
    return (
      <main className="container py-16">
        <div className="mx-auto max-w-2xl rounded-xl border bg-card p-10 text-center shadow-sm">
          <span className="grid h-12 w-12 mx-auto place-items-center rounded-full bg-primary/10 text-primary">
            <Activity className="h-6 w-6" />
          </span>
          <h1 className="mt-4 text-2xl font-semibold tracking-tight">
            Empty workbench
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Seed the demo dataset to populate every page in one click. This
            uploads 400 RCTs, trains a production-band model, runs
            hold-out-one-level on every segmentation variable, and scores 50
            candidate campaigns. Takes ~1–2 minutes (training is the slow step).
          </p>
          {error && (
            <div className="mt-4 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
              {error}
            </div>
          )}
          <button
            type="button"
            onClick={() => void onSeed()}
            disabled={seeding}
            className="mt-6 inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground shadow-sm hover:opacity-90 disabled:opacity-50"
          >
            {seeding ? "Seeding (training the model, ~1–2 min)…" : "Seed demo data"}
            <ArrowRight className="h-4 w-4" />
          </button>
          <p className="mt-4 text-xs text-muted-foreground">
            You can also walk the pipeline manually by visiting{" "}
            <Link href="/upload" className="underline">
              Upload
            </Link>
            .
          </p>
        </div>
      </main>
    );
  }

  if (!summary) {
    return (
      <main className="container py-12">
        <p className="text-muted-foreground">Loading summary…</p>
      </main>
    );
  }

  const verdict = summary.holdouts && summary.holdouts.severity_counts.severe > 0
    ? "retrain_recommended"
    : "stable";

  const histogram = summary.portfolio
    ? buildHistogram(summary.portfolio.icpd_values, 10)
    : [];

  const riskData = summary.portfolio
    ? Object.entries(summary.portfolio.risk_counts)
        .filter(([, v]) => v > 0)
        .map(([name, value]) => ({ name, value }))
    : [];

  return (
    <main className="container py-10">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm uppercase tracking-widest text-muted-foreground">
            Dashboard
          </p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">
            PIEmaker workbench
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Last seeded: {status.last_seeded_at?.replace("T", " ").slice(0, 19)} UTC
          </p>
        </div>
        <button
          type="button"
          onClick={() => void onSeed()}
          disabled={seeding}
          className="rounded-md border bg-background px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
        >
          {seeding ? "Re-seeding…" : "Re-seed demo data"}
        </button>
      </header>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <section className="mb-8 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Hero
          icon={Database}
          label="Donor pool"
          value={String(summary.donor_pool.n_admitted)}
          hint={summary.donor_pool.band.replace("_", " ")}
          badgeClass={BAND_BADGE[summary.donor_pool.band] ?? "bg-muted"}
          href="/donor-pool"
        />
        <Hero
          icon={Brain}
          label="Latest model R²"
          value={
            summary.latest_model?.weighted_r_squared !== null &&
            summary.latest_model?.weighted_r_squared !== undefined
              ? summary.latest_model.weighted_r_squared.toFixed(3)
              : "—"
          }
          hint={
            summary.latest_model?.bootstrap?.ci_lower !== null &&
            summary.latest_model?.bootstrap?.ci_upper !== null &&
            summary.latest_model?.bootstrap
              ? `CI [${summary.latest_model.bootstrap.ci_lower!.toFixed(2)}, ${summary.latest_model.bootstrap.ci_upper!.toFixed(2)}]`
              : `${summary.n_models_total} model(s) registered`
          }
          href="/models"
        />
        <Hero
          icon={BarChart3}
          label="Portfolio mean ICPD"
          value={
            summary.portfolio
              ? summary.portfolio.mean_icpd.toFixed(4)
              : "—"
          }
          hint={
            summary.portfolio
              ? `n=${summary.portfolio.n_runs} candidates`
              : "no portfolio scored yet"
          }
          href="/portfolio"
        />
        <Hero
          icon={Sliders}
          label="Hold-out verdict"
          value={verdict.replace("_", " ")}
          hint={
            summary.holdouts
              ? `${summary.holdouts.n_levels} levels, ${summary.holdouts.n_vars} vars`
              : "no hold-outs run"
          }
          badgeClass={VERDICT_BADGE[verdict] ?? "bg-muted"}
          href="/drift"
        />
      </section>

      <section className="mb-8 grid gap-6 lg:grid-cols-2">
        <ChartCard
          title="Feature ablation (latest model)"
          subtitle="Paper Figure 2 — weighted R² per spec"
        >
          {summary.latest_model?.ablation.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={summary.latest_model.ablation}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="spec"
                  tick={{ fontSize: 10 }}
                  interval={0}
                  height={60}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="weighted_r2" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty msg="No ablation data on the latest model." />
          )}
        </ChartCard>

        <ChartCard
          title="Predicted ICPD distribution"
          subtitle="Across the scored portfolio"
        >
          {histogram.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty msg="No portfolio scored yet." />
          )}
        </ChartCard>
      </section>

      <section className="mb-8 grid gap-6 lg:grid-cols-2">
        <ChartCard
          title="Risk distribution"
          subtitle="Worst extrapolation risk per portfolio campaign"
        >
          {riskData.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={riskData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={2}
                  label={(e) => `${e.name}: ${e.value}`}
                >
                  {riskData.map((d) => (
                    <Cell
                      key={d.name}
                      fill={RISK_COLORS[d.name] ?? "#94a3b8"}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <Empty msg="No portfolio scored yet." />
          )}
        </ChartCard>

        <ChartCard
          title="Quick actions"
          subtitle="Jump to the most-used workbench surfaces"
        >
          <div className="grid gap-2">
            <QuickLink href="/predict" icon={Target} label="Score a single campaign" hint="Form-driven /predict" />
            <QuickLink href="/portfolio" icon={BarChart3} label="Score a portfolio" hint="All non-RCT rows in an upload" />
            <QuickLink href="/decisions" icon={Sparkles} label="Rank with risk gates" hint="promote / hold / deprioritize / block" />
            <QuickLink href="/simulator" icon={Sliders} label="Reallocate budget" hint="Cap-bounded what-if" />
          </div>
        </ChartCard>
      </section>
    </main>
  );
}

function buildHistogram(
  values: number[],
  bins = 10,
): { bin: string; count: number }[] {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return [{ bin: min.toFixed(2), count: values.length }];
  }
  const width = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor((v - min) / width));
    counts[idx]++;
  }
  return counts.map((c, i) => ({
    bin: `${(min + i * width).toFixed(2)}`,
    count: c,
  }));
}

function Hero({
  icon: Icon,
  label,
  value,
  hint,
  badgeClass,
  href,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  hint?: string;
  badgeClass?: string;
  href: string;
}) {
  return (
    <Link
      href={href}
      className="group rounded-lg border bg-card p-5 shadow-sm transition-shadow hover:shadow-md"
    >
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <p
        className={cn(
          "mt-2 text-3xl font-semibold tabular-nums",
          badgeClass &&
            "inline-block rounded-md px-2 py-0.5 text-xl font-medium",
          badgeClass,
        )}
      >
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </Link>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-card p-5 shadow-sm">
      <p className="font-medium">{title}</p>
      {subtitle && (
        <p className="mb-3 text-xs text-muted-foreground">{subtitle}</p>
      )}
      {children}
    </div>
  );
}

function QuickLink({
  href,
  icon: Icon,
  label,
  hint,
}: {
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  hint: string;
}) {
  return (
    <Link
      href={href}
      className="flex items-center gap-3 rounded-md border bg-background p-3 text-sm hover:bg-secondary"
    >
      <span className="grid h-8 w-8 place-items-center rounded-md bg-primary/10 text-primary">
        <Icon className="h-4 w-4" />
      </span>
      <div className="flex-1">
        <p className="font-medium">{label}</p>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </div>
      <ArrowRight className="h-4 w-4 text-muted-foreground" />
    </Link>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="grid h-[260px] place-items-center text-sm text-muted-foreground">
      {msg}
    </div>
  );
}
