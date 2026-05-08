import Link from "next/link";
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  Database,
  GaugeCircle,
  ScanSearch,
  ShieldCheck,
  Sliders,
  Sparkles,
} from "lucide-react";

const features = [
  {
    icon: ShieldCheck,
    title: "Trust before UX",
    body: "Frozen formula contracts (ATT, IC, ICPD) ship before the prediction UI. R²-ceiling tells you the upper bound a model can achieve given outcome noise.",
  },
  {
    icon: Database,
    title: "Donor-pool gating",
    body: "Size-band thresholds (200 / 400 / 1600 RCTs) gate training. Below 200 = blocked; 200–399 = research-mode watermark; ≥400 = production.",
  },
  {
    icon: Brain,
    title: "Hold-out-one-level extrapolation",
    body: "Per-level R² penalty across 6 segmentation variables surfaces extrapolation risk per campaign — the paper Table 1 finding, in your dashboard.",
  },
  {
    icon: Sparkles,
    title: "Decision recommendations",
    body: "Risk-adjusted ranking with action bands (promote / hold / deprioritize / block). Severe extrapolation risk auto-sinks to the bottom.",
  },
  {
    icon: ScanSearch,
    title: "Drift monitoring",
    body: "PSI per feature, scoring distribution vs. training. Verdict: stable, watch (≥3 moderate), or retrain_recommended (any severe). No Evidently dependency.",
  },
  {
    icon: Sliders,
    title: "Decision Simulator",
    body: "What-if budget reallocation with risk-adjusted weights and per-campaign cap. Production-only — research models hard-blocked.",
  },
];

const stack = [
  "Next.js 14",
  "FastAPI · Python 3.11",
  "scikit-learn",
  "Random Forest + ablation",
  "PSI drift",
  "Recharts",
];

export default function LandingPage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[600px] bg-[radial-gradient(ellipse_at_top,rgba(59,130,246,0.15),transparent_60%)]" />

      <header className="container flex items-center justify-between py-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center rounded-md bg-primary text-primary-foreground">
            <GaugeCircle className="h-5 w-5" />
          </span>
          <span className="text-lg font-semibold tracking-tight">PIEmaker</span>
        </Link>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1.5 rounded-md border bg-background px-4 py-2 text-sm font-medium shadow-sm hover:bg-secondary"
        >
          Launch app
          <ArrowRight className="h-4 w-4" />
        </Link>
      </header>

      <section className="container pt-12 pb-20 md:pt-20 md:pb-28">
        <div className="max-w-3xl">
          <span className="inline-flex items-center gap-1.5 rounded-full border bg-background/80 px-3 py-1 text-xs font-medium text-muted-foreground">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
            Build plan v3 complete · 218 backend tests passing
          </span>
          <h1 className="mt-4 text-5xl font-semibold tracking-tight md:text-6xl">
            Predict campaign incrementality
            <span className="block bg-gradient-to-r from-blue-600 to-indigo-500 bg-clip-text text-transparent">
              before you spend the budget.
            </span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
            PIEmaker is a campaign-level incrementality prediction workbench
            built on Gordon, Moakler &amp; Zettelmeyer (NBER w35044, April
            2026). Frozen formulas, donor-pool gating, hold-out extrapolation,
            drift monitoring, and a production-only decision simulator —
            shipped end-to-end.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground shadow-sm hover:opacity-90"
            >
              Launch app
              <ArrowRight className="h-4 w-4" />
            </Link>
            <a
              href="https://github.com/matiyashu/PIEmaker-predicted-incrementality-by-experiment-"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md border bg-background px-5 py-2.5 text-sm font-medium hover:bg-secondary"
            >
              View source
            </a>
          </div>
          <p className="mt-6 text-sm text-muted-foreground">
            Built by <span className="font-medium text-foreground">Prima Hanura Akbar</span> · Jakarta · 2026
          </p>
        </div>

        <div className="mt-10 flex flex-wrap gap-2">
          {stack.map((s) => (
            <span
              key={s}
              className="rounded-full border bg-background/70 px-3 py-1 text-xs text-muted-foreground"
            >
              {s}
            </span>
          ))}
        </div>
      </section>

      <section className="container pb-24">
        <div className="mb-10 flex items-end justify-between">
          <div>
            <p className="text-sm uppercase tracking-widest text-muted-foreground">
              What it does
            </p>
            <h2 className="mt-1 text-3xl font-semibold tracking-tight">
              The trust layer ships before the prediction UI
            </h2>
          </div>
        </div>
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                className="rounded-lg border bg-card p-6 shadow-sm transition-shadow hover:shadow-md"
              >
                <span className="grid h-10 w-10 place-items-center rounded-md bg-primary/10 text-primary">
                  <Icon className="h-5 w-5" />
                </span>
                <h3 className="mt-4 font-medium">{f.title}</h3>
                <p className="mt-1.5 text-sm text-muted-foreground">{f.body}</p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="container pb-24">
        <div className="rounded-xl border bg-gradient-to-br from-slate-50 to-blue-50 p-10">
          <h2 className="text-2xl font-semibold tracking-tight">
            One click to populated demo
          </h2>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            The dashboard auto-seeds 400 RCTs into the donor pool, trains a
            production-band Random Forest, runs hold-out-one-level on every
            segmentation variable, and scores 50 candidate campaigns — so
            every page is populated on arrival.
          </p>
          <Link
            href="/dashboard"
            className="mt-6 inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground shadow-sm hover:opacity-90"
          >
            Open the dashboard
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      <footer className="border-t">
        <div className="container flex flex-wrap items-center justify-between gap-4 py-6 text-sm text-muted-foreground">
          <p>
            © 2026 PIEmaker · Built by{" "}
            <span className="font-medium text-foreground">Prima Hanura Akbar</span>
          </p>
          <p>
            Based on NBER w35044 · Gordon, Moakler &amp; Zettelmeyer (April 2026)
          </p>
        </div>
      </footer>
    </main>
  );
}
