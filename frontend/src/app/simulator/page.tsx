"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import {
  type ActionBand,
  type CampaignAllocation,
  type ModelRecord,
  type SimulatorResponse,
  listModels,
  runSimulator,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const ACTION_BADGE: Record<ActionBand, string> = {
  promote: "bg-emerald-500 text-white",
  hold: "bg-secondary text-foreground",
  deprioritize: "bg-amber-400 text-black",
  block: "bg-destructive text-white",
};

function SimulatorInner() {
  const params = useSearchParams();
  const [uploadId, setUploadId] = useState(params.get("upload_id") ?? "");
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState("");
  const [capMultiplier, setCapMultiplier] = useState(2.0);
  const [totalBudgetOverride, setTotalBudgetOverride] = useState<string>("");
  const [riskFloor, setRiskFloor] = useState<
    "low" | "medium" | "high"
  >("low");
  const [onlyNonRct, setOnlyNonRct] = useState(true);
  const [data, setData] = useState<SimulatorResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void listModels("production").then((m) => setModels(m.models));
  }, []);

  const onRun = async () => {
    if (!uploadId.trim()) {
      setError("Enter an upload_id");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await runSimulator({
        uploadId: uploadId.trim(),
        modelId: modelId || undefined,
        capMultiplier,
        totalBudgetOverride: totalBudgetOverride
          ? Number(totalBudgetOverride)
          : undefined,
        riskFloor,
        onlyNonRct,
      });
      setData(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="container py-12">
      <header className="mb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 4 · Decision Simulator
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Reallocate the budget</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Production-grade only. Reallocates total budget across non-blocked
          campaigns proportional to risk-adjusted ICPD, subject to a per-
          campaign cap (no campaign can grow beyond cap × original spend).
          Surfaces projected incremental conversion lift before you commit.
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      <section className="mb-8 rounded-md border p-5">
        <div className="grid gap-3 md:grid-cols-3">
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
            <span className="font-mono text-muted-foreground">
              model (production only)
            </span>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full rounded-md border bg-background px-2 py-1 text-sm"
            >
              <option value="">Auto (latest production)</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · {m.version_tag}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs">
            <span className="font-mono text-muted-foreground">
              cap multiplier ({capMultiplier.toFixed(2)}×)
            </span>
            <input
              type="range"
              min="1"
              max="5"
              step="0.25"
              value={capMultiplier}
              onChange={(e) => setCapMultiplier(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-muted-foreground">
              max each campaign can grow vs. original spend
            </span>
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="font-mono text-muted-foreground">
              total budget override ($)
            </span>
            <input
              type="number"
              value={totalBudgetOverride}
              onChange={(e) => setTotalBudgetOverride(e.target.value)}
              placeholder="leave blank to use original total"
              className="w-full rounded-md border bg-background px-2 py-1 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="font-mono text-muted-foreground">risk floor</span>
            <select
              value={riskFloor}
              onChange={(e) =>
                setRiskFloor(e.target.value as "low" | "medium" | "high")
              }
              className="w-full rounded-md border bg-background px-2 py-1 text-sm"
            >
              <option value="low">low — strictest</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
        </div>

        <div className="mt-4 flex items-center gap-3">
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
            {busy ? "Reallocating…" : "Run simulator"}
          </button>
        </div>
      </section>

      {data && (
        <>
          <section className="mb-6 grid gap-4 md:grid-cols-4">
            <Card
              label="Original IC total"
              value={data.original_ic_total.toFixed(0)}
              hint={`from $${data.original_total_budget.toLocaleString()} total`}
            />
            <Card
              label="Projected IC total"
              value={data.proposed_ic_total.toFixed(0)}
              hint={`at $${data.total_budget.toLocaleString()} total`}
            />
            <Card
              label="Incremental lift"
              value={
                data.ic_lift_pct !== null
                  ? `${data.ic_lift_pct.toFixed(2)}%`
                  : "—"
              }
              valueClass={
                (data.ic_lift_pct ?? 0) >= 0
                  ? "text-emerald-700"
                  : "text-destructive"
              }
            />
            <Card
              label="Active model"
              value={data.model.version_tag}
              hint={`${data.model.name} · ${data.model.status}`}
            />
          </section>

          <section className="mb-6 rounded-md border p-4 text-sm">
            <p className="text-muted-foreground">{data.rationale}</p>
            <p className="mt-2 text-xs text-muted-foreground">
              n_campaigns: {data.n_campaigns} · blocked: {data.n_blocked} ·
              capped: {data.n_capped} · promoted: {data.n_promoted} ·
              cap_multiplier: {data.cap_multiplier}× · risk_floor:{" "}
              {data.risk_floor}
            </p>
          </section>

          <section>
            <h2 className="mb-3 text-lg font-medium">
              Per-campaign reallocation
            </h2>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-secondary">
                  <tr>
                    <th className="px-3 py-2 text-left">Campaign</th>
                    <th className="px-3 py-2 text-left">Action</th>
                    <th className="px-3 py-2 text-right">ICPD</th>
                    <th className="px-3 py-2 text-right">Original spend</th>
                    <th className="px-3 py-2 text-right">Proposed spend</th>
                    <th className="px-3 py-2 text-right">Δ$</th>
                    <th className="px-3 py-2 text-right">Δ%</th>
                    <th className="px-3 py-2 text-right">Original IC</th>
                    <th className="px-3 py-2 text-right">Projected IC</th>
                    <th className="px-3 py-2 text-left">Capped</th>
                  </tr>
                </thead>
                <tbody>
                  {data.allocations.map((a) => (
                    <Row key={a.run_id} a={a} />
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

function Row({ a }: { a: CampaignAllocation }) {
  const deltaClass =
    a.delta_spend > 0
      ? "text-emerald-700"
      : a.delta_spend < 0
        ? "text-destructive"
        : "";
  return (
    <tr className="border-t">
      <td className="px-3 py-2 font-mono text-xs">{a.campaign_id}</td>
      <td className="px-3 py-2">
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-xs",
            ACTION_BADGE[a.action],
          )}
        >
          {a.action}
        </span>
      </td>
      <td className="px-3 py-2 text-right tabular-nums">
        {a.predicted_icpd.toFixed(4)}
      </td>
      <td className="px-3 py-2 text-right tabular-nums">
        ${a.original_spend.toLocaleString()}
      </td>
      <td className="px-3 py-2 text-right tabular-nums">
        ${a.proposed_spend.toLocaleString()}
      </td>
      <td className={cn("px-3 py-2 text-right tabular-nums", deltaClass)}>
        {a.delta_spend > 0 ? "+" : ""}
        ${a.delta_spend.toLocaleString()}
      </td>
      <td className={cn("px-3 py-2 text-right tabular-nums", deltaClass)}>
        {a.delta_pct !== null
          ? `${a.delta_pct > 0 ? "+" : ""}${a.delta_pct.toFixed(1)}%`
          : "—"}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
        {a.original_ic.toFixed(0)}
      </td>
      <td className="px-3 py-2 text-right tabular-nums">
        {a.proposed_ic.toFixed(0)}
      </td>
      <td className="px-3 py-2 text-xs">
        {a.capped ? (
          <span className="rounded-md bg-amber-400/20 px-2 py-0.5 text-amber-800">
            capped
          </span>
        ) : (
          ""
        )}
      </td>
    </tr>
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

export default function SimulatorPage() {
  return (
    <Suspense
      fallback={
        <main className="container py-12">
          <p className="text-muted-foreground">Loading simulator…</p>
        </main>
      }
    >
      <SimulatorInner />
    </Suspense>
  );
}
