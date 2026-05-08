"use client";

import { Suspense, useCallback, useEffect, useState } from "react";

import {
  type AgingResponse,
  type CoverageResponse,
  type EligibleRct,
  type PoolStatus,
  type ShadowRec,
  demoteRct,
  fetchAging,
  fetchCoverage,
  fetchShadowRcts,
  getPoolStatus,
  listEligibleRcts,
  promoteRcts,
} from "@/lib/api";
import { useUploadId } from "@/lib/use-upload-id";
import { cn } from "@/lib/utils";
import { SummaryCard } from "@/components/summary-card";

const BAND_BADGE: Record<PoolStatus["band"], string> = {
  blocked: "bg-destructive text-white",
  research_mode: "bg-amber-400 text-black",
  production: "bg-emerald-500 text-white",
  production_full: "bg-emerald-700 text-white",
};

const RISK_BADGE: Record<AgingResponse["extrapolation_risk"], string> = {
  low: "bg-emerald-500 text-white",
  medium: "bg-amber-400 text-black",
  high: "bg-destructive text-white",
};

function DonorPoolInner() {
  const { uploadId } = useUploadId();

  const [status, setStatus] = useState<PoolStatus | null>(null);
  const [eligible, setEligible] = useState<EligibleRct[] | null>(null);
  const [coverage, setCoverage] = useState<CoverageResponse | null>(null);
  const [aging, setAging] = useState<AgingResponse | null>(null);
  const [shadow, setShadow] = useState<ShadowRec[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await getPoolStatus();
      setStatus(s);
      if (uploadId) {
        const [e, c, a, sh] = await Promise.all([
          listEligibleRcts(uploadId),
          fetchCoverage(uploadId),
          fetchAging(uploadId),
          fetchShadowRcts(uploadId, 1),
        ]);
        setEligible(e.rcts);
        setCoverage(c);
        setAging(a);
        setShadow(sh.recommendations);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [uploadId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggle = (cid: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(cid)) next.delete(cid);
      else next.add(cid);
      return next;
    });
  };

  const promoteSelected = async () => {
    if (!uploadId || selected.size === 0) return;
    setBusy(true);
    setError(null);
    try {
      await promoteRcts(uploadId, [...selected]);
      setSelected(new Set());
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const demote = async (cid: string) => {
    setBusy(true);
    setError(null);
    try {
      await demoteRct(cid);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 2 · Donor Pool Manager
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Curate your training pool</h1>
      </header>

      <SummaryCard
        title="What you're seeing"
        body={
          <>
            Each row is a randomized controlled trial (RCT) the model can
            learn from. The <strong>quality score</strong> blends test-pool
            size, conversion volume, duration, and balance — anything ≥70 is
            production-grade. The <strong>size band</strong> badge gates
            training: ≥400 admitted RCTs flips the pool from research to
            production (no watermark on predictions).
          </>
        }
        recommendations={[
          "Promote high-quality RCTs (score ≥70) first — they reduce extrapolation risk most.",
          "Watch the aging indicator: if <50% of admitted RCTs are from the same calendar year, expect a 21pp R² penalty (PDF Table 1).",
          "Coverage gaps in the heatmap → run a shadow RCT in the missing (vertical, funnel, audience) cell.",
        ]}
      />

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {status && (
        <section className="mb-8 rounded-md border p-5">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-sm text-muted-foreground">Pool size band</p>
            <span
              className={cn(
                "rounded-full px-3 py-1 text-xs font-medium",
                BAND_BADGE[status.band],
              )}
            >
              {status.band.replace("_", " ")}
            </span>
            <span className="text-3xl font-semibold tabular-nums">
              {status.n_admitted}
            </span>
            <span className="text-sm text-muted-foreground">RCTs admitted</span>
          </div>
          <p className="mt-3 text-sm text-muted-foreground">{status.rationale}</p>
          <div className="mt-4 flex gap-2">
            <button
              type="button"
              onClick={() => void refresh()}
              disabled={busy}
              className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
            >
              {busy ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </section>
      )}

      {!uploadId && (
        <section className="rounded-md border p-5 text-sm text-muted-foreground">
          Add <code>?upload_id=…</code> to the URL (from the{" "}
          <a className="underline" href="/upload">
            Upload page
          </a>
          ) to load eligible RCTs, coverage heatmap, and shadow-RCT
          recommendations.
        </section>
      )}

      {uploadId && aging && (
        <section className="mb-8 rounded-md border p-5">
          <div className="flex flex-wrap items-baseline gap-3">
            <p className="text-sm text-muted-foreground">Aging indicator</p>
            <span
              className={cn(
                "rounded-full px-3 py-1 text-xs font-medium",
                RISK_BADGE[aging.extrapolation_risk],
              )}
            >
              {aging.extrapolation_risk} risk
            </span>
            <span className="text-2xl font-semibold tabular-nums">
              {(aging.fraction_recent * 100).toFixed(0)}%
            </span>
            <span className="text-sm text-muted-foreground">recent</span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            same year: {aging.same_year} · 1y old: {aging.one_year_ago} · older:{" "}
            {aging.older} · admitted total: {aging.total_admitted}
          </p>
          <p className="mt-2 text-sm text-muted-foreground">{aging.rationale}</p>
        </section>
      )}

      {uploadId && eligible && (
        <section className="mb-8">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-medium">Eligible RCTs</h2>
            <button
              type="button"
              onClick={() => void promoteSelected()}
              disabled={busy || selected.size === 0}
              className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
            >
              Promote {selected.size} selected
            </button>
          </div>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Pick</th>
                  <th className="px-3 py-2 text-left">Campaign</th>
                  <th className="px-3 py-2 text-left">Vertical</th>
                  <th className="px-3 py-2 text-left">Funnel</th>
                  <th className="px-3 py-2 text-left">Audience</th>
                  <th className="px-3 py-2 text-right">Quality</th>
                  <th className="px-3 py-2 text-left">Admitted</th>
                  <th className="px-3 py-2 text-left">Action</th>
                </tr>
              </thead>
              <tbody>
                {eligible.map((r) => (
                  <tr key={r.campaign_id} className="border-t">
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selected.has(r.campaign_id)}
                        onChange={() => toggle(r.campaign_id)}
                        disabled={r.admitted}
                      />
                    </td>
                    <td className="px-3 py-2 font-mono">{r.campaign_id}</td>
                    <td className="px-3 py-2">{r.vertical ?? ""}</td>
                    <td className="px-3 py-2">{r.funnel_stage ?? ""}</td>
                    <td className="px-3 py-2">{r.audience_type ?? ""}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      <span
                        className={cn(
                          "rounded-md px-2 py-0.5 text-xs",
                          r.quality_score >= 70
                            ? "bg-emerald-500/15 text-emerald-700"
                            : r.quality_score >= 40
                              ? "bg-amber-400/20 text-amber-700"
                              : "bg-destructive/15 text-destructive",
                        )}
                      >
                        {r.quality_score}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      {r.admitted ? (
                        <span className="text-emerald-600">✓</span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      {r.admitted && (
                        <button
                          type="button"
                          onClick={() => void demote(r.campaign_id)}
                          disabled={busy}
                          className="text-xs underline disabled:opacity-50"
                        >
                          Demote
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {uploadId && coverage && coverage.cells.length > 0 && (
        <section className="mb-8">
          <h2 className="mb-3 text-lg font-medium">
            Coverage heatmap (admitted only)
          </h2>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Vertical</th>
                  <th className="px-3 py-2 text-left">Funnel</th>
                  <th className="px-3 py-2 text-left">Audience</th>
                  <th className="px-3 py-2 text-right">Count</th>
                </tr>
              </thead>
              <tbody>
                {coverage.cells.map((c, i) => (
                  <tr key={i} className="border-t">
                    <td className="px-3 py-2">{c.vertical}</td>
                    <td className="px-3 py-2">{c.funnel_stage}</td>
                    <td className="px-3 py-2">{c.audience_type}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      <span
                        className={cn(
                          "rounded-md px-2 py-0.5 text-xs",
                          c.count >= 5
                            ? "bg-emerald-500/15 text-emerald-700"
                            : c.count >= 1
                              ? "bg-amber-400/20 text-amber-700"
                              : "bg-destructive/15 text-destructive",
                        )}
                      >
                        {c.count}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {uploadId && shadow && shadow.length > 0 && (
        <section className="mb-8">
          <h2 className="mb-3 text-lg font-medium">
            Shadow-RCT recommendations (coverage gaps)
          </h2>
          <ul className="space-y-3">
            {shadow.map((r, i) => (
              <li key={i} className="rounded-md border p-4">
                <p className="text-sm font-mono">
                  {r.vertical} · {r.funnel_stage} · {r.audience_type}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">{r.brief}</p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

export default function DonorPoolPage() {
  return (
    <main className="container py-12">
      <Suspense
        fallback={<p className="text-muted-foreground">Loading donor pool…</p>}
      >
        <DonorPoolInner />
      </Suspense>
    </main>
  );
}
