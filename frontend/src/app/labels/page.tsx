"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { type GenerateLabelsResponse, generateLabels } from "@/lib/api";
import { cn } from "@/lib/utils";

const MC_BADGE: Record<string, string> = {
  sample_split: "bg-emerald-500 text-white",
  shared_sample_compromise: "bg-amber-400 text-black",
  blocked: "bg-destructive text-white",
};

function LabelsInner() {
  const params = useSearchParams();
  const uploadId = params.get("upload_id");
  const [data, setData] = useState<GenerateLabelsResponse | null>(null);
  const [hasUserLevel, setHasUserLevel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (id: string) => {
    setBusy(true);
    setError(null);
    try {
      setData(await generateLabels(id, hasUserLevel));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (uploadId) void run(uploadId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId, hasUserLevel]);

  if (!uploadId) {
    return (
      <p className="text-muted-foreground">
        No <code>upload_id</code>. Start from{" "}
        <a className="underline" href="/upload">
          Upload
        </a>
        .
      </p>
    );
  }

  return (
    <>
      <header className="mb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 2 · RCT Label Generator
        </p>
        <h1 className="mt-2 text-3xl font-semibold">ATT, IC, ICPD per RCT</h1>
        <p className="mt-2 font-mono text-xs text-muted-foreground">
          upload_id: {uploadId}
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Labels are computed via the frozen formulas (Eq. 14, 22, 23). The
          mechanical-correlation defense mode is decided per row: <code>sample_split</code>{" "}
          when user-level data is available, <code>shared_sample_compromise</code>{" "}
          otherwise (test-rich pools only), or <code>blocked</code>.
        </p>
      </header>

      <div className="mb-6 flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={hasUserLevel}
            onChange={(e) => setHasUserLevel(e.target.checked)}
          />
          User-level data available (enables sample_split)
        </label>
        <button
          type="button"
          disabled={busy}
          onClick={() => void run(uploadId)}
          className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
        >
          {busy ? "Running…" : "Re-generate labels"}
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {data && (
        <section>
          <p className="mb-3 text-sm text-muted-foreground">
            {data.labels.length} RCT row{data.labels.length === 1 ? "" : "s"}{" "}
            labeled.
          </p>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left">Campaign</th>
                  <th className="px-3 py-2 text-right">D̄ (exposure)</th>
                  <th className="px-3 py-2 text-right">ATT</th>
                  <th className="px-3 py-2 text-right">IC</th>
                  <th className="px-3 py-2 text-right">ICPD</th>
                  <th className="px-3 py-2 text-left">MC defense</th>
                </tr>
              </thead>
              <tbody>
                {data.labels.map((l) => (
                  <tr key={l.campaign_id} className="border-t">
                    <td className="px-3 py-2 font-mono">{l.campaign_id}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {l.exposure_rate.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {l.att.toFixed(5)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {l.incremental_conversions.toFixed(0)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {l.icpd.toFixed(4)}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-xs",
                          MC_BADGE[l.mc_defense_mode] ?? "bg-muted",
                        )}
                      >
                        {l.mc_defense_mode}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}

export default function LabelsPage() {
  return (
    <main className="container py-12">
      <Suspense
        fallback={<p className="text-muted-foreground">Loading labels…</p>}
      >
        <LabelsInner />
      </Suspense>
    </main>
  );
}
