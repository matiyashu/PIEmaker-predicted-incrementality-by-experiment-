"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { clean, type CleanResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const MC_BADGE: Record<string, string> = {
  sample_split: "bg-emerald-500 text-white",
  shared_sample_compromise: "bg-amber-400 text-black",
  blocked: "bg-destructive text-white",
};

function CleaningInner() {
  const params = useSearchParams();
  const uploadId = params.get("upload_id");
  const [data, setData] = useState<CleanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [enableWinsorize, setEnableWinsorize] = useState(false);

  const run = async (id: string) => {
    setBusy(true);
    setError(null);
    try {
      const resp = await clean(id, { enableWinsorize, appliedBy: "ui" });
      setData(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (uploadId) run(uploadId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId]);

  if (!uploadId) {
    return (
      <p className="text-muted-foreground">
        No <code>upload_id</code> in the URL. Start from the{" "}
        <a className="underline" href="/upload">
          Upload page
        </a>
        .
      </p>
    );
  }

  return (
    <>
      <header className="mb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 1 · Cleaning Workbench
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Cleaning audit</h1>
        <p className="mt-2 font-mono text-xs text-muted-foreground">
          upload_id: {uploadId}
        </p>
      </header>

      <div className="mb-6 flex items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={enableWinsorize}
            onChange={(e) => setEnableWinsorize(e.target.checked)}
          />
          Enable winsorization (1st/99th pct)
        </label>
        <button
          disabled={busy}
          onClick={() => run(uploadId)}
          className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
        >
          {busy ? "Running…" : "Re-run pipeline"}
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {data && (
        <>
          <section className="mb-8 grid gap-4 md:grid-cols-2">
            <div className="rounded-md border p-4">
              <p className="text-sm text-muted-foreground">Rows in</p>
              <p className="mt-1 text-3xl font-semibold">{data.rows_in}</p>
            </div>
            <div className="rounded-md border p-4">
              <p className="text-sm text-muted-foreground">Rows out</p>
              <p className="mt-1 text-3xl font-semibold">{data.rows_out}</p>
            </div>
          </section>

          <section className="mb-10">
            <h2 className="mb-3 text-lg font-medium">Cleaning actions (audit log)</h2>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-secondary">
                  <tr>
                    <th className="px-3 py-2 text-left">Action</th>
                    <th className="px-3 py-2 text-left">Rows affected</th>
                    <th className="px-3 py-2 text-left">Applied at</th>
                    <th className="px-3 py-2 text-left">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {data.cleaning_actions.map((a, i) => (
                    <tr key={i} className="border-t">
                      <td className="px-3 py-2 font-mono">{a.action_type}</td>
                      <td className="px-3 py-2">{a.rows_affected}</td>
                      <td className="px-3 py-2 text-muted-foreground">
                        {a.applied_at}
                      </td>
                      <td className="px-3 py-2 text-muted-foreground">
                        {a.notes ?? ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="mb-3 text-lg font-medium">
              Mechanical-correlation defense (RCT rows)
            </h2>
            {data.mc_defense.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No RCT rows in this upload.
              </p>
            ) : (
              <ul className="space-y-3">
                {data.mc_defense.map((m) => (
                  <li key={m.campaign_id} className="rounded-md border p-4">
                    <div className="flex items-center justify-between">
                      <p className="font-mono text-sm">{m.campaign_id}</p>
                      <span
                        className={cn(
                          "rounded-full px-3 py-1 text-xs font-medium",
                          MC_BADGE[m.mc_defense_mode] ?? "bg-muted",
                        )}
                      >
                        {m.mc_defense_mode}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {m.reason}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </>
  );
}

export default function CleaningPage() {
  return (
    <main className="container py-12">
      <Suspense
        fallback={
          <p className="text-muted-foreground">Loading cleaning workbench…</p>
        }
      >
        <CleaningInner />
      </Suspense>
    </main>
  );
}
