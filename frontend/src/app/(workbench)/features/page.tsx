"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { type BuildFeaturesResponse, buildFeatures } from "@/lib/api";

function FeaturesInner() {
  const params = useSearchParams();
  const uploadId = params.get("upload_id");
  const [mode, setMode] = useState<"training" | "scoring">("training");
  const [data, setData] = useState<BuildFeaturesResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (id: string) => {
    setBusy(true);
    setError(null);
    try {
      setData(await buildFeatures(id, { mode }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (uploadId) void run(uploadId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId, mode]);

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
          Phase 2 · Feature Engineering Studio
        </p>
        <h1 className="mt-2 text-3xl font-semibold">X_pre + X_post</h1>
        <p className="mt-2 font-mono text-xs text-muted-foreground">
          upload_id: {uploadId}
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Pre-determined features are knowable before launch; post-determined
          features only after the campaign runs. v3 adds three pre features:
          conversion_optimization, custom_audience,
          advertiser_platform_experience_months.
        </p>
      </header>

      <div className="mb-6 flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm">
          Mode
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "training" | "scoring")}
            className="rounded-md border bg-background px-2 py-1 text-sm"
          >
            <option value="training">training</option>
            <option value="scoring">scoring</option>
          </select>
        </label>
        <button
          type="button"
          disabled={busy}
          onClick={() => void run(uploadId)}
          className="rounded-md border px-3 py-1.5 text-sm hover:bg-secondary disabled:opacity-50"
        >
          {busy ? "Building…" : "Re-build features"}
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {data && (
        <>
          <section className="mb-6 grid gap-4 md:grid-cols-2">
            <div className="rounded-md border p-4">
              <p className="text-sm text-muted-foreground">X_pre fields</p>
              <p className="mt-1 text-2xl font-semibold">
                {data.x_pre_fields.length}
              </p>
              <p className="mt-2 break-words text-xs text-muted-foreground">
                {data.x_pre_fields.join(", ")}
              </p>
            </div>
            <div className="rounded-md border p-4">
              <p className="text-sm text-muted-foreground">X_post fields</p>
              <p className="mt-1 text-2xl font-semibold">
                {data.x_post_fields.length}
              </p>
              <p className="mt-2 break-words text-xs text-muted-foreground">
                {data.x_post_fields.join(", ")}
              </p>
            </div>
          </section>

          <section>
            <p className="mb-3 text-sm text-muted-foreground">
              {data.rows.length} row{data.rows.length === 1 ? "" : "s"} built ·
              feature_set_version: <code>{data.feature_set_version}</code>
            </p>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-secondary">
                  <tr>
                    <th className="px-3 py-2 text-left">Campaign</th>
                    <th className="px-3 py-2 text-left">Mode</th>
                    <th className="px-3 py-2 text-left">Vertical</th>
                    <th className="px-3 py-2 text-left">Audience</th>
                    <th className="px-3 py-2 text-right">CPD</th>
                    <th className="px-3 py-2 text-right">LCC-7d/$</th>
                    <th className="px-3 py-2 text-right">Exp. rate</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.slice(0, 50).map((r) => (
                    <tr key={r.campaign_id} className="border-t">
                      <td className="px-3 py-2 font-mono">{r.campaign_id}</td>
                      <td className="px-3 py-2">{r.mode}</td>
                      <td className="px-3 py-2">
                        {String(r.x_pre.vertical ?? "")}
                      </td>
                      <td className="px-3 py-2">
                        {String(r.x_pre.audience_type ?? "")}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {numFmt(r.x_post.conversions_per_dollar)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {numFmt(r.x_post.lcc_7d_per_dollar)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {numFmt(r.x_post.exposure_rate)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {data.rows.length > 50 && (
              <p className="mt-2 text-xs text-muted-foreground">
                Showing first 50 of {data.rows.length} rows.
              </p>
            )}
          </section>
        </>
      )}
    </>
  );
}

function numFmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  const n = typeof v === "number" ? v : Number(v);
  if (Number.isNaN(n)) return "—";
  return n.toFixed(4);
}

export default function FeaturesPage() {
  return (
    <main className="container py-12">
      <Suspense
        fallback={<p className="text-muted-foreground">Loading features…</p>}
      >
        <FeaturesInner />
      </Suspense>
    </main>
  );
}
