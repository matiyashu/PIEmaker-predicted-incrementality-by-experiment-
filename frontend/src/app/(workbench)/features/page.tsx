"use client";

import { Suspense, useEffect, useState } from "react";

import { type BuildFeaturesResponse, buildFeatures } from "@/lib/api";
import { useUploadId } from "@/lib/use-upload-id";
import { SummaryCard } from "@/components/summary-card";

function FeaturesInner() {
  const { uploadId } = useUploadId();
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
      <header className="mb-6">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 2 · Feature Engineering Studio
        </p>
        <h1 className="mt-2 text-3xl font-semibold">X_pre + X_post</h1>
        <p className="mt-1 font-mono text-xs text-muted-foreground">
          upload_id: {uploadId}
        </p>
      </header>

      <SummaryCard
        title="What you're seeing"
        body={
          <>
            Two feature families. <strong>X_pre</strong> is everything
            knowable <em>before</em> a campaign launches (vertical, audience,
            objective, creative format, planned spend tier) — these power
            forward-looking forecasts. <strong>X_post</strong> is the
            post-mortem signal (CTR, exposure rate, LCC-7d/$, conversions/$)
            — paper Figure 2 shows these dominate the model&rsquo;s explanatory
            power. v3 adds three new X_pre fields:
            conversion_optimization, custom_audience, and
            advertiser_platform_experience_months.
          </>
        }
        recommendations={[
          "Use mode=training to feed the donor-pool labels into the model.",
          "Switch to mode=scoring to engineer features for new campaigns you want to forecast.",
          "Missing X_post columns? The model still predicts from X_pre alone — useful for pre-launch forecasts, but expect lower R² without LCC signals.",
        ]}
      />

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
