"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { validate, type ValidateResponse, type RuleResult } from "@/lib/api";
import { cn } from "@/lib/utils";

const SEVERITY_STYLES: Record<RuleResult["severity"], string> = {
  critical: "border-destructive/50 bg-destructive/5 text-destructive",
  warning: "border-amber-400/60 bg-amber-50 text-amber-900",
  info: "border-muted bg-muted/30 text-muted-foreground",
};

function scoreBand(score: number): { label: string; tone: string } {
  if (score >= 90) return { label: "Excellent", tone: "bg-emerald-500" };
  if (score >= 70) return { label: "Good", tone: "bg-emerald-400" };
  if (score >= 50) return { label: "Caution", tone: "bg-amber-400" };
  return { label: "Critical", tone: "bg-destructive" };
}

export default function ValidationResultsPage({
  params,
}: {
  params: { id: string };
}) {
  const [data, setData] = useState<ValidateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    validate(params.id)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [params.id]);

  if (error) {
    return (
      <main className="container py-12">
        <div className="rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="container py-12">
        <p className="text-muted-foreground">Running validation…</p>
      </main>
    );
  }

  const band = scoreBand(data.data_quality_score);
  const failed = data.rules.filter((r) => !r.passed);

  return (
    <main className="container py-12">
      <header className="mb-8 flex items-end justify-between">
        <div>
          <p className="text-sm uppercase tracking-widest text-muted-foreground">
            Phase 1 · Validation Report
          </p>
          <h1 className="mt-2 text-3xl font-semibold">Data quality score</h1>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            upload_id: {data.upload_id}
          </p>
        </div>
        <div className="text-right">
          <div className="text-5xl font-bold">{data.data_quality_score}</div>
          <span
            className={cn(
              "ml-auto mt-2 inline-block rounded-full px-3 py-1 text-xs font-medium text-white",
              band.tone,
            )}
          >
            {band.label}
          </span>
        </div>
      </header>

      <section className="mb-8 grid gap-4 md:grid-cols-3">
        <div className="rounded-md border p-4">
          <p className="text-sm text-muted-foreground">block_training</p>
          <p
            className={cn(
              "mt-1 text-2xl font-semibold",
              data.block_training ? "text-destructive" : "text-emerald-600",
            )}
          >
            {data.block_training ? "Yes" : "No"}
          </p>
        </div>
        <div className="rounded-md border p-4">
          <p className="text-sm text-muted-foreground">Severity breakdown</p>
          <p className="mt-1 text-sm">
            <span className="font-semibold text-destructive">
              {data.severity_breakdown.critical}
            </span>{" "}
            critical ·{" "}
            <span className="font-semibold text-amber-700">
              {data.severity_breakdown.warning}
            </span>{" "}
            warning ·{" "}
            <span className="font-semibold">
              {data.severity_breakdown.info}
            </span>{" "}
            info
          </p>
        </div>
        <div className="rounded-md border p-4">
          <p className="text-sm text-muted-foreground">Total rules evaluated</p>
          <p className="mt-1 text-2xl font-semibold">{data.rules.length}</p>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="mb-3 text-lg font-medium">
          Rules requiring action ({failed.length})
        </h2>
        {failed.length === 0 ? (
          <p className="rounded-md border border-emerald-300 bg-emerald-50 p-4 text-sm text-emerald-900">
            All rules passed. The dataset is ready for cleaning and donor-pool
            admission.
          </p>
        ) : (
          <ul className="space-y-3">
            {failed.map((r) => (
              <li
                key={r.rule_id}
                className={cn("rounded-md border p-4", SEVERITY_STYLES[r.severity])}
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="font-mono text-xs uppercase">
                      {r.severity}
                    </p>
                    <p className="mt-1 font-semibold text-foreground">
                      {r.rule_id}
                    </p>
                  </div>
                  <span className="rounded-full bg-foreground/10 px-2 py-0.5 text-xs">
                    {r.affected_rows.length} row{r.affected_rows.length === 1 ? "" : "s"}
                  </span>
                </div>
                <p className="mt-2 text-sm text-foreground">{r.description}</p>
                {r.fix_suggestion && (
                  <p className="mt-2 text-sm">
                    <span className="font-medium">Fix:</span> {r.fix_suggestion}
                  </p>
                )}
                {r.paper_reference && (
                  <p className="mt-2 text-xs text-foreground/70">
                    Reference: {r.paper_reference}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-lg font-medium">All rules</h2>
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full text-sm">
            <thead className="bg-secondary">
              <tr>
                <th className="px-3 py-2 text-left">Rule</th>
                <th className="px-3 py-2 text-left">Severity</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Rows</th>
              </tr>
            </thead>
            <tbody>
              {data.rules.map((r) => (
                <tr key={r.rule_id} className="border-t">
                  <td className="px-3 py-2 font-mono">{r.rule_id}</td>
                  <td className="px-3 py-2">{r.severity}</td>
                  <td
                    className={cn(
                      "px-3 py-2",
                      r.passed ? "text-emerald-700" : "text-destructive",
                    )}
                  >
                    {r.passed ? "passed" : "failed"}
                  </td>
                  <td className="px-3 py-2">{r.affected_rows.length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-8 flex gap-3">
          <Link
            href={`/cleaning?upload_id=${data.upload_id}`}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
          >
            Continue to cleaning →
          </Link>
        </div>
      </section>
    </main>
  );
}
