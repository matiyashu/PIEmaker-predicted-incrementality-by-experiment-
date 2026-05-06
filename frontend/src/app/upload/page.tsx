"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";

import { uploadFile, type UploadResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const MAX_BYTES = 100 * 1024 * 1024;

export default function UploadPage() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      setError(null);
      const file = accepted[0];
      if (!file) return;
      if (file.size > MAX_BYTES) {
        setError(`File exceeds ${MAX_BYTES / (1024 * 1024)} MB limit.`);
        return;
      }
      setBusy(true);
      try {
        const resp = await uploadFile(file);
        setResult(resp);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
  });

  return (
    <main className="container py-12">
      <header className="mb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          Phase 1 · Upload Studio
        </p>
        <h1 className="mt-2 text-3xl font-semibold">Upload campaign or RCT data</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Drop a CSV or Excel file. We&rsquo;ll detect columns, suggest mappings to
          the standard PIE schema (Appendix B), then run validation and cleaning.
        </p>
      </header>

      <div
        {...getRootProps()}
        className={cn(
          "flex h-48 cursor-pointer items-center justify-center rounded-lg border-2 border-dashed",
          isDragActive ? "border-primary bg-primary/5" : "border-muted",
          busy && "pointer-events-none opacity-60",
        )}
      >
        <input {...getInputProps()} />
        <div className="text-center">
          <p className="font-medium">
            {busy
              ? "Uploading…"
              : isDragActive
                ? "Drop the file here"
                : "Drag & drop a file, or click to browse"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            CSV, XLSX, XLS · max 100 MB
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-6 rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {result && (
        <section className="mt-8 space-y-6">
          <div className="rounded-md border p-4">
            <h2 className="font-medium">Upload received</h2>
            <dl className="mt-2 grid gap-2 text-sm md:grid-cols-3">
              <div>
                <dt className="text-muted-foreground">upload_id</dt>
                <dd className="font-mono">{result.upload_id}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">filename</dt>
                <dd>{result.filename}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">rows × columns</dt>
                <dd>
                  {result.rows.toLocaleString()} × {result.columns.length}
                </dd>
              </div>
            </dl>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => router.push(`/upload/results/${result.upload_id}`)}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
            >
              Run validation →
            </button>
            <button
              onClick={() => router.push(`/cleaning?upload_id=${result.upload_id}`)}
              className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-secondary"
            >
              Skip to cleaning
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
