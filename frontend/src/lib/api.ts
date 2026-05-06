// Typed fetch helpers for the PIEmaker backend.
// Uses Next.js rewrites: /api/backend/* -> http://localhost:8000/*

const BASE = "/api/backend";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

// --- Types -----------------------------------------------------------------

export interface UploadResponse {
  upload_id: string;
  filename: string;
  rows: number;
  columns: string[];
}

export interface MappingSuggestion {
  source_column: string;
  target_field: string | null;
  confidence: number;
  reason: string;
}

export interface SchemaSuggestResponse {
  upload_id: string;
  source_columns: string[];
  suggestions: MappingSuggestion[];
}

export interface RuleResult {
  rule_id: string;
  severity: "critical" | "warning" | "info";
  description: string;
  passed: boolean;
  affected_rows: number[];
  fix_suggestion: string | null;
  paper_reference: string | null;
}

export interface ValidateResponse {
  upload_id: string;
  data_quality_score: number;
  block_training: boolean;
  severity_breakdown: { critical: number; warning: number; info: number };
  rules: RuleResult[];
}

export interface CleaningAction {
  action_type: string;
  rows_affected: number;
  before_summary: Record<string, unknown>;
  after_summary: Record<string, unknown>;
  applied_at: string;
  applied_by: string | null;
  notes: string | null;
}

export interface MCDefense {
  campaign_id: string;
  mc_defense_mode: "sample_split" | "shared_sample_compromise" | "blocked";
  sample_split_seed: number | null;
  reason: string;
}

export interface CleanResponse {
  upload_id: string;
  rows_in: number;
  rows_out: number;
  cleaning_actions: CleaningAction[];
  mc_defense: MCDefense[];
}

// --- API calls -------------------------------------------------------------

export async function uploadFile(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch(`${BASE}/api/upload`, { method: "POST", body: fd });
  if (!resp.ok) {
    throw new Error(`upload failed: ${resp.status} ${await resp.text()}`);
  }
  return resp.json();
}

export function suggestMapping(uploadId: string): Promise<SchemaSuggestResponse> {
  return jsonFetch(`${BASE}/api/schema/suggest`, {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId }),
  });
}

export function validate(
  uploadId: string,
  columnMapping?: Record<string, string>,
): Promise<ValidateResponse> {
  return jsonFetch(`${BASE}/api/validate`, {
    method: "POST",
    body: JSON.stringify({
      upload_id: uploadId,
      column_mapping: columnMapping ?? null,
    }),
  });
}

export function clean(
  uploadId: string,
  options?: {
    columnMapping?: Record<string, string>;
    fxRates?: Record<string, number>;
    enableWinsorize?: boolean;
    appliedBy?: string;
  },
): Promise<CleanResponse> {
  return jsonFetch(`${BASE}/api/clean`, {
    method: "POST",
    body: JSON.stringify({
      upload_id: uploadId,
      column_mapping: options?.columnMapping ?? null,
      fx_rates: options?.fxRates ?? null,
      enable_winsorize: options?.enableWinsorize ?? false,
      applied_by: options?.appliedBy ?? null,
    }),
  });
}
