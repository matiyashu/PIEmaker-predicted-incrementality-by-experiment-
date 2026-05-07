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

// --- Phase 2: Donor Pool (2.1) ---------------------------------------------

export type PoolBand =
  | "blocked"
  | "research_mode"
  | "production"
  | "production_full";

export interface PoolStatus {
  n_admitted: number;
  band: PoolBand;
  rationale: string;
}

export interface EligibleRct {
  campaign_id: string;
  vertical?: string;
  audience_type?: string;
  funnel_stage?: string;
  test_users?: number;
  control_users?: number;
  test_conversions?: number;
  duration_days?: number;
  end_date?: string;
  quality_score: number;
  admitted: boolean;
  promoted_at: string | null;
  demoted_at: string | null;
  [k: string]: unknown;
}

export interface EligibleResponse {
  upload_id: string;
  rcts: EligibleRct[];
}

export interface CoverageCell {
  vertical: string;
  funnel_stage: string;
  audience_type: string;
  count: number;
}

export interface CoverageResponse {
  verticals: string[];
  funnels: string[];
  audiences: string[];
  cells: CoverageCell[];
}

export interface AgingResponse {
  total_admitted: number;
  same_year: number;
  one_year_ago: number;
  older: number;
  fraction_recent: number;
  extrapolation_risk: "low" | "medium" | "high";
  rationale: string;
}

export interface ShadowRec {
  vertical: string;
  funnel_stage: string;
  audience_type: string;
  gap_score: number;
  status: string;
  brief: string;
}

export interface PromoteResponse {
  promoted: string[];
  status: PoolStatus;
}

export interface DemoteResponse {
  demoted: { campaign_id: string };
  status: PoolStatus;
}

export const getPoolStatus = (): Promise<PoolStatus> =>
  jsonFetch(`${BASE}/api/donor-pool/status`, { method: "GET" });

export const listEligibleRcts = (uploadId: string): Promise<EligibleResponse> =>
  jsonFetch(`${BASE}/api/donor-pool/eligible`, {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId }),
  });

export const promoteRcts = (
  uploadId: string,
  campaignIds: string[],
): Promise<PromoteResponse> =>
  jsonFetch(`${BASE}/api/donor-pool/promote`, {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId, campaign_ids: campaignIds }),
  });

export const demoteRct = (campaignId: string): Promise<DemoteResponse> =>
  jsonFetch(`${BASE}/api/donor-pool/demote`, {
    method: "POST",
    body: JSON.stringify({ campaign_id: campaignId }),
  });

export const fetchCoverage = (uploadId: string): Promise<CoverageResponse> =>
  jsonFetch(`${BASE}/api/donor-pool/coverage`, {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId }),
  });

export const fetchAging = (uploadId: string): Promise<AgingResponse> =>
  jsonFetch(`${BASE}/api/donor-pool/aging`, {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId }),
  });

export const fetchShadowRcts = (
  uploadId: string,
  gapThreshold = 1,
): Promise<{ recommendations: ShadowRec[] }> =>
  jsonFetch(`${BASE}/api/donor-pool/shadow-rcts`, {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId, gap_threshold: gapThreshold }),
  });

// --- Phase 2: Labels (2.2) -------------------------------------------------

export interface LabelRow {
  campaign_id: string;
  att: number;
  incremental_conversions: number;
  icpd: number;
  exposure_rate: number;
  mc_defense_mode: "sample_split" | "shared_sample_compromise" | "blocked";
  sample_split_seed: number | null;
  admitted_to_donor_pool: boolean;
  created_at: string;
}

export interface GenerateLabelsResponse {
  upload_id: string;
  labels: LabelRow[];
}

export const generateLabels = (
  uploadId: string,
  hasUserLevelData = false,
): Promise<GenerateLabelsResponse> =>
  jsonFetch(`${BASE}/api/labels/generate`, {
    method: "POST",
    body: JSON.stringify({
      upload_id: uploadId,
      has_user_level_data: hasUserLevelData,
    }),
  });

// --- Phase 2: Features (2.2) -----------------------------------------------

export interface FeatureRow {
  campaign_id: string;
  feature_set_version: string;
  mode: "training" | "scoring";
  sample_id: string | null;
  x_pre: Record<string, unknown>;
  x_post: Record<string, unknown>;
  created_at: string;
}

export interface BuildFeaturesResponse {
  upload_id: string;
  mode: "training" | "scoring";
  feature_set_version: string;
  x_pre_fields: string[];
  x_post_fields: string[];
  rows: FeatureRow[];
}

export const buildFeatures = (
  uploadId: string,
  options?: {
    mode?: "training" | "scoring";
    featureSetVersion?: string;
    sampleId?: string | null;
  },
): Promise<BuildFeaturesResponse> =>
  jsonFetch(`${BASE}/api/features/build`, {
    method: "POST",
    body: JSON.stringify({
      upload_id: uploadId,
      mode: options?.mode ?? "training",
      feature_set_version: options?.featureSetVersion ?? "v1",
      sample_id: options?.sampleId ?? null,
    }),
  });

// --- Phase 2: Models (2.3 + 2.4) -------------------------------------------

export interface ModelRecord {
  id: string;
  name: string;
  version_tag: string;
  status: "research" | "production";
  algorithm: string;
  feature_set_version: string;
  hyperparameters: Record<string, unknown>;
  training_donor_pool_size: number;
  concept_drift_baseline: { feature_importances?: number[] } | null;
  artifact_path: string;
  created_at: string;
}

export interface BootstrapResult {
  mean: number;
  p025: number;
  p975: number;
  n_draws: number;
}

export interface AblationRow {
  spec: string;
  weighted_r2: number;
  n_train: number;
  n_test: number;
}

export interface TrainResponse {
  model: ModelRecord;
  n_observations: number;
  weighted_r_squared: number;
  bootstrap: BootstrapResult;
  r_squared_ceiling: number;
  lcc_diagnostics: { slope: number | null; spearman_rho: number | null };
  ablation: AblationRow[];
  donor_pool_status: PoolStatus;
}

export interface ModelMetric {
  id: string;
  model_version_id: string;
  metric_type: string;
  value: number;
  ci_lower: number | null;
  ci_upper: number | null;
  segment: Record<string, unknown> | null;
  created_at: string;
}

export interface HoldoutLevel {
  segmentation_var: string;
  level: string;
  within_r2_median: number;
  extrapolation_r2_median: number;
  penalty_pp: number;
  n_iterations: number;
  risk: "severe" | "high" | "medium" | "low" | "unknown";
}

export interface HoldoutResponse {
  segmentation_var: string;
  n_iterations: number;
  results: HoldoutLevel[];
}

export const SEGMENTATION_VARS = [
  "vertical",
  "audience_type",
  "conversion_optimization",
  "custom_audience",
  "advertiser_platform_experience_months",
  "month",
] as const;

export const listModels = (
  status?: "research" | "production",
): Promise<{ models: ModelRecord[] }> => {
  const q = status ? `?status=${status}` : "";
  return jsonFetch(`${BASE}/api/models${q}`, { method: "GET" });
};

export const trainModel = (options?: {
  featureSetVersion?: string;
  name?: string;
  nBootstrap?: number;
}): Promise<TrainResponse> =>
  jsonFetch(`${BASE}/api/models/train`, {
    method: "POST",
    body: JSON.stringify({
      feature_set_version: options?.featureSetVersion ?? "v1",
      name: options?.name ?? "pie_random_forest",
      n_bootstrap: options?.nBootstrap ?? 200,
    }),
  });

export const fetchModelMetrics = (
  modelId: string,
): Promise<{ model_id: string; metrics: ModelMetric[] }> =>
  jsonFetch(`${BASE}/api/models/${modelId}/metrics`, { method: "GET" });

export const promoteModel = (modelId: string): Promise<ModelRecord> =>
  jsonFetch(`${BASE}/api/models/promote`, {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });

export const runHoldout = (
  segmentationVar: string,
  options?: { featureSetVersion?: string; nIterations?: number },
): Promise<HoldoutResponse> =>
  jsonFetch(`${BASE}/api/models/holdout-one-level`, {
    method: "POST",
    body: JSON.stringify({
      segmentation_var: segmentationVar,
      feature_set_version: options?.featureSetVersion ?? "v1",
      n_iterations: options?.nIterations ?? 50,
    }),
  });

// --- Phase 3: Predictions (3.1) --------------------------------------------

export interface SegmentRisk {
  segmentation_var: string;
  level: string;
  within_r2_median?: number | null;
  extrapolation_r2_median?: number | null;
  penalty_pp: number | null;
  risk: "severe" | "high" | "medium" | "low" | "unknown";
}

export interface PredictionRun {
  id: string;
  campaign_id: string;
  model_version_id: string;
  model_status: "research" | "production";
  feature_set_version: string;
  predicted_icpd: number;
  ci_lower: number | null;
  ci_upper: number | null;
  segment_risks: SegmentRisk[];
  worst_segment_risk: SegmentRisk | null;
  watermark: string | null;
  spec: Record<string, unknown>;
  created_at: string;
}

export const scoreCampaign = (
  spec: Record<string, unknown>,
  options?: { modelId?: string; featureSetVersion?: string },
): Promise<PredictionRun> =>
  jsonFetch(`${BASE}/api/predictions/score`, {
    method: "POST",
    body: JSON.stringify({
      spec,
      model_id: options?.modelId ?? null,
      feature_set_version: options?.featureSetVersion ?? "v1",
    }),
  });

export const listPredictions = (
  limit = 50,
): Promise<{ runs: PredictionRun[] }> =>
  jsonFetch(`${BASE}/api/predictions?limit=${limit}`, { method: "GET" });

export interface PortfolioAggregates {
  n: number;
  mean_icpd: number;
  median_icpd: number;
  stdev_icpd: number;
  p10_icpd: number;
  p90_icpd: number;
  risk_counts: Record<"severe" | "high" | "medium" | "low" | "unknown", number>;
}

export interface PortfolioResponse {
  model: ModelRecord;
  runs: PredictionRun[];
  aggregates: PortfolioAggregates;
  worst_segment_risk: SegmentRisk | null;
  watermark: string | null;
}

export const scorePortfolio = (options: {
  uploadId?: string;
  rows?: Record<string, unknown>[];
  modelId?: string;
  featureSetVersion?: string;
  onlyNonRct?: boolean;
}): Promise<PortfolioResponse> =>
  jsonFetch(`${BASE}/api/predictions/score-portfolio`, {
    method: "POST",
    body: JSON.stringify({
      upload_id: options.uploadId ?? null,
      rows: options.rows ?? null,
      model_id: options.modelId ?? null,
      feature_set_version: options.featureSetVersion ?? "v1",
      only_non_rct: options.onlyNonRct ?? true,
    }),
  });
