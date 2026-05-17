// Canned realistic responses for the PIEmaker frontend. Used when demo mode
// is active so the dashboard renders without a backend (e.g. Vercel-only,
// offline screenshots, presentation fallback).
//
// Every export here matches the type contract from `api.ts`. When the user
// opens any page in demo mode, the typed clients in `api.ts` short-circuit
// to these functions instead of fetching.
//
// All randomness is driven by a single seeded PRNG so refreshes are stable
// (good for screenshots / repeatable demos).

import type {
  AgingResponse,
  BuildFeaturesResponse,
  CleanResponse,
  CoverageResponse,
  DashboardSummary,
  DecisionResponse,
  DemoSeedResult,
  DemoStatus,
  DemoteResponse,
  DriftResponse,
  EligibleRct,
  EligibleResponse,
  GenerateLabelsResponse,
  HoldoutLevel,
  HoldoutResponse,
  ModelMetric,
  ModelRecord,
  PoolStatus,
  PortfolioResponse,
  PredictionRun,
  PromoteResponse,
  Recommendation,
  SchemaSuggestResponse,
  SegmentRisk,
  ShadowRec,
  SimulatorResponse,
  TrainResponse,
  UploadResponse,
  ValidateResponse,
} from "./api";

// --- Seeded PRNG ----------------------------------------------------------

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(2026);
const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
const pick = <T>(xs: readonly T[]) => xs[Math.floor(rng() * xs.length)];

// --- Shared dimensions ----------------------------------------------------

const VERTICALS = ["ecommerce", "travel", "finance", "media"] as const;
const AUDIENCES = ["retargeting", "prospecting", "lookalike"] as const;
const FUNNELS = ["upper", "mid", "lower"] as const;
const CREATIVES = ["video", "image", "carousel"] as const;
const PLACEMENTS = ["feed", "story", "reels"] as const;

const NOW = "2026-05-08T07:42:11.000000+00:00";

// --- Models / metrics -----------------------------------------------------

const DEMO_MODEL_ID = "mockmodel0001";

const DEMO_MODEL: ModelRecord = {
  id: DEMO_MODEL_ID,
  name: "pie_random_forest_demo",
  version_tag: "v-mock01",
  status: "production",
  algorithm: "random_forest",
  feature_set_version: "v1",
  hyperparameters: {
    n_estimators: 150,
    max_depth: 10,
    min_samples_leaf: 3,
    max_features: "sqrt",
  },
  training_donor_pool_size: 450,
  concept_drift_baseline: {
    feature_importances: Array.from({ length: 26 }, () => Number(rand(0, 0.15).toFixed(4))),
  },
  artifact_path: "<mock>/state/models/mockmodel0001.pkl",
  created_at: NOW,
};

const ABLATION = [
  { spec: "PIE(Pre)", weighted_r2: 0.42 },
  { spec: "PIE(Pre+Yt)", weighted_r2: 0.55 },
  { spec: "PIE(Pre+Yt+LCC-7D)", weighted_r2: 0.67 },
  { spec: "PIE(Full)", weighted_r2: 0.78 },
  { spec: "Raw_LCC_7D_benchmark", weighted_r2: 0.61 },
];

const HOLDOUT_LEVELS: HoldoutLevel[] = [
  // vertical
  { segmentation_var: "vertical", level: "ecommerce", within_r2_median: 0.80, extrapolation_r2_median: 0.74, penalty_pp: 6.0, n_iterations: 10, risk: "medium" },
  { segmentation_var: "vertical", level: "travel", within_r2_median: 0.77, extrapolation_r2_median: 0.62, penalty_pp: 14.5, n_iterations: 10, risk: "medium" },
  { segmentation_var: "vertical", level: "finance", within_r2_median: 0.74, extrapolation_r2_median: 0.51, penalty_pp: 22.5, n_iterations: 10, risk: "high" },
  { segmentation_var: "vertical", level: "media", within_r2_median: 0.70, extrapolation_r2_median: 0.41, penalty_pp: 28.7, n_iterations: 10, risk: "severe" },
  // audience_type
  { segmentation_var: "audience_type", level: "retargeting", within_r2_median: 0.81, extrapolation_r2_median: 0.78, penalty_pp: 3.2, n_iterations: 10, risk: "low" },
  { segmentation_var: "audience_type", level: "prospecting", within_r2_median: 0.74, extrapolation_r2_median: 0.62, penalty_pp: 12.3, n_iterations: 10, risk: "medium" },
  { segmentation_var: "audience_type", level: "lookalike", within_r2_median: 0.71, extrapolation_r2_median: 0.58, penalty_pp: 13.4, n_iterations: 10, risk: "medium" },
  // conversion_optimization
  { segmentation_var: "conversion_optimization", level: "yes", within_r2_median: 0.79, extrapolation_r2_median: 0.74, penalty_pp: 5.1, n_iterations: 10, risk: "medium" },
  { segmentation_var: "conversion_optimization", level: "no", within_r2_median: 0.71, extrapolation_r2_median: 0.62, penalty_pp: 9.0, n_iterations: 10, risk: "medium" },
  // custom_audience
  { segmentation_var: "custom_audience", level: "yes", within_r2_median: 0.78, extrapolation_r2_median: 0.73, penalty_pp: 5.5, n_iterations: 10, risk: "medium" },
  { segmentation_var: "custom_audience", level: "no", within_r2_median: 0.74, extrapolation_r2_median: 0.66, penalty_pp: 7.6, n_iterations: 10, risk: "medium" },
  // advertiser_platform_experience_months
  { segmentation_var: "advertiser_platform_experience_months", level: "12", within_r2_median: 0.75, extrapolation_r2_median: 0.68, penalty_pp: 7.0, n_iterations: 10, risk: "medium" },
  { segmentation_var: "advertiser_platform_experience_months", level: "24", within_r2_median: 0.78, extrapolation_r2_median: 0.74, penalty_pp: 4.0, n_iterations: 10, risk: "low" },
  { segmentation_var: "advertiser_platform_experience_months", level: "36", within_r2_median: 0.80, extrapolation_r2_median: 0.76, penalty_pp: 4.5, n_iterations: 10, risk: "low" },
  // month
  { segmentation_var: "month", level: "1", within_r2_median: 0.75, extrapolation_r2_median: 0.71, penalty_pp: 4.1, n_iterations: 10, risk: "low" },
  { segmentation_var: "month", level: "2", within_r2_median: 0.77, extrapolation_r2_median: 0.69, penalty_pp: 8.0, n_iterations: 10, risk: "medium" },
  { segmentation_var: "month", level: "3", within_r2_median: 0.79, extrapolation_r2_median: 0.74, penalty_pp: 5.0, n_iterations: 10, risk: "medium" },
  { segmentation_var: "month", level: "4", within_r2_median: 0.78, extrapolation_r2_median: 0.71, penalty_pp: 7.4, n_iterations: 10, risk: "medium" },
];

function _pickHoldoutForLevel(
  segVar: string,
  level: string | number | undefined,
): HoldoutLevel | undefined {
  if (level === undefined || level === null) return undefined;
  return HOLDOUT_LEVELS.find(
    (h) => h.segmentation_var === segVar && h.level === String(level),
  );
}

// --- RCTs / portfolio runs ------------------------------------------------

function _genRct(i: number, isAdmitted: boolean): EligibleRct {
  const test_users = Math.floor(rand(800_000, 3_000_000));
  const control_users = Math.floor(test_users * rand(0.85, 1.15));
  const test_conversions = Math.floor(test_users * rand(0.018, 0.030));
  const duration_days = pick([14, 21, 28, 35]);
  return {
    campaign_id: `RCT-${String(i).padStart(4, "0")}`,
    vertical: pick(VERTICALS),
    audience_type: pick(AUDIENCES),
    funnel_stage: pick(FUNNELS),
    test_users,
    control_users,
    test_conversions,
    duration_days,
    end_date: "2026-04-15",
    quality_score: Math.floor(rand(35, 95)),
    admitted: isAdmitted,
    promoted_at: isAdmitted ? NOW : null,
    demoted_at: null,
  };
}

const ELIGIBLE_RCTS: EligibleRct[] = Array.from({ length: 60 }, (_, i) => _genRct(i, i < 50));

const RCT_LABELS = Array.from({ length: 50 }, (_, i) => {
  const exposure_rate = rand(0.6, 0.95);
  const att = rand(0.003, 0.012);
  const test_users = Math.floor(rand(1_000_000, 2_500_000));
  const ic = att * exposure_rate * test_users;
  const cost = rand(20_000, 200_000);
  return {
    campaign_id: `RCT-${String(i).padStart(4, "0")}`,
    att,
    incremental_conversions: ic,
    icpd: ic / cost,
    exposure_rate,
    mc_defense_mode:
      i % 5 === 0
        ? ("blocked" as const)
        : i % 3 === 0
          ? ("sample_split" as const)
          : ("shared_sample_compromise" as const),
    sample_split_seed: i % 3 === 0 ? Math.floor(rand(1, 1e9)) : null,
    admitted_to_donor_pool: false,
    created_at: NOW,
  };
});

function _genPredictionRun(i: number, isPortfolio: boolean): PredictionRun {
  const cid = isPortfolio
    ? `CMP-${String(i).padStart(4, "0")}`
    : `PRED-${String(i).padStart(4, "0")}`;
  const predicted_icpd = Number(rand(0.02, 0.18).toFixed(4));
  const ci_half = 0.025;
  const vertical = pick(VERTICALS);
  const audience_type = pick(AUDIENCES);
  const verticalRisk = _pickHoldoutForLevel("vertical", vertical);
  const audienceRisk = _pickHoldoutForLevel("audience_type", audience_type);
  const segment_risks: SegmentRisk[] = [];
  if (verticalRisk) segment_risks.push({ ...verticalRisk });
  if (audienceRisk) segment_risks.push({ ...audienceRisk });
  const worst_segment_risk =
    segment_risks
      .filter((r) => r.penalty_pp !== null && r.penalty_pp !== undefined)
      .sort((a, b) => (b.penalty_pp ?? 0) - (a.penalty_pp ?? 0))[0] ?? null;

  const cost = Number(rand(20_000, 200_000).toFixed(2));

  return {
    id: `mockrun${String(i).padStart(6, "0")}`,
    campaign_id: cid,
    model_version_id: DEMO_MODEL_ID,
    model_status: "production",
    feature_set_version: "v1",
    predicted_icpd,
    ci_lower: Number((predicted_icpd - ci_half).toFixed(4)),
    ci_upper: Number((predicted_icpd + ci_half).toFixed(4)),
    segment_risks,
    worst_segment_risk,
    watermark: null,
    spec: {
      campaign_id: cid,
      vertical,
      audience_type,
      funnel_stage: pick(FUNNELS),
      objective: "conversions",
      conversion_optimization: pick(["yes", "no"]),
      custom_audience: pick(["yes", "no"]),
      advertiser_platform_experience_months: pick([12, 24, 36]),
      creative_format: pick(CREATIVES),
      placement: pick(PLACEMENTS),
      bid_strategy: pick(["lowest_cost", "cost_cap", "bid_cap"]),
      market: pick(["US", "UK", "ID", "DE"]),
      spend_tier: pick(["low", "medium", "high"]),
      platform: pick(["meta", "tiktok", "google"]),
      cost,
    },
    created_at: NOW,
  };
}

const PORTFOLIO_RUNS: PredictionRun[] = Array.from({ length: 50 }, (_, i) =>
  _genPredictionRun(i, true),
);

// --- Public mock entrypoints (one per api.ts function) --------------------

export function mockDemoStatus(): DemoStatus {
  return {
    seeded: true,
    last_seeded_at: NOW,
    model_count: 1,
    donor_pool_admitted: 450,
    prediction_run_count: PORTFOLIO_RUNS.length,
    holdout_var_count: 6,
  };
}

export function mockSeedDemo(): DemoSeedResult {
  return {
    upload_id: "mockupload01",
    model: DEMO_MODEL,
    weighted_r_squared: 0.78,
    n_rcts: 400,
    n_non_rcts: 50,
    donor_pool_band: "production",
    durations: { upload: 0.4, donor_pool: 1.2, labels: 0.6, features: 0.7, train: 90.1, holdouts: 12.4, portfolio: 0.9 },
  };
}

export function mockDashboardSummary(): DashboardSummary {
  const icpd_values = PORTFOLIO_RUNS.map((r) => r.predicted_icpd);
  const risk_counts: Record<string, number> = {
    severe: 0,
    high: 0,
    medium: 0,
    low: 0,
    unknown: 0,
  };
  for (const r of PORTFOLIO_RUNS) {
    const k = r.worst_segment_risk?.risk ?? "unknown";
    risk_counts[k] = (risk_counts[k] ?? 0) + 1;
  }
  const severity_counts: Record<string, number> = {
    severe: 1,
    high: 1,
    medium: 11,
    low: 5,
    unknown: 0,
  };

  return {
    donor_pool: {
      n_admitted: 450,
      band: "production",
      rationale:
        "450 RCTs admitted; 400–1599 → production mode. Expected weighted R² in the 0.72–0.81 range.",
    },
    latest_model: {
      ...DEMO_MODEL,
      weighted_r_squared: 0.78,
      bootstrap: { mean: 0.78, ci_lower: 0.71, ci_upper: 0.84 },
      ablation: ABLATION,
    },
    portfolio: {
      n_runs: PORTFOLIO_RUNS.length,
      mean_icpd: Number(
        (icpd_values.reduce((a, b) => a + b, 0) / icpd_values.length).toFixed(4),
      ),
      min_icpd: Math.min(...icpd_values),
      max_icpd: Math.max(...icpd_values),
      risk_counts,
      icpd_values,
    },
    holdouts: {
      n_levels: HOLDOUT_LEVELS.length,
      n_vars: 6,
      severity_counts,
    },
    n_models_total: 1,
    n_prediction_runs_total: PORTFOLIO_RUNS.length,
  };
}

export function mockPoolStatus(): PoolStatus {
  return {
    n_admitted: 450,
    band: "production",
    rationale:
      "450 RCTs admitted; 400–1599 → production mode. Expected weighted R² in the 0.72–0.81 range.",
  };
}

export function mockEligibleRcts(uploadId: string): EligibleResponse {
  return { upload_id: uploadId, rcts: ELIGIBLE_RCTS };
}

export function mockPromoteRcts(campaignIds: string[]): PromoteResponse {
  return { promoted: campaignIds, status: mockPoolStatus() };
}

export function mockDemoteRct(campaignId: string): DemoteResponse {
  return { demoted: { campaign_id: campaignId }, status: mockPoolStatus() };
}

export function mockCoverage(): CoverageResponse {
  const cells = [];
  const verticals = [...VERTICALS];
  const funnels = [...FUNNELS];
  const audiences = [...AUDIENCES];
  for (const v of verticals) {
    for (const f of funnels) {
      for (const a of audiences) {
        cells.push({
          vertical: v,
          funnel_stage: f,
          audience_type: a,
          count: Math.floor(rand(0, 12)),
        });
      }
    }
  }
  return {
    verticals: verticals as unknown as string[],
    funnels: funnels as unknown as string[],
    audiences: audiences as unknown as string[],
    cells,
  };
}

export function mockAging(): AgingResponse {
  return {
    total_admitted: 450,
    same_year: 312,
    one_year_ago: 98,
    older: 40,
    fraction_recent: 0.693,
    extrapolation_risk: "low",
    rationale:
      "Fraction of admitted RCTs from the same calendar year as today. Year-to-year drift carries a 21.3-ppt R² penalty (PDF Table 1).",
  };
}

export function mockShadowRcts(): { recommendations: ShadowRec[] } {
  return {
    recommendations: [
      {
        vertical: "media",
        funnel_stage: "upper",
        audience_type: "lookalike",
        gap_score: 1,
        status: "open",
        brief:
          "Donor pool coverage gap in media / upper / lookalike. Run a shadow RCT to fill this segment.",
      },
      {
        vertical: "finance",
        funnel_stage: "upper",
        audience_type: "prospecting",
        gap_score: 1,
        status: "open",
        brief:
          "Donor pool coverage gap in finance / upper / prospecting. Run a shadow RCT to fill this segment.",
      },
    ],
  };
}

export function mockGenerateLabels(uploadId: string): GenerateLabelsResponse {
  return { upload_id: uploadId, labels: RCT_LABELS };
}

export function mockBuildFeatures(
  uploadId: string,
  mode: "training" | "scoring",
): BuildFeaturesResponse {
  const x_pre_fields = [
    "objective", "vertical", "audience_type", "funnel_stage",
    "conversion_optimization", "custom_audience",
    "advertiser_platform_experience_months",
    "creative_format", "placement", "bid_strategy",
    "market", "spend_tier", "platform",
    "month", "quarter", "campaign_duration_days",
  ];
  const x_post_fields = [
    "exposure_rate", "ctr", "clicks_per_dollar", "conversions_per_dollar",
    "lcc_1h_per_dollar", "lcc_1d_per_dollar", "lcc_7d_per_dollar",
    "lcc_28d_per_dollar", "view_through_per_dollar", "avg_dwell_time",
  ];
  const rows = Array.from({ length: 30 }, (_, i) => ({
    campaign_id: `RCT-${String(i).padStart(4, "0")}`,
    feature_set_version: "v1",
    mode,
    sample_id: null,
    x_pre: {
      objective: "conversions",
      vertical: pick(VERTICALS),
      audience_type: pick(AUDIENCES),
      funnel_stage: pick(FUNNELS),
      conversion_optimization: pick(["yes", "no"]),
      custom_audience: pick(["yes", "no"]),
      advertiser_platform_experience_months: Math.floor(rand(6, 60)),
      creative_format: pick(CREATIVES),
      placement: pick(PLACEMENTS),
      bid_strategy: pick(["lowest_cost", "cost_cap", "bid_cap"]),
      market: pick(["US", "UK", "ID", "DE"]),
      spend_tier: pick(["low", "medium", "high"]),
      platform: pick(["meta", "tiktok", "google"]),
      month: Math.floor(rand(1, 6)),
      quarter: Math.floor(rand(1, 3)),
      campaign_duration_days: pick([14, 21, 28, 35]),
    },
    x_post: {
      exposure_rate: Number(rand(0.6, 0.95).toFixed(3)),
      ctr: Number(rand(0.012, 0.04).toFixed(4)),
      clicks_per_dollar: Number(rand(0.8, 4.0).toFixed(3)),
      conversions_per_dollar: Number(rand(0.5, 5.0).toFixed(3)),
      lcc_1h_per_dollar: Number(rand(0.005, 0.02).toFixed(4)),
      lcc_1d_per_dollar: Number(rand(0.01, 0.04).toFixed(4)),
      lcc_7d_per_dollar: Number(rand(0.05, 0.5).toFixed(3)),
      lcc_28d_per_dollar: Number(rand(0.05, 0.18).toFixed(4)),
      view_through_per_dollar: Number(rand(0.005, 0.03).toFixed(4)),
      avg_dwell_time: Number(rand(4, 18).toFixed(2)),
    },
    created_at: NOW,
  }));
  return {
    upload_id: uploadId,
    mode,
    feature_set_version: "v1",
    x_pre_fields,
    x_post_fields,
    rows,
  };
}

export function mockListModels(): { models: ModelRecord[] } {
  return { models: [DEMO_MODEL] };
}

export function mockTrainModel(): TrainResponse {
  return {
    model: DEMO_MODEL,
    n_observations: 400,
    weighted_r_squared: 0.78,
    bootstrap: { mean: 0.78, p025: 0.71, p975: 0.84, n_draws: 20 },
    r_squared_ceiling: 0.81,
    lcc_diagnostics: { slope: 1.42, spearman_rho: 0.71 },
    ablation: ABLATION.map((a) => ({ ...a, n_train: 300, n_test: 100 })),
    donor_pool_status: mockPoolStatus(),
  };
}

export function mockModelMetrics(modelId: string): { model_id: string; metrics: ModelMetric[] } {
  const metrics: ModelMetric[] = [
    { id: "m1", model_version_id: modelId, metric_type: "weighted_r_squared", value: 0.78, ci_lower: null, ci_upper: null, segment: null, created_at: NOW },
    { id: "m2", model_version_id: modelId, metric_type: "weighted_r_squared_bootstrap_mean", value: 0.78, ci_lower: 0.71, ci_upper: 0.84, segment: null, created_at: NOW },
    { id: "m3", model_version_id: modelId, metric_type: "r_squared_ceiling", value: 0.81, ci_lower: null, ci_upper: null, segment: null, created_at: NOW },
    { id: "m4", model_version_id: modelId, metric_type: "lcc_ols_slope", value: 1.42, ci_lower: null, ci_upper: null, segment: null, created_at: NOW },
    { id: "m5", model_version_id: modelId, metric_type: "lcc_spearman_rho", value: 0.71, ci_lower: null, ci_upper: null, segment: null, created_at: NOW },
    ...ABLATION.map((a, i) => ({
      id: `m${10 + i}`,
      model_version_id: modelId,
      metric_type: "ablation_weighted_r2",
      value: a.weighted_r2,
      ci_lower: null,
      ci_upper: null,
      segment: { spec: a.spec },
      created_at: NOW,
    })),
  ];
  return { model_id: modelId, metrics };
}

export function mockPromoteModel(): ModelRecord {
  return DEMO_MODEL;
}

export function mockHoldout(segmentationVar: string): HoldoutResponse {
  const results = HOLDOUT_LEVELS.filter((h) => h.segmentation_var === segmentationVar);
  return {
    segmentation_var: segmentationVar,
    n_iterations: 10,
    results,
  };
}

export function mockScoreCampaign(spec: Record<string, unknown>): PredictionRun {
  const run = _genPredictionRun(99999, false);
  run.spec = { ...run.spec, ...spec };
  return run;
}

export function mockListPredictions(): { runs: PredictionRun[] } {
  return { runs: PORTFOLIO_RUNS };
}

export function mockScorePortfolio(): PortfolioResponse {
  const icpds = PORTFOLIO_RUNS.map((r) => r.predicted_icpd);
  const sorted = [...icpds].sort((a, b) => a - b);
  const p10 = sorted[Math.floor(sorted.length * 0.1)];
  const p90 = sorted[Math.floor(sorted.length * 0.9)];
  const mean = icpds.reduce((a, b) => a + b, 0) / icpds.length;
  const variance =
    icpds.reduce((s, v) => s + (v - mean) ** 2, 0) / icpds.length;
  const stdev = Math.sqrt(variance);
  const risk_counts: Record<"severe" | "high" | "medium" | "low" | "unknown", number> = {
    severe: 0, high: 0, medium: 0, low: 0, unknown: 0,
  };
  let worst: SegmentRisk | null = null;
  for (const r of PORTFOLIO_RUNS) {
    const risk = r.worst_segment_risk?.risk ?? "unknown";
    risk_counts[risk] += 1;
    if (
      r.worst_segment_risk &&
      r.worst_segment_risk.penalty_pp !== null &&
      (worst === null ||
        (r.worst_segment_risk.penalty_pp ?? 0) > (worst.penalty_pp ?? 0))
    ) {
      worst = r.worst_segment_risk;
    }
  }
  return {
    model: DEMO_MODEL,
    runs: PORTFOLIO_RUNS,
    aggregates: {
      n: icpds.length,
      mean_icpd: Number(mean.toFixed(4)),
      median_icpd: Number(sorted[Math.floor(sorted.length / 2)].toFixed(4)),
      stdev_icpd: Number(stdev.toFixed(4)),
      p10_icpd: Number(p10.toFixed(4)),
      p90_icpd: Number(p90.toFixed(4)),
      risk_counts,
    },
    worst_segment_risk: worst,
    watermark: null,
  };
}

export function mockRecommendDecisions(): DecisionResponse {
  const recs: Recommendation[] = PORTFOLIO_RUNS.map((r, idx) => {
    const risk = r.worst_segment_risk?.risk ?? "unknown";
    const ci_lower = r.ci_lower ?? r.predicted_icpd;
    let action: Recommendation["action"];
    let rationale: string;
    if (risk === "severe") {
      action = "block";
      rationale = "Severe extrapolation risk on a segment level — training pool does not cover this campaign's regime.";
    } else if (risk === "high" || (ci_lower ?? 0) <= 0) {
      action = "deprioritize";
      rationale = "High extrapolation risk; predictions on this level have shown ≥15pp R² penalty.";
    } else if (risk === "medium") {
      action = idx % 4 === 0 ? "hold" : "hold";
      rationale = "Medium extrapolation risk (≥5pp penalty). Run a shadow RCT in this segment to upgrade trust.";
    } else if (idx < 12) {
      action = "promote";
      rationale = "Top-tercile risk-adjusted ICPD with low/unknown extrapolation risk.";
    } else {
      action = "hold";
      rationale = "Below top-tercile risk-adjusted ICPD; not promote-worthy but no risk flag fires.";
    }
    return {
      run_id: r.id,
      campaign_id: r.campaign_id,
      predicted_icpd: r.predicted_icpd,
      ci_lower: r.ci_lower,
      ci_upper: r.ci_upper,
      worst_risk: risk,
      risk_adjusted_score: ci_lower !== null ? Number((ci_lower * 0.85).toFixed(4)) : null,
      action,
      rationale,
      rank: idx + 1,
    };
  });
  recs.sort((a, b) => (b.risk_adjusted_score ?? 0) - (a.risk_adjusted_score ?? 0));
  recs.forEach((r, i) => (r.rank = i + 1));
  const action_counts: Record<Recommendation["action"], number> = {
    promote: 0,
    hold: 0,
    deprioritize: 0,
    block: 0,
  };
  for (const r of recs) action_counts[r.action] += 1;
  const followed = recs.filter((r) => r.action === "promote" || r.action === "hold");
  const naive = recs.reduce((s, r) => s + r.predicted_icpd, 0) / recs.length;
  const advised = followed.length
    ? followed.reduce((s, r) => s + r.predicted_icpd, 0) / followed.length
    : 0;
  return {
    recommendations: recs,
    action_counts,
    projected_lift: {
      naive_portfolio_icpd: Number(naive.toFixed(4)),
      advised_portfolio_icpd: Number(advised.toFixed(4)),
      lift_pp: Number(((advised - naive) * 100).toFixed(2)),
      n_followed: followed.length,
      n_total: recs.length,
      rationale:
        "Naive mean ICPD across all candidates vs. the mean across the campaigns the recommender keeps (promote + hold). Block and deprioritize are excluded.",
    },
    is_research_model: false,
    risk_floor: "low",
    portfolio: {
      model: DEMO_MODEL,
      aggregates: mockScorePortfolio().aggregates,
      watermark: null,
    },
  };
}

export function mockDrift(): DriftResponse {
  const featureSpecs: { name: string; kind: "numeric" | "categorical"; psi: number }[] = [
    { name: "objective", kind: "categorical", psi: 0.01 },
    { name: "vertical", kind: "categorical", psi: 0.18 },
    { name: "audience_type", kind: "categorical", psi: 0.04 },
    { name: "funnel_stage", kind: "categorical", psi: 0.02 },
    { name: "conversion_optimization", kind: "categorical", psi: 0.03 },
    { name: "custom_audience", kind: "categorical", psi: 0.02 },
    { name: "advertiser_platform_experience_months", kind: "numeric", psi: 0.07 },
    { name: "creative_format", kind: "categorical", psi: 0.03 },
    { name: "placement", kind: "categorical", psi: 0.02 },
    { name: "bid_strategy", kind: "categorical", psi: 0.02 },
    { name: "market", kind: "categorical", psi: 0.05 },
    { name: "spend_tier", kind: "categorical", psi: 0.02 },
    { name: "platform", kind: "categorical", psi: 0.04 },
    { name: "month", kind: "numeric", psi: 0.02 },
    { name: "quarter", kind: "numeric", psi: 0.02 },
    { name: "campaign_duration_days", kind: "numeric", psi: 0.05 },
    { name: "exposure_rate", kind: "numeric", psi: 0.04 },
    { name: "ctr", kind: "numeric", psi: 0.03 },
    { name: "clicks_per_dollar", kind: "numeric", psi: 0.06 },
    { name: "conversions_per_dollar", kind: "numeric", psi: 0.13 },
    { name: "lcc_1h_per_dollar", kind: "numeric", psi: 0.04 },
    { name: "lcc_1d_per_dollar", kind: "numeric", psi: 0.09 },
    { name: "lcc_7d_per_dollar", kind: "numeric", psi: 0.27 },
    { name: "lcc_28d_per_dollar", kind: "numeric", psi: 0.08 },
    { name: "view_through_per_dollar", kind: "numeric", psi: 0.04 },
    { name: "avg_dwell_time", kind: "numeric", psi: 0.05 },
  ];
  const drifts = featureSpecs
    .map((f) => ({
      feature: f.name,
      kind: f.kind,
      psi: f.psi,
      severity:
        f.psi >= 0.25
          ? ("severe" as const)
          : f.psi >= 0.1
            ? ("moderate" as const)
            : ("stable" as const),
      n_train: 400,
      n_score: 50,
      bins: Array.from({ length: 5 }, (_, i) => ({
        bin: `bin_${i}`,
        expected: Number(rand(0.1, 0.3).toFixed(3)),
        actual: Number(rand(0.05, 0.35).toFixed(3)),
        delta: Number(rand(-0.1, 0.1).toFixed(3)),
      })),
    }))
    .sort((a, b) => b.psi - a.psi);
  const severity_counts: Record<"stable" | "moderate" | "severe", number> = {
    stable: 0,
    moderate: 0,
    severe: 0,
  };
  for (const d of drifts) severity_counts[d.severity] += 1;
  return {
    feature_set_version: "v1",
    n_training_rows: 400,
    n_scoring_rows: 50,
    max_psi: drifts[0].psi,
    mean_psi: Number(
      (drifts.reduce((s, d) => s + d.psi, 0) / drifts.length).toFixed(4),
    ),
    severity_counts,
    verdict: severity_counts.severe > 0 ? "retrain_recommended" : "watch",
    rationale:
      "1 feature crossed the severe drift threshold (PSI ≥ 0.25). Retraining the model on a refreshed donor pool is recommended before serving production predictions.",
    drifts,
  };
}

export function mockSimulator(capMultiplier: number): SimulatorResponse {
  const allocations = PORTFOLIO_RUNS.map((r, i) => {
    const original_spend = Number((r.spec.cost as number) ?? 50_000);
    const risk = r.worst_segment_risk?.risk ?? "unknown";
    const blocked = risk === "severe";
    let proposed_spend = blocked
      ? 0
      : Math.min(original_spend * capMultiplier, original_spend * (1 + (i % 8 === 0 ? 0.5 : -0.1)));
    proposed_spend = Number(proposed_spend.toFixed(2));
    const delta = Number((proposed_spend - original_spend).toFixed(2));
    const delta_pct = original_spend > 0 ? Number(((delta / original_spend) * 100).toFixed(2)) : null;
    return {
      run_id: r.id,
      campaign_id: r.campaign_id,
      action: (blocked
        ? "block"
        : i % 6 === 0
          ? "promote"
          : i % 5 === 0
            ? "deprioritize"
            : "hold") as Recommendation["action"],
      worst_risk: risk,
      predicted_icpd: r.predicted_icpd,
      risk_adjusted_score: r.ci_lower !== null ? Number((r.ci_lower * 0.85).toFixed(4)) : null,
      original_spend,
      proposed_spend,
      delta_spend: delta,
      delta_pct,
      capped: !blocked && proposed_spend >= original_spend * capMultiplier - 1,
      original_ic: Number((r.predicted_icpd * original_spend).toFixed(2)),
      proposed_ic: Number((r.predicted_icpd * proposed_spend).toFixed(2)),
    };
  });
  const original_total = allocations.reduce((s, a) => s + a.original_spend, 0);
  const ic_orig = allocations.reduce((s, a) => s + a.original_ic, 0);
  const ic_new = allocations.reduce((s, a) => s + a.proposed_ic, 0);
  return {
    model: DEMO_MODEL,
    cap_multiplier: capMultiplier,
    risk_floor: "low",
    original_total_budget: Number(original_total.toFixed(2)),
    total_budget: Number(original_total.toFixed(2)),
    original_ic_total: Number(ic_orig.toFixed(2)),
    proposed_ic_total: Number(ic_new.toFixed(2)),
    ic_lift_pct: Number((((ic_new - ic_orig) / ic_orig) * 100).toFixed(2)),
    n_campaigns: allocations.length,
    n_blocked: allocations.filter((a) => a.action === "block").length,
    n_capped: allocations.filter((a) => a.capped).length,
    n_promoted: allocations.filter((a) => a.action === "promote").length,
    allocations,
    rationale: `Reallocated $${original_total.toLocaleString()} across ${allocations.length} campaigns using risk-adjusted weights with a cap of ${capMultiplier}× original spend per campaign.`,
  };
}

// --- Phase 1 / upload-pipeline mocks --------------------------------------

export function mockUpload(): UploadResponse {
  return {
    upload_id: "mockupload01",
    filename: "piemaker_demo.csv",
    rows: 450,
    columns: [
      "campaign_id", "advertiser_id", "is_rct", "objective", "vertical",
      "audience_type", "funnel_stage", "conversion_optimization",
      "custom_audience", "advertiser_platform_experience_months",
      "creative_format", "placement", "bid_strategy", "market",
      "spend_tier", "platform", "start_date", "end_date", "duration_days",
      "test_users", "control_users", "exposed_test_users",
      "test_conversions", "control_conversions", "cost",
      "clicks", "impressions", "conversions",
      "lcc_1h", "lcc_1d", "lcc_7d", "lcc_28d",
      "view_through_conversions", "avg_dwell_time",
    ],
  };
}

export function mockSchemaSuggest(uploadId: string): SchemaSuggestResponse {
  return {
    upload_id: uploadId,
    source_columns: ["campaign_id", "vertical", "test_users", "cost"],
    suggestions: [
      { source_column: "campaign_id", target_field: "campaign_id", confidence: 1.0, reason: "exact match" },
      { source_column: "vertical", target_field: "vertical", confidence: 1.0, reason: "exact match" },
      { source_column: "test_users", target_field: "test_users", confidence: 1.0, reason: "exact match" },
      { source_column: "cost", target_field: "cost", confidence: 1.0, reason: "exact match" },
    ],
  };
}

export function mockValidate(uploadId: string): ValidateResponse {
  return {
    upload_id: uploadId,
    data_quality_score: 92,
    block_training: false,
    severity_breakdown: { critical: 0, warning: 2, info: 1 },
    rules: [
      { rule_id: "R001", severity: "info", description: "All campaign_ids unique", passed: true, affected_rows: [], fix_suggestion: null, paper_reference: null },
      { rule_id: "R002", severity: "warning", description: "3 rows with low test_users (<500K)", passed: false, affected_rows: [12, 47, 209], fix_suggestion: "Filter or merge low-volume RCTs", paper_reference: "§3.2" },
      { rule_id: "R003", severity: "warning", description: "1 row with control_users == 0", passed: false, affected_rows: [88], fix_suggestion: "Drop rows lacking a control arm", paper_reference: "§3.1" },
      { rule_id: "R004", severity: "info", description: "Date range 2026-01-15 to 2026-05-29", passed: true, affected_rows: [], fix_suggestion: null, paper_reference: null },
    ],
  };
}

export function mockClean(uploadId: string): CleanResponse {
  return {
    upload_id: uploadId,
    rows_in: 450,
    rows_out: 446,
    cleaning_actions: [
      { action_type: "drop_low_volume_rct", rows_affected: 3, before_summary: { count: 450 }, after_summary: { count: 447 }, applied_at: NOW, applied_by: "demo", notes: "Filtered RCTs with test_users < 500K" },
      { action_type: "drop_zero_control", rows_affected: 1, before_summary: { count: 447 }, after_summary: { count: 446 }, applied_at: NOW, applied_by: "demo", notes: "Removed campaign with no control arm" },
    ],
    mc_defense: RCT_LABELS.slice(0, 8).map((l) => ({
      campaign_id: l.campaign_id,
      mc_defense_mode: l.mc_defense_mode,
      sample_split_seed: l.sample_split_seed,
      reason:
        l.mc_defense_mode === "blocked"
          ? "Insufficient test pool for sample-split (PDF §4.4)"
          : l.mc_defense_mode === "sample_split"
            ? "User-level data available; sample_split active"
            : "No user-level data; using shared-sample compromise",
    })),
  };
}

// ============================================================================
// V.4 Wave 2 — diagnostics mocks
// ============================================================================

export function mockCalibration(segmentationVar: string): {
  segmentation_var: string;
  n_levels: number;
  results: Array<{
    segmentation_var: string;
    level: string;
    n: number;
    bias_ratio: number | null;
    ols_slope: number | null;
    spearman_rho: number | null;
    raw_lcc_r2: number | null;
    residual_mean: number;
    residual_p10: number;
    residual_p90: number;
  }>;
} {
  const levelsByVar: Record<string, string[]> = {
    vertical: ["ecommerce", "travel", "finance", "media"],
    audience_type: ["retargeting", "prospecting", "lookalike"],
    funnel_stage: ["upper", "mid", "lower"],
    advertiser_size: ["smb", "mid_market", "enterprise"],
    campaign_year: ["2024", "2025", "2026"],
  };
  const levels = levelsByVar[segmentationVar] ?? ["a", "b", "c"];
  // Paper-shape: LCC overstates lift, so bias_ratio > 1 in most segments.
  const results = levels.map((level, i) => {
    const baseBias = 1.15 + (i % 3) * 0.18;
    const n = 12 + (i * 7) % 23;
    return {
      segmentation_var: segmentationVar,
      level,
      n,
      bias_ratio: Number(baseBias.toFixed(3)),
      ols_slope: Number((0.65 + 0.05 * i).toFixed(3)),
      spearman_rho: Number((0.42 + 0.04 * i).toFixed(3)),
      raw_lcc_r2: Number((0.22 + 0.03 * i).toFixed(3)),
      residual_mean: Number((-0.018 - 0.004 * i).toFixed(4)),
      residual_p10: Number((-0.082 - 0.005 * i).toFixed(4)),
      residual_p90: Number((0.041 + 0.004 * i).toFixed(4)),
    };
  });
  return {
    segmentation_var: segmentationVar,
    n_levels: results.length,
    results,
  };
}

export function mockSampleSizeCurve(): {
  n_points: number;
  points: Array<{
    pool_size: number;
    weighted_r2_median: number;
    weighted_r2_p025: number;
    weighted_r2_p975: number;
    n_subsamples: number;
    n_splits: number;
  }>;
} {
  // Paper Figure 6 shape: rises from ~0.37 at n=50, plateaus near 0.88 at n=1600.
  const targets = [
    { pool_size: 50, r2: 0.37, ci: 0.10 },
    { pool_size: 100, r2: 0.55, ci: 0.08 },
    { pool_size: 200, r2: 0.66, ci: 0.06 },
    { pool_size: 400, r2: 0.74, ci: 0.05 },
    { pool_size: 800, r2: 0.81, ci: 0.04 },
    { pool_size: 1600, r2: 0.86, ci: 0.03 },
  ];
  const points = targets.map((t) => ({
    pool_size: t.pool_size,
    weighted_r2_median: t.r2,
    weighted_r2_p025: Number((t.r2 - t.ci).toFixed(3)),
    weighted_r2_p975: Number((t.r2 + t.ci).toFixed(3)),
    n_subsamples: 5,
    n_splits: 5,
  }));
  return { n_points: points.length, points };
}

export function mockAdvertiserCv(): {
  n_total: number;
  n_splits: number;
  cohorts: Array<{
    cohort: "existing" | "new";
    n: number;
    weighted_r2: number;
    n_advertisers: number;
  }>;
  cohort_gap_pp: number;
} {
  // Paper §5.3 cold-start finding: new advertisers lose ~12pp of R²
  // because the model has never seen their idiosyncratic baseline.
  return {
    n_total: 450,
    n_splits: 5,
    cohorts: [
      { cohort: "existing", n: 380, weighted_r2: 0.79, n_advertisers: 76 },
      { cohort: "new", n: 70, weighted_r2: 0.67, n_advertisers: 70 },
    ],
    cohort_gap_pp: 12.0,
  };
}

function _mockBootstrapDistribution(
  mean: number,
  std: number,
  n: number,
  seed = 71,
): number[] {
  // Deterministic-ish "bootstrap": fixed permutation around the mean. Not
  // a real sample, just shape-correct values for chart rendering.
  const out: number[] = [];
  let state = seed;
  for (let i = 0; i < n; i++) {
    state = (state * 1103515245 + 12345) % 2 ** 31;
    const u = state / 2 ** 31;
    // Box-Muller half (approximate enough for visuals).
    const z = Math.sqrt(-2 * Math.log(u || 1e-9)) * Math.cos(2 * Math.PI * u);
    out.push(Number((mean + std * z).toFixed(4)));
  }
  return out;
}

export function mockBootstrapAdvertisers(): {
  n_draws: number;
  n_advertisers: number;
  mean: number;
  p025: number;
  p975: number;
  distribution: number[];
} {
  const distribution = _mockBootstrapDistribution(0.78, 0.045, 100, 71);
  const sorted = [...distribution].sort((a, b) => a - b);
  return {
    n_draws: distribution.length,
    n_advertisers: 76,
    mean: 0.78,
    p025: Number(sorted[Math.floor(sorted.length * 0.025)].toFixed(3)),
    p975: Number(sorted[Math.floor(sorted.length * 0.975)].toFixed(3)),
    distribution,
  };
}

export function mockHoldoutDistributions(): {
  distributions: Array<{
    id: string;
    segmentation_var: string;
    level: string;
    within_r2_median: number;
    extrapolation_r2_median: number;
    penalty_pp: number;
    n_iterations: number;
    risk: string;
    within_r2_dist: number[];
    extrapolation_r2_dist: number[];
    penalty_pp_dist: number[];
    within_r2_p10: number;
    within_r2_p90: number;
    extrapolation_r2_p10: number;
    extrapolation_r2_p90: number;
    penalty_pp_p10: number;
    penalty_pp_p90: number;
  }>;
  count: number;
} {
  // Match the (var, level) shape Wave 2 hold-out test produces, with
  // realistic dispersion so the box-plots render with whisker variety.
  const seeds: Array<{
    seg: string;
    level: string;
    within: number;
    extrap: number;
    risk: string;
  }> = [
    { seg: "vertical", level: "ecommerce", within: 0.80, extrap: 0.74, risk: "medium" },
    { seg: "vertical", level: "travel", within: 0.77, extrap: 0.62, risk: "medium" },
    { seg: "vertical", level: "finance", within: 0.74, extrap: 0.51, risk: "high" },
    { seg: "vertical", level: "media", within: 0.70, extrap: 0.41, risk: "severe" },
    { seg: "audience_type", level: "retargeting", within: 0.81, extrap: 0.78, risk: "low" },
    { seg: "audience_type", level: "prospecting", within: 0.74, extrap: 0.62, risk: "medium" },
    { seg: "audience_type", level: "lookalike", within: 0.71, extrap: 0.58, risk: "medium" },
    { seg: "advertiser_size", level: "smb", within: 0.71, extrap: 0.58, risk: "medium" },
    { seg: "advertiser_size", level: "mid_market", within: 0.78, extrap: 0.71, risk: "medium" },
    { seg: "advertiser_size", level: "enterprise", within: 0.81, extrap: 0.77, risk: "low" },
    { seg: "campaign_year", level: "2024", within: 0.75, extrap: 0.54, risk: "high" },
    { seg: "campaign_year", level: "2025", within: 0.78, extrap: 0.71, risk: "medium" },
    { seg: "campaign_year", level: "2026", within: 0.81, extrap: 0.76, risk: "low" },
  ];
  let bs = 91;
  const distributions = seeds.map((s) => {
    const withinDist = _mockBootstrapDistribution(s.within, 0.025, 20, bs++);
    const extrapDist = _mockBootstrapDistribution(s.extrap, 0.04, 20, bs++);
    const penaltyDist = withinDist.map((w, i) => Number(((w - extrapDist[i]) * 100).toFixed(2)));
    const sortedPenalty = [...penaltyDist].sort((a, b) => a - b);
    const sortedW = [...withinDist].sort((a, b) => a - b);
    const sortedE = [...extrapDist].sort((a, b) => a - b);
    return {
      id: `${s.seg}|${s.level}`,
      segmentation_var: s.seg,
      level: s.level,
      within_r2_median: s.within,
      extrapolation_r2_median: s.extrap,
      penalty_pp: Number(((s.within - s.extrap) * 100).toFixed(2)),
      n_iterations: 20,
      risk: s.risk,
      within_r2_dist: withinDist,
      extrapolation_r2_dist: extrapDist,
      penalty_pp_dist: penaltyDist,
      within_r2_p10: sortedW[2],
      within_r2_p90: sortedW[17],
      extrapolation_r2_p10: sortedE[2],
      extrapolation_r2_p90: sortedE[17],
      penalty_pp_p10: sortedPenalty[2],
      penalty_pp_p90: sortedPenalty[17],
    };
  });
  return { distributions, count: distributions.length };
}

// ============================================================================
// V.4 Wave 3 — decision-disagreement curves
// ============================================================================

function _scanRatios(low = 0.5, high = 1.5, step = 0.05): number[] {
  const n = Math.round((high - low) / step) + 1;
  return Array.from({ length: n }, (_, i) => Number((low + i * step).toFixed(2)));
}

function _curveAt(
  ratios: number[],
  // Disagreement minimum near 1.0 ratio (segment median is where the decision
  // boundary is most ambiguous), rises toward the tails. Paper-shape.
  base: number,
  refMedian: number,
  type1Bias: number,
  type2Bias: number,
) {
  return ratios.map((r) => {
    const distFromCenter = Math.abs(r - 1.0);
    const d = Math.max(0.02, base + 0.18 * distFromCenter);
    const type1 = Math.max(0, d * type1Bias);
    const type2 = Math.max(0, d - type1);
    return {
      threshold_ratio: r,
      threshold: Number((r * refMedian).toFixed(4)),
      disagreement: Number(d.toFixed(4)),
      type_1: Number(type1.toFixed(4)),
      type_2: Number(type2.toFixed(4)),
    };
  });
}

export function mockDecisionCurves(): {
  curve: ReturnType<typeof _curveAt>;
  expected_cost_curve: null;
  reference_median: number;
  low_ratio: number;
  high_ratio: number;
  step: number;
} {
  const ratios = _scanRatios();
  return {
    curve: _curveAt(ratios, 0.10, 0.085, 0.5, 0.5),
    expected_cost_curve: null,
    reference_median: 0.085,
    low_ratio: 0.5,
    high_ratio: 1.5,
    step: 0.05,
  };
}

export function mockDecisionCurvesCompare(): {
  reference_median: number;
  low_ratio: number;
  high_ratio: number;
  step: number;
  pie: ReturnType<typeof _curveAt>;
  raw_lcc: ReturnType<typeof _curveAt>;
} {
  const ratios = _scanRatios();
  // Paper baselines: PIE 8–12% disagreement vs RCT, LCC 12–20%.
  return {
    reference_median: 0.085,
    low_ratio: 0.5,
    high_ratio: 1.5,
    step: 0.05,
    pie: _curveAt(ratios, 0.09, 0.085, 0.45, 0.55),
    raw_lcc: _curveAt(ratios, 0.17, 0.085, 0.65, 0.35),
  };
}
