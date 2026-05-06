// Shared types between frontend and backend.
// Phase 0 placeholder — populated by Phase 1+ as endpoints come online.

export type Phase = 0 | 1 | 2 | 3 | 4;

export interface HealthResponse {
  status: "ok";
  phase: string;
  formulas_version: string;
}

export interface InfoResponse {
  app: string;
  env: string;
  frozen_formula_count: number;
}

export type ReliabilityBand =
  | "high"          // 80-100
  | "medium"        // 60-79
  | "low"           // 40-59
  | "experiment";   // < 40 — forces "Run RCT"

export type DecisionRecommendation = "Scale" | "Optimize" | "Pause" | "Run RCT";
