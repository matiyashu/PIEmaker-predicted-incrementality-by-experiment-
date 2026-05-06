"""
Validation engine (PDF §4.1, §4.3, Prompt 1.1).

Runs Pandera schema validation, then a sequence of business rules. Each rule
returns a structured result:
    {rule_id, severity, affected_rows, fix_suggestion, paper_reference}
Severity ∈ {"critical", "warning", "info"}.

Critical failures set block_training=True on the validation_runs row.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import pandera as pa

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from schemas.pie_upload import (  # noqa: E402
    MIN_TEST_CONVERSIONS,
    MIN_TEST_USERS,
    pie_upload_schema,
)


@dataclass
class RuleResult:
    rule_id: str
    severity: str  # "critical" | "warning" | "info"
    description: str
    passed: bool
    affected_rows: list[int] = field(default_factory=list)
    fix_suggestion: str | None = None
    paper_reference: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    data_quality_score: int
    block_training: bool
    severity_breakdown: dict[str, int]
    rules: list[dict]

    def to_dict(self) -> dict:
        return {
            "data_quality_score": self.data_quality_score,
            "block_training": self.block_training,
            "severity_breakdown": self.severity_breakdown,
            "rules": self.rules,
        }


# --- Pandera shape rule -----------------------------------------------------


def _run_pandera(df: pd.DataFrame) -> RuleResult:
    try:
        pie_upload_schema.validate(df, lazy=True)
        return RuleResult(
            rule_id="pandera_schema",
            severity="info",
            description="DataFrame matches the PIE upload schema.",
            passed=True,
        )
    except pa.errors.SchemaErrors as exc:
        rows: set[int] = set()
        failures = exc.failure_cases
        if failures is not None and "index" in failures.columns:
            rows = {int(i) for i in failures["index"].dropna().tolist()}
        return RuleResult(
            rule_id="pandera_schema",
            severity="critical",
            description="DataFrame does not match the PIE upload schema.",
            passed=False,
            affected_rows=sorted(rows),
            fix_suggestion=(
                "Review the column types, allowed categorical values, and "
                "DataFrame-level checks (funnel monotonicity, date validity, "
                "RCT admission thresholds)."
            ),
            paper_reference="PDF Appendix B; §4.1 hard exclusions",
        )


# --- §4.1 hard exclusions --------------------------------------------------


def _rule_test_users_threshold(df: pd.DataFrame) -> RuleResult:
    rct = df[df["is_rct"] == 1]
    if rct.empty:
        return RuleResult(
            "rct_test_users_min", "info",
            "No RCT rows; threshold check skipped.", True,
        )
    bad = rct[rct["test_users"].fillna(0) < MIN_TEST_USERS]
    return RuleResult(
        rule_id="rct_test_users_min",
        severity="critical" if not bad.empty else "info",
        description=(
            f"RCT donor-pool admission requires test_users ≥ {MIN_TEST_USERS:,}."
        ),
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion="Exclude under-powered RCTs or upload a larger sample.",
        paper_reference="Gordon et al. 2026, p. 17 (eligibility cutoff)",
    )


def _rule_test_conversions_threshold(df: pd.DataFrame) -> RuleResult:
    rct = df[df["is_rct"] == 1]
    if rct.empty:
        return RuleResult(
            "rct_test_conversions_min", "info",
            "No RCT rows; threshold check skipped.", True,
        )
    bad = rct[rct["test_conversions"].fillna(0) < MIN_TEST_CONVERSIONS]
    return RuleResult(
        rule_id="rct_test_conversions_min",
        severity="critical" if not bad.empty else "info",
        description=(
            f"RCT donor-pool admission requires test_conversions ≥ "
            f"{MIN_TEST_CONVERSIONS:,}."
        ),
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion=(
            "Exclude RCTs with low conversion volume or aggregate cells with a "
            "comparable design."
        ),
        paper_reference="Gordon et al. 2026, p. 17 (eligibility cutoff)",
    )


def _rule_cost_validity(df: pd.DataFrame) -> RuleResult:
    bad = df[df["cost"] <= 0]
    return RuleResult(
        rule_id="cost_positive",
        severity="critical" if not bad.empty else "info",
        description="Every row must have cost > 0 (ICPD denominator).",
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion="Remove or correct rows with non-positive cost.",
        paper_reference="PDF §4.1; Eq. 23 (ICPD)",
    )


def _rule_funnel_monotonicity(df: pd.DataFrame) -> RuleResult:
    bad = df[
        ~((df["impressions"] >= df["clicks"]) & (df["clicks"] >= df["conversions"]))
    ]
    return RuleResult(
        rule_id="funnel_monotonicity",
        severity="critical" if not bad.empty else "info",
        description="impressions ≥ clicks ≥ conversions on every row.",
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion=(
            "Verify your attribution counter — clicks cannot exceed impressions; "
            "conversions cannot exceed clicks."
        ),
        paper_reference="PDF §4.1 (funnel monotonicity)",
    )


def _rule_date_validity(df: pd.DataFrame) -> RuleResult:
    bad = df[df["start_date"] >= df["end_date"]]
    return RuleResult(
        rule_id="date_validity",
        severity="critical" if not bad.empty else "info",
        description="start_date < end_date on every row.",
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion="Correct campaign dates so start_date precedes end_date.",
        paper_reference="PDF §4.1",
    )


def _rule_control_exposure_zero(df: pd.DataFrame) -> RuleResult:
    """RCT control group must have zero exposure (one-sided noncompliance)."""
    rct = df[df["is_rct"] == 1].copy()
    if rct.empty:
        return RuleResult(
            "rct_control_exposure_zero", "info",
            "No RCT rows; control-exposure check skipped.", True,
        )
    if "control_exposed_users" not in rct.columns:
        return RuleResult(
            "rct_control_exposure_zero", "info",
            "control_exposed_users not provided; assumed zero.", True,
        )
    bad = rct[rct["control_exposed_users"].fillna(0) > 0]
    return RuleResult(
        rule_id="rct_control_exposure_zero",
        severity="critical" if not bad.empty else "info",
        description=(
            "RCT control group exposure must be zero (one-sided noncompliance)."
        ),
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion=(
            "If control users were exposed, the RCT does not meet the ATT "
            "estimator assumption (Eq. 14)."
        ),
        paper_reference="Gordon et al. 2026, p. 11 (ATT estimator assumption)",
    )


def _rule_exposure_rate_validity(df: pd.DataFrame) -> RuleResult:
    rct = df[df["is_rct"] == 1].copy()
    if rct.empty:
        return RuleResult(
            "rct_exposure_rate_validity", "info",
            "No RCT rows; exposure-rate check skipped.", True,
        )
    rct["_exp_rate"] = rct["exposed_test_users"] / rct["test_users"]
    bad = rct[(rct["_exp_rate"] <= 0) | (rct["_exp_rate"] > 1)]
    return RuleResult(
        rule_id="rct_exposure_rate_validity",
        severity="critical" if not bad.empty else "info",
        description="0 < exposed_test_users / test_users ≤ 1.",
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion=(
            "Verify test_users and exposed_test_users; exposure rate outside "
            "(0, 1] breaks the ATT denominator."
        ),
        paper_reference="Gordon et al. 2026, p. 11",
    )


# --- §4.3 LCC tiered requirements ------------------------------------------


def _rule_lcc_7d_required(df: pd.DataFrame) -> RuleResult:
    bad = df[df["lcc_7d"].isnull()]
    return RuleResult(
        rule_id="lcc_7d_required",
        severity="critical" if not bad.empty else "info",
        description="lcc_7d is required (drives 96.6% of model R² lift).",
        passed=bad.empty,
        affected_rows=bad.index.tolist(),
        fix_suggestion="Pull a 7-day click-attribution view from your platform.",
        paper_reference="Gordon et al. 2026, Figure 2 (p. 22)",
    )


def _rule_lcc_1d_recommended(df: pd.DataFrame) -> RuleResult:
    if "lcc_1d" not in df.columns:
        bad_idx = df.index.tolist()
    else:
        bad_idx = df[df["lcc_1d"].isnull()].index.tolist()
    return RuleResult(
        rule_id="lcc_1d_recommended",
        severity="warning" if bad_idx else "info",
        description="lcc_1d is strongly recommended; missing values reduce R².",
        passed=not bad_idx,
        affected_rows=bad_idx,
        fix_suggestion="If your platform exposes 1-day click attribution, include it.",
        paper_reference="PDF §4.3",
    )


def _rule_lcc_28d_recommended(df: pd.DataFrame) -> RuleResult:
    if "lcc_28d" not in df.columns:
        bad_idx = df.index.tolist()
    else:
        bad_idx = df[df["lcc_28d"].isnull()].index.tolist()
    return RuleResult(
        rule_id="lcc_28d_recommended",
        severity="warning" if bad_idx else "info",
        description=(
            "lcc_28d is strongly recommended for long-cycle conversions "
            "(travel, finance)."
        ),
        passed=not bad_idx,
        affected_rows=bad_idx,
        fix_suggestion="If your platform exposes 28-day click attribution, include it.",
        paper_reference="PDF §4.3",
    )


def _rule_lcc_1h_optional(df: pd.DataFrame) -> RuleResult:
    if "lcc_1h" not in df.columns or df["lcc_1h"].isnull().all():
        return RuleResult(
            rule_id="lcc_1h_optional",
            severity="info",
            description="lcc_1h not provided; optional and rarely available.",
            passed=True,
        )
    return RuleResult(
        rule_id="lcc_1h_optional", severity="info",
        description="lcc_1h provided.", passed=True,
    )


# --- Orchestration ---------------------------------------------------------


_RULES = [
    _run_pandera,
    _rule_test_users_threshold,
    _rule_test_conversions_threshold,
    _rule_cost_validity,
    _rule_funnel_monotonicity,
    _rule_date_validity,
    _rule_control_exposure_zero,
    _rule_exposure_rate_validity,
    _rule_lcc_7d_required,
    _rule_lcc_1d_recommended,
    _rule_lcc_28d_recommended,
    _rule_lcc_1h_optional,
]


def _data_quality_score(rules: Iterable[RuleResult]) -> int:
    """0-100 score: critical = -25, warning = -5, info = 0."""
    score = 100
    for r in rules:
        if r.passed:
            continue
        if r.severity == "critical":
            score -= 25
        elif r.severity == "warning":
            score -= 5
    return max(0, score)


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Run the full validation pipeline against an uploaded DataFrame."""
    results: list[RuleResult] = [rule(df) for rule in _RULES]
    severity_breakdown = {"critical": 0, "warning": 0, "info": 0}
    block_training = False
    for r in results:
        if r.passed:
            continue
        severity_breakdown[r.severity] = severity_breakdown.get(r.severity, 0) + 1
        if r.severity == "critical":
            block_training = True
    return ValidationResult(
        data_quality_score=_data_quality_score(results),
        block_training=block_training,
        severity_breakdown=severity_breakdown,
        rules=[r.to_dict() for r in results],
    )
