"""Initial schema — 15 tables (PDF §6.2).

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-06

Tables:
  Source data:    campaigns, experiments, experiment_cells,
                  campaign_metrics_daily, attribution_metrics
  Computed:       rct_labels, feature_store
  Operational:    validation_runs, cleaning_actions, monitoring_snapshots
  Registry:       model_versions, model_metrics
  Output:         prediction_runs, shadow_rct_recommendations
  Config:         decision_thresholds
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # --- Source data --------------------------------------------------------
    op.create_table(
        "campaigns",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("advertiser_id", sa.String(length=128), nullable=False),
        sa.Column("campaign_id_external", sa.String(length=256), nullable=False),
        sa.Column("campaign_name", sa.String(length=512), nullable=False),
        sa.Column("is_rct", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("vertical", sa.String(length=64), nullable=False),
        sa.Column("funnel_stage", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.String(length=64), nullable=False),
        sa.Column("audience_type", sa.String(length=32), nullable=False),
        sa.Column("conversion_optimization", sa.String(length=32), nullable=True),
        sa.Column("custom_audience", sa.String(length=32), nullable=True),
        sa.Column("advertiser_platform_experience_months", sa.Integer(), nullable=True),
        sa.Column("prior_rct_count", sa.Integer(), nullable=True),
        sa.Column("market", sa.String(length=64), nullable=True),
        sa.Column("spend_tier", sa.String(length=32), nullable=True),
        sa.Column("platform", sa.String(length=64), nullable=True),
        sa.Column("placement", sa.String(length=64), nullable=True),
        sa.Column("creative_format", sa.String(length=64), nullable=True),
        sa.Column("bid_strategy", sa.String(length=64), nullable=True),
        sa.Column("budget", sa.Numeric(18, 2), nullable=True),
        sa.Column("cost", sa.Numeric(18, 2), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("advertiser_id", "campaign_id_external", name="uq_campaigns_advertiser_extid"),
        sa.CheckConstraint("cost > 0", name="ck_campaigns_cost_positive"),
        sa.CheckConstraint("start_date < end_date", name="ck_campaigns_date_validity"),
    )
    op.create_index("ix_campaigns_advertiser_id", "campaigns", ["advertiser_id"])
    op.create_index("ix_campaigns_vertical_funnel", "campaigns", ["vertical", "funnel_stage"])
    op.create_index("ix_campaigns_is_rct", "campaigns", ["is_rct"])

    op.create_table(
        "experiments",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("design_notes", sa.Text(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("start_date < end_date", name="ck_experiments_date_validity"),
    )
    op.create_index("ix_experiments_campaign_id", "experiments", ["campaign_id"])

    op.create_table(
        "experiment_cells",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "experiment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cell_type", sa.String(length=16), nullable=False),  # 'test' | 'control'
        sa.Column("users", sa.BigInteger(), nullable=False),
        sa.Column("exposed_users", sa.BigInteger(), nullable=False),
        sa.Column("conversions", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("users > 0", name="ck_cells_users_positive"),
        sa.CheckConstraint("exposed_users >= 0", name="ck_cells_exposed_nonneg"),
        sa.CheckConstraint("conversions >= 0", name="ck_cells_conv_nonneg"),
        sa.CheckConstraint(
            "cell_type IN ('test', 'control')", name="ck_cells_type_enum"
        ),
    )
    op.create_index("ix_cells_experiment_id", "experiment_cells", ["experiment_id"])

    op.create_table(
        "campaign_metrics_daily",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("impressions", sa.BigInteger(), nullable=False),
        sa.Column("clicks", sa.BigInteger(), nullable=False),
        sa.Column("conversions", sa.BigInteger(), nullable=False),
        sa.Column("spend", sa.Numeric(18, 2), nullable=False),
        sa.UniqueConstraint("campaign_id", "date", name="uq_metrics_campaign_date"),
        sa.CheckConstraint("impressions >= clicks", name="ck_metrics_funnel_imp_click"),
        sa.CheckConstraint("clicks >= conversions", name="ck_metrics_funnel_click_conv"),
    )
    op.create_index("ix_metrics_campaign_date", "campaign_metrics_daily", ["campaign_id", "date"])

    op.create_table(
        "attribution_metrics",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("lcc_1h", sa.BigInteger(), nullable=True),
        sa.Column("lcc_1d", sa.BigInteger(), nullable=True),
        sa.Column("lcc_7d", sa.BigInteger(), nullable=False),  # required per §4.3
        sa.Column("lcc_28d", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- Computed -----------------------------------------------------------
    op.create_table(
        "rct_labels",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("att", sa.Float(), nullable=False),
        sa.Column("incremental_conversions", sa.Float(), nullable=False),
        sa.Column("icpd", sa.Float(), nullable=False),
        sa.Column("exposure_rate", sa.Float(), nullable=False),
        # Mechanical-correlation defense mode (§4.4): sample_split | shared_sample_compromise | blocked
        sa.Column("mc_defense_mode", sa.String(length=32), nullable=False),
        sa.Column("sample_split_seed", sa.BigInteger(), nullable=True),
        sa.Column("admitted_to_donor_pool", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "mc_defense_mode IN ('sample_split', 'shared_sample_compromise', 'blocked')",
            name="ck_rct_labels_mc_mode",
        ),
    )
    op.create_index("ix_rct_labels_admitted", "rct_labels", ["admitted_to_donor_pool"])

    op.create_table(
        "feature_store",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feature_set_version", sa.String(length=32), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),  # 'training' | 'scoring'
        sa.Column("x_pre", JSONB(), nullable=False),
        sa.Column("x_post", JSONB(), nullable=False),
        sa.Column("sample_id", sa.String(length=16), nullable=True),  # 'sample_1' | 'sample_2' | NULL
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "campaign_id",
            "feature_set_version",
            "mode",
            name="uq_feature_store_campaign_version_mode",
        ),
        sa.CheckConstraint("mode IN ('training', 'scoring')", name="ck_feature_store_mode"),
    )
    op.create_index("ix_feature_store_version", "feature_store", ["feature_set_version"])

    # --- Operational --------------------------------------------------------
    op.create_table(
        "validation_runs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("upload_id", sa.String(length=128), nullable=False),
        sa.Column("data_quality_score", sa.Integer(), nullable=False),
        sa.Column("severity_breakdown", JSONB(), nullable=False),
        sa.Column("rule_results", JSONB(), nullable=False),
        sa.Column("block_training", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_by", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_validation_runs_upload", "validation_runs", ["upload_id"])

    op.create_table(
        "cleaning_actions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("upload_id", sa.String(length=128), nullable=False),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("rows_affected", sa.Integer(), nullable=False),
        sa.Column("before_summary", JSONB(), nullable=True),
        sa.Column("after_summary", JSONB(), nullable=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("applied_by", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_cleaning_actions_upload", "cleaning_actions", ["upload_id"])

    op.create_table(
        "monitoring_snapshots",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("model_version_id", UUID(as_uuid=True), nullable=True),
        sa.Column("snapshot_type", sa.String(length=32), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "snapshot_type IN ('data_drift', 'segment_drift', 'prediction_drift', "
            "'error_drift', 'concept_drift', 'donor_pool_aging')",
            name="ck_monitoring_type",
        ),
    )
    op.create_index("ix_monitoring_type_time", "monitoring_snapshots", ["snapshot_type", "created_at"])

    # --- Registry -----------------------------------------------------------
    op.create_table(
        "model_versions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version_tag", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),  # research | production
        sa.Column("algorithm", sa.String(length=64), nullable=False),  # random_forest | xgboost | ...
        sa.Column("feature_set_version", sa.String(length=32), nullable=False),
        sa.Column("hyperparameters", JSONB(), nullable=False),
        sa.Column("training_donor_pool_size", sa.Integer(), nullable=False),
        sa.Column("concept_drift_baseline", JSONB(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.UniqueConstraint("name", "version_tag", name="uq_model_versions_name_tag"),
        sa.CheckConstraint(
            "status IN ('research', 'production', 'archived')",
            name="ck_model_versions_status",
        ),
    )
    op.create_index("ix_model_versions_status", "model_versions", ["status"])

    op.create_table(
        "model_metrics",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "model_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric_type", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("ci_lower", sa.Float(), nullable=True),
        sa.Column("ci_upper", sa.Float(), nullable=True),
        sa.Column("segment", JSONB(), nullable=True),  # {vertical, funnel, ...}
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_model_metrics_model", "model_metrics", ["model_version_id"])
    op.create_index("ix_model_metrics_type", "model_metrics", ["metric_type"])

    # --- Output -------------------------------------------------------------
    op.create_table(
        "prediction_runs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "model_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("predicted_icpd", sa.Float(), nullable=False),
        sa.Column("predicted_ic", sa.Float(), nullable=False),
        sa.Column("predicted_cpic", sa.Float(), nullable=True),  # NULL when ICPD <= 0 (hard-block)
        sa.Column("predicted_inc_revenue", sa.Float(), nullable=True),
        sa.Column("predicted_iroas", sa.Float(), nullable=True),
        sa.Column("reliability_score", sa.Integer(), nullable=False),
        sa.Column("reliability_components", JSONB(), nullable=False),
        sa.Column("decision_recommendation", sa.String(length=16), nullable=True),
        sa.Column("notes", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "reliability_score BETWEEN 0 AND 100", name="ck_pred_runs_rel_range"
        ),
    )
    op.create_index("ix_pred_runs_model", "prediction_runs", ["model_version_id"])
    op.create_index("ix_pred_runs_campaign", "prediction_runs", ["campaign_id"])

    op.create_table(
        "shadow_rct_recommendations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("vertical", sa.String(length=64), nullable=False),
        sa.Column("funnel_stage", sa.String(length=32), nullable=False),
        sa.Column("audience_type", sa.String(length=32), nullable=False),
        sa.Column("gap_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'open'")),
        sa.Column("brief_payload", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('open', 'in_progress', 'completed', 'declined')",
            name="ck_shadow_rct_status",
        ),
    )

    # --- Config -------------------------------------------------------------
    op.create_table(
        "decision_thresholds",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("client_id", sa.String(length=128), nullable=False),
        sa.Column("vertical", sa.String(length=64), nullable=False),
        sa.Column("funnel_stage", sa.String(length=32), nullable=False),
        sa.Column("threshold_mode", sa.String(length=32), nullable=False),
        sa.Column("threshold_value", sa.Float(), nullable=False),
        sa.Column("cost_per_fp", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("cost_per_fn", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "client_id", "vertical", "funnel_stage", name="uq_thresholds_client_seg"
        ),
        sa.CheckConstraint(
            "threshold_mode IN ('absolute', 'segment_relative', 'risk_adjusted')",
            name="ck_thresholds_mode",
        ),
    )


def downgrade() -> None:
    for table in (
        "decision_thresholds",
        "shadow_rct_recommendations",
        "prediction_runs",
        "model_metrics",
        "model_versions",
        "monitoring_snapshots",
        "cleaning_actions",
        "validation_runs",
        "feature_store",
        "rct_labels",
        "attribution_metrics",
        "campaign_metrics_daily",
        "experiment_cells",
        "experiments",
        "campaigns",
    ):
        op.drop_table(table)
