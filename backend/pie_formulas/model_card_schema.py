"""
V.4 model card schema (Phase 0 of the V.4 research-alignment contract).

Every model registered via `ml.model_registry.register_model` must satisfy
this schema. The schema enforces that paper-aligned models declare:
  * OOF evaluation (not in-sample R²)
  * True cost weights (not proxy)
  * n_splits >= 10 in paper mode
  * n_bootstrap >= 1000 on OOF
  * label-noise R² ceiling (not residual-derived)

The schema is intentionally permissive on lower thresholds — small donor
pools may run with reduced CV / bootstrap, but the model card must declare
the deviation so it surfaces in the trust UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

EvalMode = Literal["paper", "research", "debug"]


@dataclass
class ModelCardCriteria:
    """Acceptance criteria for a paper-aligned model."""

    eval_mode: EvalMode = "paper"
    n_splits: int = 10
    n_bootstrap: int = 1000
    weights_source: Literal["true_cost", "proxy", "uniform"] = "true_cost"
    headline_r2_basis: Literal["oof", "in_sample"] = "oof"
    r2_ceiling_method: Literal["label_noise", "residual"] = "label_noise"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelCard:
    """Model card attached to every registered model.

    `criteria` is the declared methodology; `paper_aligned` is True only if
    every criterion matches the paper-mode defaults. The model registry
    records both — the UI shows a clear warning when False.
    """

    model_id: str
    name: str
    version_tag: str
    paper_alignment_version: str  # e.g. "v4.0.0" — matches paper_to_code_matrix.json
    criteria: ModelCardCriteria = field(default_factory=ModelCardCriteria)
    deviations: list[str] = field(default_factory=list)
    paper_aligned: bool = True

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "criteria"},
            "criteria": self.criteria.to_dict(),
        }


def evaluate_alignment(criteria: ModelCardCriteria) -> tuple[bool, list[str]]:
    """Return (paper_aligned, list_of_deviations).

    A model is paper-aligned only when every criterion matches the V.4
    defaults. Deviations are reported verbatim so the UI can render them
    inline on the Model Trust page.
    """
    deviations: list[str] = []
    if criteria.eval_mode != "paper":
        deviations.append(f"eval_mode={criteria.eval_mode} (paper requires 'paper')")
    if criteria.n_splits < 10:
        deviations.append(f"n_splits={criteria.n_splits} (paper requires >= 10)")
    if criteria.n_bootstrap < 1000:
        deviations.append(
            f"n_bootstrap={criteria.n_bootstrap} (paper requires >= 1000)"
        )
    if criteria.weights_source != "true_cost":
        deviations.append(
            f"weights_source={criteria.weights_source} (paper requires 'true_cost')"
        )
    if criteria.headline_r2_basis != "oof":
        deviations.append(
            f"headline_r2_basis={criteria.headline_r2_basis} (paper requires 'oof')"
        )
    if criteria.r2_ceiling_method != "label_noise":
        deviations.append(
            f"r2_ceiling_method={criteria.r2_ceiling_method} "
            "(paper requires 'label_noise')"
        )
    return (len(deviations) == 0, deviations)
