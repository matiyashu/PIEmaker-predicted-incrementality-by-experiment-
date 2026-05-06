"""
ATT decomposition (PDF Section 3.5) — RESEARCH MODE ONLY.

The decomposition ψ_r = τ̄ᴱ_r + δ_r × C̄ᴱ_r (Eq. 16, Gordon et al. 2026, p. 11)
is theoretical: the components are NOT directly identifiable from observable
features. This module is tagged research-mode and must NEVER surface as a
production output. Use only for diagnostic exploration in the Model Lab.
"""

from __future__ import annotations


def att_decomposition(
    treatment_effect_on_treated: float,
    selection_term: float,
    counterfactual_treatment_response: float,
) -> dict:
    """ψ_r = τ̄ᴱ_r + δ_r × C̄ᴱ_r — research-mode diagnostic.

    Reference: Eq. 16, Gordon et al. 2026, p. 11.

    Args:
        treatment_effect_on_treated: τ̄ᴱ_r — average treatment effect on the
            treated subpopulation (would-be-exposed).
        selection_term: δ_r — selection bias term capturing the difference in
            potential outcomes between treated and untreated subpopulations.
        counterfactual_treatment_response: C̄ᴱ_r — counterfactual treatment
            response for the untreated subpopulation.

    Returns:
        dict with `att`, the three components, and a `research_mode_only` flag
        that downstream services MUST check before surfacing.
    """
    att_value = (
        treatment_effect_on_treated
        + selection_term * counterfactual_treatment_response
    )
    return {
        "att": att_value,
        "treatment_effect_on_treated": treatment_effect_on_treated,
        "selection_term": selection_term,
        "counterfactual_treatment_response": counterfactual_treatment_response,
        "research_mode_only": True,
    }
