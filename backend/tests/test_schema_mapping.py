"""Tests for schema-mapping auto-suggest."""

from __future__ import annotations

import pytest

from services.schema_mapping import apply_mapping, suggest_mappings


def test_exact_match_high_confidence():
    s = suggest_mappings(["campaign_id", "advertiser_id", "cost"])
    targets = {x.source_column: x.target_field for x in s}
    confidences = {x.source_column: x.confidence for x in s}
    assert targets == {
        "campaign_id": "campaign_id",
        "advertiser_id": "advertiser_id",
        "cost": "cost",
    }
    assert all(c == 1.0 for c in confidences.values())


def test_alias_match():
    s = suggest_mappings(["spend", "imps", "linkclicks"])
    targets = {x.source_column: x.target_field for x in s}
    assert targets["spend"] == "cost"
    assert targets["imps"] == "impressions"
    assert targets["linkclicks"] == "clicks"


def test_unknown_columns_return_none():
    s = suggest_mappings(["random_garbage_column"])
    assert s[0].target_field is None


def test_apply_mapping_validates_targets():
    out = apply_mapping(
        ["spend", "imps"], mapping={"spend": "cost", "imps": "impressions"}
    )
    assert out == {"spend": "cost", "imps": "impressions"}


def test_apply_mapping_rejects_unknown_target():
    with pytest.raises(ValueError):
        apply_mapping(["spend"], mapping={"spend": "not_a_field"})


def test_apply_mapping_rejects_duplicate_target():
    with pytest.raises(ValueError):
        apply_mapping(
            ["spend", "amount_spent"],
            mapping={"spend": "cost", "amount_spent": "cost"},
        )


def test_apply_mapping_rejects_missing_source():
    with pytest.raises(ValueError):
        apply_mapping([], mapping={"phantom": "cost"})
