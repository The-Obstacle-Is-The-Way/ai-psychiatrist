"""Regression tests for Spec 047 (remove deprecated keyword backfill)."""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from ai_psychiatrist.config import QuantitativeSettings
from ai_psychiatrist.domain.enums import NAReason
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.services.experiment_tracking import (
    ExperimentProvenance,
    generate_output_filename,
)

pytestmark = pytest.mark.unit


def test_quantitative_settings_has_no_backfill_fields() -> None:
    """Spec 047: QuantitativeSettings removes keyword backfill knobs."""
    assert "enable_keyword_backfill" not in QuantitativeSettings.model_fields
    assert "keyword_backfill_cap" not in QuantitativeSettings.model_fields


def test_na_reason_has_no_backfill_values() -> None:
    """Spec 047: NAReason removes keyword/backfill-specific values."""
    assert "LLM_ONLY_MISSED" not in NAReason.__members__
    assert "KEYWORDS_INSUFFICIENT" not in NAReason.__members__


def test_item_assessment_has_no_keyword_evidence_count() -> None:
    """Spec 047: ItemAssessment no longer tracks keyword-injected evidence."""
    field_names = {f.name for f in fields(ItemAssessment)}
    assert "keyword_evidence_count" not in field_names


def test_output_filename_generation_has_no_backfill_parameter() -> None:
    """Spec 047: output filenames no longer encode backfill-on/off."""
    params = inspect.signature(generate_output_filename).parameters
    assert "backfill" not in params


def test_experiment_provenance_has_no_enable_keyword_backfill() -> None:
    """Spec 047: provenance should not include removed backfill config."""
    field_names = {f.name for f in fields(ExperimentProvenance)}
    assert "enable_keyword_backfill" not in field_names
