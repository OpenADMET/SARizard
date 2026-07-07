"""Tests for the report-card pivot and row-relative coloring."""

import numpy as np
import pandas as pd
import pytest

from sarizard.analysis.report_card import (
    SPACER_COLUMN,
    _row_relative,
    augment_with_references,
    build_matrix,
    build_reference_series,
    collapse_seed_variants,
    meta_model_series,
)


@pytest.fixture
def tidy_metrics() -> pd.DataFrame:
    """A tiny tidy metrics frame spanning two datasets and two flavors."""
    return pd.DataFrame(
        [
            {"flavor": "osmordred", "dataset": "herg", "endpoint": "herg", "r2": 0.5},
            {"flavor": "ecfp", "dataset": "herg", "endpoint": "herg", "r2": 0.3},
            {"flavor": "osmordred", "dataset": "cyp", "endpoint": "cyp3a4", "r2": 0.7},
            {"flavor": "ecfp", "dataset": "cyp", "endpoint": "cyp3a4", "r2": 0.6},
        ]
    )


def test_rows_ordered_by_dataset_rank(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2")

    # cyp precedes herg in the DATASETS ordering, regardless of input row order
    assert list(pivot.index) == ["cyp · cyp3a4", "herg · herg"]


def test_columns_default_to_registry_order(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2")

    # osmordred precedes ecfp in the flavor registry
    assert list(pivot.columns) == ["osmordred", "ecfp"]


def test_explicit_column_order_is_honored(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2", columns=["ecfp", "osmordred"])

    assert list(pivot.columns) == ["ecfp", "osmordred"]


def test_absent_columns_are_dropped(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2", columns=["ecfp", "osmordred", "whim"])

    assert list(pivot.columns) == ["ecfp", "osmordred"]


def test_cell_value_is_the_metric(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2")

    assert pivot.loc["herg · herg", "osmordred"] == 0.5


def test_row_relative_maps_best_to_one_when_higher_is_better():
    normed = _row_relative(np.array([[0.1, 0.5, 0.9]]), higher_better=True)

    assert np.allclose(normed, [[0.0, 0.5, 1.0]])


def test_row_relative_inverts_when_lower_is_better():
    normed = _row_relative(np.array([[0.1, 0.5, 0.9]]), higher_better=False)

    assert np.allclose(normed, [[1.0, 0.5, 0.0]])


def test_row_relative_constant_row_is_midpoint():
    normed = _row_relative(np.array([[0.3, 0.3]]), higher_better=True)

    assert np.allclose(normed, [[0.5, 0.5]])


def test_row_relative_all_nan_row_stays_nan():
    normed = _row_relative(np.array([[np.nan, np.nan]]), higher_better=True)

    assert np.isnan(normed).all()


def test_collapse_seed_variants_maps_to_base_flavor():
    frame = pd.DataFrame({"flavor": ["ecfp__s1", "ecfp__s2", "osmordred__s1"]})

    collapsed = collapse_seed_variants(frame)

    assert list(collapsed["flavor"]) == ["ecfp", "ecfp", "osmordred"]


def test_seed_variants_average_into_one_flavor_column():
    # two seeds of ecfp on one endpoint must average to a single matrix cell
    frame = pd.DataFrame(
        [
            {"flavor": "ecfp__s1", "dataset": "herg", "endpoint": "herg", "r2": 0.4},
            {"flavor": "ecfp__s2", "dataset": "herg", "endpoint": "herg", "r2": 0.6},
        ]
    )

    pivot = build_matrix(collapse_seed_variants(frame), "r2", columns=["ecfp"])

    assert pivot.shape == (1, 1)
    assert pivot.loc["herg · herg", "ecfp"] == pytest.approx(0.5)


def test_build_reference_series_extracts_one_flavor(tidy_metrics):
    baseline_row = pd.DataFrame(
        [{"flavor": "chemeleon_stock", "dataset": "herg", "endpoint": "herg", "r2": 0.4}]
    )
    frame = pd.concat([tidy_metrics, baseline_row], ignore_index=True)

    series = build_reference_series(frame, "chemeleon_stock", "r2")

    assert series.to_dict() == {"herg · herg": 0.4}


def test_build_reference_series_empty_when_flavor_absent(tidy_metrics):
    series = build_reference_series(tidy_metrics, "chemeleon_stock", "r2")

    assert series.empty


def test_meta_model_series_reads_r2_column(tmp_path):
    csv = tmp_path / "meta_model_lgbm.csv"
    pd.DataFrame(
        [{"dataset": "herg", "endpoint": "herg", "meta_r2": 0.8, "meta_rmse": 0.2}]
    ).to_csv(csv, index=False)

    series = meta_model_series(csv, "r2")

    assert series.to_dict() == {"herg · herg": 0.8}


def test_meta_model_series_empty_for_unsupported_metric(tmp_path):
    csv = tmp_path / "meta_model_lgbm.csv"
    pd.DataFrame(
        [{"dataset": "herg", "endpoint": "herg", "meta_r2": 0.8, "meta_rmse": 0.2}]
    ).to_csv(csv, index=False)

    series = meta_model_series(csv, "spearman")

    assert series.empty


def test_meta_model_series_empty_when_csv_missing(tmp_path):
    series = meta_model_series(tmp_path / "missing.csv", "r2")

    assert series.empty


def test_augment_with_references_adds_spacer_and_labeled_columns(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2")
    baseline = pd.Series({"herg · herg": 0.2, "cyp · cyp3a4": 0.4})
    meta = pd.Series({"herg · herg": 0.9, "cyp · cyp3a4": 0.8})

    augmented, divider_at = augment_with_references(pivot, baseline=baseline, meta_model=meta)

    assert divider_at == pivot.shape[1]
    assert list(augmented.columns) == [
        "osmordred", "ecfp", SPACER_COLUMN,
        "chemeleon baseline (stock, external)", "meta-model (LGBM, all flavors)",
    ]
    assert augmented.loc["herg · herg", "chemeleon baseline (stock, external)"] == 0.2
    assert augmented.loc["cyp · cyp3a4", "meta-model (LGBM, all flavors)"] == 0.8
    assert np.isnan(augmented[SPACER_COLUMN]).all()


def test_augment_with_references_skips_empty_series(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2")

    augmented, divider_at = augment_with_references(
        pivot, baseline=pd.Series(dtype=float), meta_model=None
    )

    assert divider_at == pivot.shape[1]
    assert list(augmented.columns) == list(pivot.columns)
