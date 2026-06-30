"""Tests for the report-card pivot and row-relative coloring."""

import numpy as np
import pandas as pd
import pytest

from analysis.report_card import _row_relative, build_matrix


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
