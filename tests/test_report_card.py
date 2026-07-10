"""Tests for the report-card pivots, MAE-delta, and card assembly."""

import numpy as np
import pandas as pd
import pytest

from sarizard.analysis.report_card import (
    AVERAGE_LABEL,
    BASELINE_LABEL,
    META_LABEL,
    append_average_row,
    assemble_r2_card,
    build_matrix,
    build_reference_series,
    collapse_seed_variants,
    filter_lr_mode,
    mae_delta_matrix,
    meta_model_series,
    source_groups,
)


@pytest.fixture
def tidy_metrics() -> pd.DataFrame:
    """A tiny tidy metrics frame spanning two datasets and two flavors."""
    return pd.DataFrame(
        [
            {"flavor": "osmordred", "dataset": "herg", "endpoint": "herg", "r2": 0.5, "mae": 0.4},
            {"flavor": "ecfp", "dataset": "herg", "endpoint": "herg", "r2": 0.3, "mae": 0.6},
            {"flavor": "osmordred", "dataset": "cyp", "endpoint": "cyp3a4", "r2": 0.7, "mae": 0.2},
            {"flavor": "ecfp", "dataset": "cyp", "endpoint": "cyp3a4", "r2": 0.6, "mae": 0.3},
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


def test_cell_value_is_the_metric(tidy_metrics):
    pivot = build_matrix(tidy_metrics, "r2")

    assert pivot.loc["herg · herg", "osmordred"] == 0.5


def test_seed_variants_average_into_one_flavor_column():
    # two seeds of ecfp on one endpoint must average to a single matrix cell
    frame = pd.DataFrame(
        [
            {"flavor": "ecfp__s1", "dataset": "herg", "endpoint": "herg", "r2": 0.4},
            {"flavor": "ecfp__s2", "dataset": "herg", "endpoint": "herg", "r2": 0.6},
        ]
    )

    pivot = build_matrix(collapse_seed_variants(frame), "r2", columns=["ecfp"])

    assert pivot.loc["herg · herg", "ecfp"] == pytest.approx(0.5)


def test_filter_lr_mode_keeps_one_mode_and_strips_prefix():
    frame = pd.DataFrame(
        {"flavor": ["lr_reduced__ecfp", "lr_reduced__osmordred", "lr_unlocked__ecfp"]}
    )

    kept = filter_lr_mode(frame, "reduced")

    assert list(kept["flavor"]) == ["ecfp", "osmordred"]


def test_mae_delta_is_percentage_change_from_baseline():
    mae = pd.DataFrame(
        {"osmordred": [0.4, 0.2], "ecfp": [0.6, 0.3]},
        index=["cyp · cyp3a4", "herg · herg"],
    )
    baseline = pd.Series({"cyp · cyp3a4": 0.5, "herg · herg": 0.4})

    delta = mae_delta_matrix(mae, baseline)

    # osmordred at cyp: (0.4 - 0.5) / 0.5 = -20%; ecfp at herg: (0.3 - 0.4) / 0.4 = -25%
    assert delta.loc["cyp · cyp3a4", "osmordred"] == pytest.approx(-20.0)
    assert delta.loc["herg · herg", "ecfp"] == pytest.approx(-25.0)


def test_mae_delta_is_nan_when_baseline_missing():
    mae = pd.DataFrame({"osmordred": [0.4]}, index=["cyp · cyp3a4"])
    baseline = pd.Series(dtype=float)

    delta = mae_delta_matrix(mae, baseline)

    assert np.isnan(delta.loc["cyp · cyp3a4", "osmordred"])


def test_assemble_r2_card_orders_baseline_first_and_meta_last(tidy_metrics):
    flavors = build_matrix(tidy_metrics, "r2")
    baseline = pd.Series({"cyp · cyp3a4": 0.4, "herg · herg": 0.2})
    meta = pd.Series({"cyp · cyp3a4": 0.8, "herg · herg": 0.9})

    matrix, spacer_cols, ref_cols = assemble_r2_card(flavors, baseline, meta)

    # baseline is the first column, meta the last, each behind a spacer bounding the flavor block
    assert matrix.columns[0] == BASELINE_LABEL
    assert matrix.columns[-1] == META_LABEL
    assert list(matrix.columns[2:4]) == ["osmordred", "ecfp"]
    assert ref_cols == [0, len(matrix.columns) - 1]
    assert len(spacer_cols) == 2
    assert matrix.loc["herg · herg", BASELINE_LABEL] == 0.2


def test_assemble_r2_card_without_references_is_flavors_only(tidy_metrics):
    flavors = build_matrix(tidy_metrics, "r2")

    matrix, spacer_cols, ref_cols = assemble_r2_card(
        flavors, pd.Series(dtype=float), pd.Series(dtype=float)
    )

    assert list(matrix.columns) == ["osmordred", "ecfp"]
    assert spacer_cols == []
    assert ref_cols == []


def test_append_average_row_means_each_column_over_endpoints():
    matrix = pd.DataFrame(
        {"osmordred": [0.7, 0.5], "ecfp": [0.6, 0.3]},
        index=["cyp · cyp3a4", "herg · herg"],
    )

    out, average_row = append_average_row(matrix)

    assert out.index[average_row] == AVERAGE_LABEL
    assert out.iloc[average_row]["osmordred"] == pytest.approx(0.6)
    assert out.iloc[average_row]["ecfp"] == pytest.approx(0.45)
    # a blank spacer row sits just above the average row
    assert out.iloc[average_row - 1].isna().all()


def test_source_groups_are_contiguous_runs_by_dataset():
    index = pd.Index(["cyp · a", "cyp · b", "herg · h", "pxr · p"])

    groups = source_groups(index)

    assert groups == [(0, 2, "cyp"), (2, 3, "herg"), (3, 4, "pxr")]


def test_meta_model_series_reads_r2_column(tmp_path):
    csv = tmp_path / "meta_model_lgbm.csv"
    pd.DataFrame(
        [{"dataset": "herg", "endpoint": "herg", "meta_r2": 0.8, "meta_rmse": 0.2}]
    ).to_csv(csv, index=False)

    series = meta_model_series(csv, "r2")

    assert series.to_dict() == {"herg · herg": 0.8}


def test_meta_model_series_empty_when_csv_missing(tmp_path):
    series = meta_model_series(tmp_path / "missing.csv", "r2")

    assert series.empty


def test_build_reference_series_extracts_one_flavor(tidy_metrics):
    baseline_row = pd.DataFrame(
        [{"flavor": "chemeleon_stock", "dataset": "herg", "endpoint": "herg", "r2": 0.4}]
    )
    frame = pd.concat([tidy_metrics, baseline_row], ignore_index=True)

    series = build_reference_series(frame, "chemeleon_stock", "r2")

    assert series.to_dict() == {"herg · herg": 0.4}
