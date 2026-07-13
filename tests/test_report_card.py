"""Tests for the report-card pivots, row disambiguation, MAE-delta, and card assembly."""

import numpy as np
import pandas as pd
import pytest

from sarizard.analysis.report_card import (
    AVERAGE_LABEL,
    BASELINE_LABEL,
    append_average_row,
    assemble_r2_card,
    build_matrix,
    build_reference_series,
    collapse_seed_variants,
    filter_lr_mode,
    mae_delta_matrix,
    mae_delta_std,
    prepare_rows,
    source_groups,
)


@pytest.fixture
def tidy_metrics() -> pd.DataFrame:
    """Metrics where cyp1a2 is scored by two recipes (single-task and multi-task cyp)."""
    # the stored dataset is intentionally the pre-rename value; prepare_rows re-derives it
    return pd.DataFrame(
        [
            {"flavor": "osmordred", "recipe": "cyp1a2_st", "dataset": "cyp1a2",
             "endpoint": "OPENADMET_LOGAC50_cyp1a2", "r2": 0.5, "mae": 0.4},
            {"flavor": "osmordred", "recipe": "cyp_mt", "dataset": "cyp",
             "endpoint": "OPENADMET_LOGAC50_cyp1a2", "r2": 0.7, "mae": 0.2},
            {"flavor": "ecfp", "recipe": "cyp1a2_st", "dataset": "cyp1a2",
             "endpoint": "OPENADMET_LOGAC50_cyp1a2", "r2": 0.3, "mae": 0.6},
            {"flavor": "ecfp", "recipe": "cyp_mt", "dataset": "cyp",
             "endpoint": "OPENADMET_LOGAC50_cyp1a2", "r2": 0.6, "mae": 0.3},
            {"flavor": "osmordred", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "pchembl_value_mean", "r2": 0.5, "mae": 0.4},
            {"flavor": "ecfp", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "pchembl_value_mean", "r2": 0.3, "mae": 0.6},
        ]
    )


def test_prepare_rows_rederives_dataset_from_recipe(tidy_metrics):
    prepared = prepare_rows(tidy_metrics)

    # both CYP recipes are regrouped under openadmet_cyp regardless of the stored dataset
    cyp = prepared[prepared["endpoint"] == "OPENADMET_LOGAC50_cyp1a2"]
    assert set(cyp["dataset"]) == {"openadmet_cyp"}


def test_prepare_rows_disambiguates_shared_endpoint_by_recipe(tidy_metrics):
    prepared = prepare_rows(tidy_metrics)

    # cyp1a2 comes from two recipes, so each row carries its recipe; herg (one recipe) does not
    cyp_rows = set(prepared[prepared["endpoint"] == "OPENADMET_LOGAC50_cyp1a2"]["row"])
    assert cyp_rows == {
        "openadmet_cyp · OPENADMET_LOGAC50_cyp1a2 (cyp1a2_st)",
        "openadmet_cyp · OPENADMET_LOGAC50_cyp1a2 (cyp_mt)",
    }
    assert set(prepared[prepared["recipe"] == "herg_st"]["row"]) == {"herg · pchembl_value_mean"}


def test_shared_endpoint_stays_two_rows_instead_of_averaging(tidy_metrics):
    pivot = build_matrix(prepare_rows(tidy_metrics), "r2")

    # the single-task and multi-task cyp1a2 keep their own osmordred cells (0.5 and 0.7), not a
    # silent mean of the two
    st = "openadmet_cyp · OPENADMET_LOGAC50_cyp1a2 (cyp1a2_st)"
    mt = "openadmet_cyp · OPENADMET_LOGAC50_cyp1a2 (cyp_mt)"
    assert pivot.loc[st, "osmordred"] == 0.5
    assert pivot.loc[mt, "osmordred"] == 0.7


def test_rows_ordered_by_dataset_rank(tidy_metrics):
    pivot = build_matrix(prepare_rows(tidy_metrics), "r2")

    # openadmet_cyp precedes herg in the dataset order; the two cyp rows come first
    datasets = [row.split(" · ", 1)[0] for row in pivot.index]
    assert datasets == ["openadmet_cyp", "openadmet_cyp", "herg"]


def test_columns_default_to_registry_order(tidy_metrics):
    pivot = build_matrix(prepare_rows(tidy_metrics), "r2")

    # osmordred precedes ecfp in the flavor registry
    assert list(pivot.columns) == ["osmordred", "ecfp"]


def test_seed_variants_average_into_one_flavor_column():
    frame = pd.DataFrame(
        [
            {"flavor": "ecfp__s1", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "herg", "r2": 0.4},
            {"flavor": "ecfp__s2", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "herg", "r2": 0.6},
        ]
    )

    pivot = build_matrix(prepare_rows(collapse_seed_variants(frame)), "r2", columns=["ecfp"])

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
        index=["openadmet_cyp · cyp3a4", "herg · herg"],
    )
    baseline = pd.Series({"openadmet_cyp · cyp3a4": 0.5, "herg · herg": 0.4})

    delta = mae_delta_matrix(mae, baseline)

    # osmordred at cyp: (0.4 - 0.5) / 0.5 = -20%; ecfp at herg: (0.3 - 0.4) / 0.4 = -25%
    assert delta.loc["openadmet_cyp · cyp3a4", "osmordred"] == pytest.approx(-20.0)
    assert delta.loc["herg · herg", "ecfp"] == pytest.approx(-25.0)


def test_mae_delta_is_nan_when_baseline_missing():
    mae = pd.DataFrame({"osmordred": [0.4]}, index=["herg · herg"])

    delta = mae_delta_matrix(mae, pd.Series(dtype=float))

    assert np.isnan(delta.loc["herg · herg", "osmordred"])


def test_build_matrix_std_aggregates_the_seed_spread():
    frame = pd.DataFrame(
        [
            {"flavor": "ecfp__s1", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "herg", "r2": 0.4},
            {"flavor": "ecfp__s2", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "herg", "r2": 0.6},
        ]
    )

    std = build_matrix(prepare_rows(collapse_seed_variants(frame)), "r2", columns=["ecfp"],
                       aggfunc="std")

    # sample std of [0.4, 0.6] is sqrt(((-0.1)^2 + 0.1^2) / 1) = sqrt(0.02)
    assert std.loc["herg · herg", "ecfp"] == pytest.approx(np.sqrt(0.02))


def test_build_reference_series_std_aggregates_baseline_seed_spread():
    frame = pd.DataFrame(
        [
            {"flavor": "chemeleon_stock__s1", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "herg", "r2": 0.4},
            {"flavor": "chemeleon_stock__s2", "recipe": "herg_st", "dataset": "herg",
             "endpoint": "herg", "r2": 0.6},
        ]
    )

    std = build_reference_series(
        prepare_rows(collapse_seed_variants(frame)), "chemeleon_stock", "r2", agg="std"
    )

    assert std.loc["herg · herg"] == pytest.approx(np.sqrt(0.02))


def test_mae_delta_std_propagates_both_seed_spreads():
    mae = pd.DataFrame({"ecfp": [0.6]}, index=["herg · herg"])
    mae_std = pd.DataFrame({"ecfp": [0.1]}, index=["herg · herg"])
    baseline = pd.Series({"herg · herg": 0.5})
    baseline_std = pd.Series({"herg · herg": 0.05})

    sigma = mae_delta_std(mae, mae_std, baseline, baseline_std)

    # 100 * sqrt((0.1/0.5)^2 + (0.6*0.05/0.5^2)^2) = 100 * sqrt(0.2^2 + 0.12^2)
    assert sigma.loc["herg · herg", "ecfp"] == pytest.approx(100.0 * np.sqrt(0.2**2 + 0.12**2))


def test_mae_delta_std_is_nan_when_a_side_is_single_seed():
    mae = pd.DataFrame({"ecfp": [0.6]}, index=["herg · herg"])
    mae_std = pd.DataFrame({"ecfp": [np.nan]}, index=["herg · herg"])
    baseline = pd.Series({"herg · herg": 0.5})
    baseline_std = pd.Series({"herg · herg": 0.05})

    sigma = mae_delta_std(mae, mae_std, baseline, baseline_std)

    assert np.isnan(sigma.loc["herg · herg", "ecfp"])


def test_assemble_r2_card_orders_baseline_first_then_flavors(tidy_metrics):
    flavors = build_matrix(prepare_rows(tidy_metrics), "r2")
    baseline = pd.Series(0.4, index=flavors.index)

    matrix, spacer_cols, ref_cols = assemble_r2_card(flavors, baseline)

    # baseline is the first column, behind one spacer that bounds the flavor block; no meta column
    assert matrix.columns[0] == BASELINE_LABEL
    assert list(matrix.columns[2:]) == ["osmordred", "ecfp"]
    assert ref_cols == [0]
    assert spacer_cols == [1]


def test_assemble_r2_card_without_baseline_is_flavors_only(tidy_metrics):
    flavors = build_matrix(prepare_rows(tidy_metrics), "r2")

    matrix, spacer_cols, ref_cols = assemble_r2_card(flavors, pd.Series(dtype=float))

    assert list(matrix.columns) == ["osmordred", "ecfp"]
    assert spacer_cols == []
    assert ref_cols == []


def test_append_average_row_means_each_column_over_endpoints():
    matrix = pd.DataFrame(
        {"osmordred": [0.7, 0.5], "ecfp": [0.6, 0.3]},
        index=["openadmet_cyp · a", "herg · h"],
    )

    out, average_row = append_average_row(matrix)

    assert out.index[average_row] == AVERAGE_LABEL
    assert out.iloc[average_row]["osmordred"] == pytest.approx(0.6)
    assert out.iloc[average_row]["ecfp"] == pytest.approx(0.45)
    # a blank spacer row sits just above the average row
    assert out.iloc[average_row - 1].isna().all()


def test_source_groups_are_contiguous_runs_by_dataset():
    index = pd.Index(["openadmet_cyp · a", "openadmet_cyp · b", "herg · h", "pxr · p"])

    groups = source_groups(index)

    assert groups == [(0, 2, "openadmet_cyp"), (2, 3, "herg"), (3, 4, "pxr")]


def test_build_reference_series_uses_disambiguated_rows(tidy_metrics):
    baseline_rows = pd.DataFrame(
        [
            {"flavor": "chemeleon_stock", "recipe": "cyp_mt", "dataset": "cyp",
             "endpoint": "OPENADMET_LOGAC50_cyp1a2", "r2": 0.42, "mae": 0.1},
        ]
    )
    frame = prepare_rows(pd.concat([tidy_metrics, baseline_rows], ignore_index=True))

    series = build_reference_series(frame, "chemeleon_stock", "r2")

    assert series.to_dict() == {"openadmet_cyp · OPENADMET_LOGAC50_cyp1a2 (cyp_mt)": 0.42}
