"""Tests for the report-card pivots, row disambiguation, MAE-delta, and card assembly."""

import sys
import warnings

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
    mae_average_pvalues,
    mae_delta_matrix,
    mae_delta_std,
    mae_significance_pvalues,
    main,
    prepare_rows,
    source_groups,
)


@pytest.fixture
def tidy_metrics() -> pd.DataFrame:
    """Metrics where cyp1a2 is scored by two recipes (single-task and multi-task cyp)."""
    # the stored dataset is intentionally the pre-rename value; prepare_rows re-derives it
    return pd.DataFrame(
        [
            {
                "flavor": "osmordred",
                "recipe": "cyp1a2_st",
                "dataset": "cyp1a2",
                "endpoint": "OPENADMET_LOGAC50_cyp1a2",
                "r2": 0.5,
                "mae": 0.4,
            },
            {
                "flavor": "osmordred",
                "recipe": "cyp_mt",
                "dataset": "cyp",
                "endpoint": "OPENADMET_LOGAC50_cyp1a2",
                "r2": 0.7,
                "mae": 0.2,
            },
            {
                "flavor": "ecfp",
                "recipe": "cyp1a2_st",
                "dataset": "cyp1a2",
                "endpoint": "OPENADMET_LOGAC50_cyp1a2",
                "r2": 0.3,
                "mae": 0.6,
            },
            {
                "flavor": "ecfp",
                "recipe": "cyp_mt",
                "dataset": "cyp",
                "endpoint": "OPENADMET_LOGAC50_cyp1a2",
                "r2": 0.6,
                "mae": 0.3,
            },
            {
                "flavor": "osmordred",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "pchembl_value_mean",
                "r2": 0.5,
                "mae": 0.4,
            },
            {
                "flavor": "ecfp",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "pchembl_value_mean",
                "r2": 0.3,
                "mae": 0.6,
            },
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
            {
                "flavor": "ecfp__s1",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "herg",
                "r2": 0.4,
            },
            {
                "flavor": "ecfp__s2",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "herg",
                "r2": 0.6,
            },
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
            {
                "flavor": "ecfp__s1",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "herg",
                "r2": 0.4,
            },
            {
                "flavor": "ecfp__s2",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "herg",
                "r2": 0.6,
            },
        ]
    )

    std = build_matrix(
        prepare_rows(collapse_seed_variants(frame)), "r2", columns=["ecfp"], aggfunc="std"
    )

    # sample std of [0.4, 0.6] is sqrt(((-0.1)^2 + 0.1^2) / 1) = sqrt(0.02)
    assert std.loc["herg · herg", "ecfp"] == pytest.approx(np.sqrt(0.02))


def test_build_reference_series_std_aggregates_baseline_seed_spread():
    frame = pd.DataFrame(
        [
            {
                "flavor": "chemeleon_stock__s1",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "herg",
                "r2": 0.4,
            },
            {
                "flavor": "chemeleon_stock__s2",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "herg",
                "r2": 0.6,
            },
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


def _seeded_mae_frame(per_flavor_maes: dict[str, list[float]]) -> pd.DataFrame:
    """Prepared metrics frame with one herg row per flavor per seed, from given MAE samples."""
    rows = [
        {"flavor": flavor, "recipe": "herg_st", "dataset": "herg", "endpoint": "herg", "mae": mae}
        for flavor, maes in per_flavor_maes.items()
        for mae in maes
    ]
    return prepare_rows(pd.DataFrame(rows))


def test_mae_significance_flags_clear_difference_and_spares_overlap():
    frame = _seeded_mae_frame(
        {
            "chemeleon_stock": [0.50, 0.52, 0.48, 0.51, 0.49],
            "far": [0.80, 0.82, 0.78, 0.81, 0.79],  # cleanly separated from baseline
            "near": [0.50, 0.53, 0.47, 0.52, 0.48],  # overlapping baseline
        }
    )
    mae_matrix = pd.DataFrame({"far": [0.80], "near": [0.50]}, index=["herg · herg"])

    pvalues = mae_significance_pvalues(frame, frame, "chemeleon_stock", mae_matrix)

    assert pvalues.loc["herg · herg", "far"] < 0.05
    assert pvalues.loc["herg · herg", "near"] > 0.05


def test_mae_significance_is_nan_without_enough_seeds():
    frame = _seeded_mae_frame(
        {
            "chemeleon_stock": [0.50, 0.52, 0.48],
            "single": [0.80],  # one seed: no test possible
        }
    )
    mae_matrix = pd.DataFrame({"single": [0.80]}, index=["herg · herg"])

    pvalues = mae_significance_pvalues(frame, frame, "chemeleon_stock", mae_matrix)

    assert np.isnan(pvalues.loc["herg · herg", "single"])


def test_mae_significance_family_is_the_displayed_columns():
    # the correction is sized to the columns the card actually shows, not to every flavor present
    # in the metrics frame, so a standalone --columns card pays for its own column set only
    frame = _seeded_mae_frame(
        {
            "chemeleon_stock": [0.500, 0.518, 0.492, 0.511, 0.487],
            "shown": [0.545, 0.560, 0.538, 0.556, 0.533],
            "q1": [0.501, 0.502, 0.5005, 0.5015, 0.501],
            "q2": [0.502, 0.503, 0.5015, 0.5025, 0.502],
            "q3": [0.503, 0.504, 0.5025, 0.5035, 0.503],
        }
    )
    alone = pd.DataFrame({"shown": [0.546]}, index=["herg · herg"])
    with_family = pd.DataFrame(
        {name: [0.546] for name in ("shown", "q1", "q2", "q3")}, index=["herg · herg"]
    )

    p_alone = mae_significance_pvalues(frame, frame, "chemeleon_stock", alone)
    p_family = mae_significance_pvalues(frame, frame, "chemeleon_stock", with_family)

    assert p_alone.loc["herg · herg", "shown"] != p_family.loc["herg · herg", "shown"]


def test_mae_significance_tests_a_flavor_with_no_seed_spread():
    # Dunnett pools variance across the family, so a flavor whose own seeds are identical is still
    # testable against the pooled estimate; the per-cell Welch test this replaced skipped such a
    # group for having zero variance and left the cell painted white
    frame = _seeded_mae_frame(
        {
            "chemeleon_stock": [0.500, 0.518, 0.492, 0.511, 0.487],
            "flat": [0.800] * 5,  # identical across seeds: no spread of its own
            "spread": [0.545, 0.560, 0.538, 0.556, 0.533],
        }
    )
    mae_matrix = pd.DataFrame({"flat": [0.800], "spread": [0.546]}, index=["herg · herg"])

    # scipy warns of precision loss on a group this degenerate, so pin only that the cell enters
    # the family at all rather than the magnitude it comes back with
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pvalues = mae_significance_pvalues(frame, frame, "chemeleon_stock", mae_matrix)

    assert not np.isnan(pvalues.loc["herg · herg", "flat"])


_AVERAGE_ENDPOINTS = ["e1", "e2", "e3"]


def _multi_endpoint_seed_frame(per_flavor: dict[str, list[list[float]]]) -> pd.DataFrame:
    """Prepared frame of three endpoints per flavor per seed, from ``__s<seed>`` labels."""
    rows = [
        {
            "flavor": f"{flavor}__s{seed}",
            "recipe": "herg_st",
            "dataset": "herg",
            "endpoint": endpoint,
            "mae": mae,
        }
        for flavor, seeds in per_flavor.items()
        for seed, values in enumerate(seeds, start=1)
        for endpoint, mae in zip(_AVERAGE_ENDPOINTS, values, strict=True)
    ]
    return prepare_rows(collapse_seed_variants(pd.DataFrame(rows)))


def test_collapse_seed_variants_keeps_the_seed_number():
    frame = pd.DataFrame({"flavor": ["ecfp__s3", "chemeleon_stock"]})

    collapsed = collapse_seed_variants(frame)

    # the AVERAGE-row test needs the replicate identity the collapsed label drops, to line one
    # seed's rows up across endpoints
    assert list(collapsed["flavor"]) == ["ecfp", "chemeleon_stock"]
    assert collapsed["seed"].iloc[0] == 3
    assert np.isnan(collapsed["seed"].iloc[1])


def test_mae_average_pvalues_gates_a_small_mean_change_but_not_a_large_one():
    # both columns improve on the baseline across every endpoint and seed; only the one whose
    # mean change clears the family's pooled seed spread should carry color on the AVERAGE row
    frame = _multi_endpoint_seed_frame(
        {
            "chemeleon_stock": [
                [0.500, 0.600, 0.700],
                [0.510, 0.610, 0.690],
                [0.490, 0.590, 0.710],
                [0.505, 0.605, 0.705],
                [0.495, 0.595, 0.695],
            ],
            "big": [
                [0.450, 0.545, 0.625],
                [0.462, 0.544, 0.624],
                [0.437, 0.532, 0.646],
                [0.458, 0.541, 0.630],
                [0.443, 0.538, 0.622],
            ],
            "small": [
                [0.495, 0.596, 0.692],
                [0.503, 0.607, 0.680],
                [0.487, 0.581, 0.706],
                [0.502, 0.597, 0.702],
                [0.488, 0.592, 0.687],
            ],
        }
    )
    mae = build_matrix(frame, "mae", columns=["big", "small"])
    baseline = build_reference_series(frame, "chemeleon_stock", "mae")

    pvalues = mae_average_pvalues(frame, frame, "chemeleon_stock", mae, baseline)

    assert pvalues["big"] <= 0.05
    assert pvalues["small"] > 0.05


def test_mae_average_pvalues_tolerates_a_ragged_seed_grid():
    # a flavor missing one (seed, endpoint) result is the real-data case (one sweep flavor is
    # short a single finetune); it must still produce a p-value rather than raise or go NaN
    frame = _multi_endpoint_seed_frame(
        {
            "chemeleon_stock": [
                [0.500, 0.600, 0.700],
                [0.510, 0.610, 0.690],
                [0.490, 0.590, 0.710],
                [0.505, 0.605, 0.705],
                [0.495, 0.595, 0.695],
            ],
            "ragged": [
                [0.450, 0.545, 0.625],
                [0.462, 0.544, 0.624],
                [0.437, 0.532, 0.646],
                [0.458, 0.541, 0.630],
                [0.443, 0.538, 0.622],
            ],
        }
    )
    frame = frame.drop(frame[(frame["flavor"] == "ragged") & (frame["seed"] == 2)].index[:1])
    mae = build_matrix(frame, "mae", columns=["ragged"])
    baseline = build_reference_series(frame, "chemeleon_stock", "mae")

    pvalues = mae_average_pvalues(frame, frame, "chemeleon_stock", mae, baseline)

    assert not np.isnan(pvalues["ragged"])


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
    # the average follows the endpoint rows directly, with no blank row between
    assert list(out.index) == [*matrix.index, AVERAGE_LABEL]


def test_source_groups_are_contiguous_runs_by_dataset():
    index = pd.Index(["openadmet_cyp · a", "openadmet_cyp · b", "herg · h", "pxr · p"])

    groups = source_groups(index)

    assert groups == [(0, 2, "openadmet_cyp"), (2, 3, "herg"), (3, 4, "pxr")]


def test_build_reference_series_uses_disambiguated_rows(tidy_metrics):
    baseline_rows = pd.DataFrame(
        [
            {
                "flavor": "chemeleon_stock",
                "recipe": "cyp_mt",
                "dataset": "cyp",
                "endpoint": "OPENADMET_LOGAC50_cyp1a2",
                "r2": 0.42,
                "mae": 0.1,
            },
        ]
    )
    frame = prepare_rows(pd.concat([tidy_metrics, baseline_rows], ignore_index=True))

    series = build_reference_series(frame, "chemeleon_stock", "r2")

    assert series.to_dict() == {"openadmet_cyp · OPENADMET_LOGAC50_cyp1a2 (cyp_mt)": 0.42}


def test_columns_override_renders_standalone_card_excluding_the_registry(tmp_path, monkeypatch):
    # a registry flavor, two external foundations, and the baseline on one endpoint; the frame
    # carries osmordred so a regression back to the registry default would surface it as a column
    metrics = pd.DataFrame(
        [
            {
                "flavor": "osmordred",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "pchembl_value_mean",
                "r2": 0.5,
                "mae": 0.4,
            },
            {
                "flavor": "molpile_1M",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "pchembl_value_mean",
                "r2": 0.4,
                "mae": 0.5,
            },
            {
                "flavor": "molpile_5M",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "pchembl_value_mean",
                "r2": 0.45,
                "mae": 0.45,
            },
            {
                "flavor": "chemeleon_stock",
                "recipe": "herg_st",
                "dataset": "herg",
                "endpoint": "pchembl_value_mean",
                "r2": 0.35,
                "mae": 0.55,
            },
        ]
    )
    metrics_csv = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_card",
            "--metrics-csv",
            str(metrics_csv),
            "--out-dir",
            str(tmp_path),
            "--columns",
            "molpile_1M",
            "molpile_5M",
        ],
    )

    main()

    card = pd.read_csv(tmp_path / "report_card_r2.csv", index_col=0)
    assert BASELINE_LABEL in card.columns
    assert {"molpile_1M", "molpile_5M"} <= set(card.columns)
    assert "osmordred" not in card.columns
