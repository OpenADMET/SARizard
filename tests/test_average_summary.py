"""Tests for the AVERAGE-row summary boxplots and the card colors they borrow."""

import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

from sarizard.analysis.average_summary import (
    _per_seed_average_metric,
    main,
    render_mae_delta_summary,
    render_r2_summary,
)
from sarizard.analysis.report_card import (
    BASELINE_LABEL,
    average_cell_colors,
    build_mae_delta_card,
    collapse_seed_variants,
    prepare_rows,
)

# two endpoints on different source datasets, so the summary runs over more than one row
_ENDPOINTS = [
    ("herg_st", "herg", "pchembl_value_mean"),
    ("cyp_mt", "openadmet_cyp", "OPENADMET_LOGAC50_cyp1a2"),
]

# per-flavor (r2, mae) means; the baseline sits between the clear winner and the clear loser so
# the delta card has one significantly better column, one significantly worse, and one that only
# drifts. The seed jitter below is small against these gaps, so the verdicts are not borderline
_LEVELS = {
    "osmordred": (0.50, 0.30),
    "ecfp": (0.20, 0.70),
    "rdkit2d": (0.35, 0.50),
    "chemeleon_stock": (0.35, 0.50),
}


def _metrics(
    levels: dict[str, tuple[float, float]] | None = None,
    *,
    label: str = "{name}__s{seed}",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a tidy metrics frame with one seed-jittered row per (label, endpoint, seed).

    ``label`` templates the result label so a caller can wrap the flavor names in whatever
    namespace its setup uses (the ablations prefix and suffix theirs).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, (r2, mae) in (levels or _LEVELS).items():
        for recipe, dataset, endpoint in _ENDPOINTS:
            for replicate in (1, 2, 3, 4, 5):
                rows.append(
                    {
                        "flavor": label.format(name=name, seed=replicate),
                        "recipe": recipe,
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "r2": r2 + rng.normal(scale=0.005),
                        "mae": mae + rng.normal(scale=0.005),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def prepared() -> pd.DataFrame:
    """The tidy frame as the renderers take it: seed-collapsed with row identities built."""
    return prepare_rows(collapse_seed_variants(_metrics()))


def test_per_seed_average_metric_means_endpoints_within_each_seed(prepared):
    rows = prepared["row"].drop_duplicates()

    values = _per_seed_average_metric(prepared, "osmordred", rows, "r2")

    # one value per seed, each the mean of that seed's two endpoints
    assert values.size == 5
    expected = prepared[prepared["flavor"] == "osmordred"].groupby("seed")["r2"].mean().to_numpy()
    np.testing.assert_allclose(np.sort(values), np.sort(expected))


def test_per_seed_average_metric_is_empty_for_an_absent_flavor(prepared):
    rows = prepared["row"].drop_duplicates()

    assert _per_seed_average_metric(prepared, "nonexistent", rows, "r2").size == 0


def test_average_cell_colors_track_the_significance_verdict(prepared):
    card = build_mae_delta_card(
        prepared, prepared, "chemeleon_stock", columns=["osmordred", "ecfp", "rdkit2d"]
    )

    colors = average_cell_colors(card)

    # osmordred's MAE is well below the baseline's and ecfp's well above, both far outside the
    # seed jitter; rdkit2d sits on the baseline, so its box stays white however it rounds
    assert card.average_pvalues["osmordred"] <= 0.05
    assert card.average_pvalues["ecfp"] <= 0.05
    assert card.average_pvalues["rdkit2d"] > 0.05
    # a non-significant column takes the ramp's center, which the 256-entry LUT renders a shade
    # off pure white; what matters is that it is the center rather than its own change
    np.testing.assert_allclose(to_rgb(colors["rdkit2d"]), (1.0, 1.0, 1.0), atol=0.02)
    # green for the improvement, red for the regression, wherever the ramp lands them
    improved_r, improved_g, _ = to_rgb(colors["osmordred"])
    worsened_r, worsened_g, _ = to_rgb(colors["ecfp"])
    assert improved_g > improved_r
    assert worsened_r > worsened_g


def test_mae_delta_summary_csv_matches_the_card_average_row(prepared, tmp_path):
    columns = ["osmordred", "ecfp", "rdkit2d"]
    card = build_mae_delta_card(prepared, prepared, "chemeleon_stock", columns=columns)

    render_mae_delta_summary(
        prepared, prepared, "chemeleon_stock", tmp_path / "summary.png", columns=columns
    )

    summary = pd.read_csv(tmp_path / "summary.csv", index_col=0)
    assert (tmp_path / "summary.png").exists()
    # the plotted average is the card's AVERAGE row, not a re-derivation of it
    np.testing.assert_allclose(
        summary["average"].reindex(columns).to_numpy(),
        card.average_delta.reindex(columns).to_numpy(),
    )
    assert summary.loc["osmordred", "significant"]
    assert not summary.loc["rdkit2d", "significant"]
    # the box spans the endpoints, the error bar the seeds
    assert (summary["n_endpoints"] == len(_ENDPOINTS)).all()
    assert (summary["n_seeds"] == 5).all()


def test_r2_summary_leads_with_the_baseline_column(prepared, tmp_path):
    render_r2_summary(
        prepared, prepared, "chemeleon_stock", tmp_path / "summary.png", columns=["osmordred"]
    )

    summary = pd.read_csv(tmp_path / "summary.csv", index_col=0)
    assert list(summary.index) == [BASELINE_LABEL, "osmordred"]
    # no significance is claimed on the R² side, so the CSV carries no verdict columns
    assert "significant" not in summary.columns
    assert summary.loc["osmordred", "average"] > summary.loc[BASELINE_LABEL, "average"]


def test_r2_summary_orders_the_columns_by_descending_average(prepared, tmp_path):
    # passed in worst-first, so registry order and plotted order cannot coincide by accident
    render_r2_summary(
        prepared,
        prepared,
        "chemeleon_stock",
        tmp_path / "summary.png",
        columns=["ecfp", "rdkit2d", "osmordred"],
    )

    summary = pd.read_csv(tmp_path / "summary.csv", index_col=0)

    # the baseline stays pinned at the left ahead of the ranking, then best R² to worst
    assert list(summary.index) == [BASELINE_LABEL, "osmordred", "rdkit2d", "ecfp"]


def test_mae_delta_summary_orders_the_columns_by_ascending_average(prepared, tmp_path):
    render_mae_delta_summary(
        prepared,
        prepared,
        "chemeleon_stock",
        tmp_path / "summary.png",
        columns=["ecfp", "rdkit2d", "osmordred"],
    )

    summary = pd.read_csv(tmp_path / "summary.csv", index_col=0)

    # the largest MAE reduction leads; ecfp, whose MAE is well above the baseline's, sits last
    assert list(summary.index) == ["osmordred", "rdkit2d", "ecfp"]
    assert summary["average"].is_monotonic_increasing


def test_missing_baseline_skips_the_mae_summary(prepared, tmp_path, caplog):
    render_mae_delta_summary(
        prepared, prepared, "absent_baseline", tmp_path / "summary.png", columns=["osmordred"]
    )

    assert not (tmp_path / "summary.png").exists()
    assert "absent_baseline" in caplog.text


def test_main_reads_the_ablation_protocol_off_the_label_suffix(tmp_path, monkeypatch):
    # ablation labels tag the protocol as a __<mode> suffix rather than the lr_<mode>__ prefix the
    # flavor sweep uses. The frozen rows carry a different R² from the reduced ones, so picking
    # the wrong protocol changes the summary rather than passing silently
    frozen = _metrics({"minimal": (0.60, 0.30)}, label="ablation_{name}__s{seed}")
    reduced = _metrics({"minimal": (0.20, 0.70)}, label="ablation_{name}__s{seed}__reduced")
    baseline = _metrics({"chemeleon_stock_reduced": (0.35, 0.50)}, label="{name}__s{seed}")
    metrics_csv = tmp_path / "ablation_metrics.csv"
    pd.concat([frozen, reduced, baseline], ignore_index=True).to_csv(metrics_csv, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "average_summary",
            "--metrics-csv",
            str(metrics_csv),
            "--out-dir",
            str(tmp_path),
            "--baseline-flavor",
            "chemeleon_stock_reduced",
            "--lr-mode",
            "reduced",
            "--ablations",
            "--prefix",
            "ablation_average_summary",
        ],
    )

    main()

    assert (tmp_path / "ablation_average_summary_mae_delta_reduced.png").exists()
    summary = pd.read_csv(tmp_path / "ablation_average_summary_r2_reduced.csv", index_col=0)
    # the ablation_ prefix and the __reduced suffix are both off the column name
    assert list(summary.index) == [BASELINE_LABEL, "minimal"]
    # the reduced rows, not the frozen ones the same recipe also carries
    assert summary.loc["minimal", "average"] == pytest.approx(0.20, abs=0.01)
