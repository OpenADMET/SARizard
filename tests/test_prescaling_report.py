"""Tests for the prescaling-ablation ranking direction, win counts, and seed aggregation."""

import pandas as pd
import pytest

from sarizard.analysis import prescaling_report
from sarizard.analysis.prescaling_report import (
    collapse_seed_variants,
    mode_comparison,
    rank_ablations,
)
from sarizard.analysis.report_card import build_matrix


@pytest.fixture
def pivot() -> pd.DataFrame:
    """Endpoints (rows) by ablation (columns); 'full' dominates 'minimal' on every row."""
    return pd.DataFrame(
        {"minimal": [0.1, 0.2], "full": [0.5, 0.6]},
        index=["endpoint_a", "endpoint_b"],
    )


def test_higher_is_better_ranks_dominant_ablation_first(pivot):
    summary = rank_ablations(pivot, "r2")

    assert summary.index[0] == "full"


def test_higher_is_better_counts_endpoint_wins(pivot):
    summary = rank_ablations(pivot, "r2")

    assert summary.loc["full", "wins"] == 2
    assert summary.loc["minimal", "wins"] == 0


def test_lower_is_better_flips_the_ranking(pivot):
    summary = rank_ablations(pivot, "rmse")

    assert summary.index[0] == "minimal"
    assert summary.loc["minimal", "wins"] == 2


def test_collapse_seed_variants_maps_to_plain_ablation_label():
    frame = pd.DataFrame(
        {"flavor": ["ablation_full__s1", "ablation_full__s2", "ablation_minimal"]}
    )

    collapsed = collapse_seed_variants(frame)

    assert list(collapsed["flavor"]) == ["ablation_full", "ablation_full", "ablation_minimal"]


def test_seed_variants_average_to_one_column_per_ablation():
    # two seeds of the same ablation on one endpoint must average to a single matrix cell
    frame = pd.DataFrame(
        {
            "flavor": ["ablation_full__s1", "ablation_full__s2"],
            "dataset": ["d", "d"],
            "endpoint": ["e", "e"],
            "r2": [0.4, 0.6],
        }
    )

    pivot = build_matrix(collapse_seed_variants(frame), "r2", columns=["ablation_full"])

    assert pivot.shape == (1, 1)
    assert pivot.loc["d · e", "ablation_full"] == pytest.approx(0.5)


def test_mode_comparison_orders_columns_frozen_to_unlocked():
    # dict insertion is out of protocol order; the comparison must reorder frozen→reduced→unlocked
    per_mode_mean = {
        "unlocked": pd.Series({"full": 0.7, "minimal": 0.3}),
        "frozen": pd.Series({"full": 0.5, "minimal": 0.1}),
        "reduced": pd.Series({"full": 0.6, "minimal": 0.2}),
    }

    comparison = mode_comparison(per_mode_mean)

    assert list(comparison.columns) == ["frozen", "reduced", "unlocked"]
    assert comparison.loc["full", "unlocked"] == pytest.approx(0.7)


@pytest.fixture
def ablation_metrics_csv(tmp_path):
    """Write a two-endpoint ablation metrics CSV spanning frozen, reduced, and unlocked."""
    rows = []
    # 'full' beats 'minimal' under every protocol, but the margin grows as the backbone unfreezes
    means = {
        ("full", "frozen"): 0.50, ("minimal", "frozen"): 0.40,
        ("full", "reduced"): 0.60, ("minimal", "reduced"): 0.45,
        ("full", "unlocked"): 0.70, ("minimal", "unlocked"): 0.48,
    }
    for (name, mode), value in means.items():
        label = "ablation_" + name + "__s42" + ("" if mode == "frozen" else f"__{mode}")
        for endpoint in ("herg", "cyp"):
            rows.append(
                {"flavor": label, "recipe": endpoint, "endpoint": endpoint, "r2": value}
            )
    csv_path = tmp_path / "ablation_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_main_writes_per_protocol_and_comparison_artifacts(
    ablation_metrics_csv, tmp_path, monkeypatch
):
    # redirect plot output into tmp_path and drive main over all three protocols
    plots_dir = tmp_path / "plots"
    monkeypatch.setattr(prescaling_report, "PLOTS_DIR", plots_dir)
    monkeypatch.setattr(
        "sys.argv",
        ["prescaling_report", "--metrics-csv", str(ablation_metrics_csv), "--metric", "r2"],
    )

    prescaling_report.main()

    # frozen keeps unsuffixed names; the LR protocols and the comparison are suffixed
    assert (plots_dir / "prescaling_ranking_r2.csv").exists()
    assert (plots_dir / "prescaling_ranking_r2_reduced.csv").exists()
    assert (plots_dir / "prescaling_ranking_r2_unlocked.csv").exists()
    comparison = pd.read_csv(plots_dir / "prescaling_mode_comparison_r2.csv", index_col=0)
    assert list(comparison.columns) == ["frozen", "reduced", "unlocked"]
    assert comparison.loc["full", "unlocked"] == pytest.approx(0.70)


def test_main_single_protocol_writes_no_comparison(
    ablation_metrics_csv, tmp_path, monkeypatch
):
    plots_dir = tmp_path / "plots"
    monkeypatch.setattr(prescaling_report, "PLOTS_DIR", plots_dir)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prescaling_report", "--metrics-csv", str(ablation_metrics_csv),
            "--metric", "r2", "--mpnn-lr-mode", "frozen",
        ],
    )

    prescaling_report.main()

    assert (plots_dir / "prescaling_ranking_r2.csv").exists()
    assert not (plots_dir / "prescaling_mode_comparison_r2.csv").exists()
