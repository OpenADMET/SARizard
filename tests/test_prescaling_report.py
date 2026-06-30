"""Tests for the prescaling-ablation ranking direction, win counts, and seed aggregation."""

import pandas as pd
import pytest

from sarizard.analysis.prescaling_report import collapse_seed_variants, rank_ablations
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
