"""Tests for the prescaling-ablation ranking direction and win counts."""

import pandas as pd
import pytest

from sarizard.analysis.prescaling_report import rank_ablations


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
