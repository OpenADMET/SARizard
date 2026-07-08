"""Tests for the finetune LR-experiment comparison (frozen vs reduced vs unlocked)."""

import pandas as pd
import pytest

from sarizard.analysis.lr_report import (
    build_mode_matrix,
    mode_flavor_frame,
    parse_lr_variant,
    summarize,
)


def test_parse_lr_variant_frozen_has_no_prefix():
    assert parse_lr_variant("ecfp__s1") == ("frozen", "ecfp")


def test_parse_lr_variant_reads_mode_from_prefix():
    assert parse_lr_variant("lr_reduced__ecfp__s2") == ("reduced", "ecfp")
    assert parse_lr_variant("lr_unlocked__surrogate_adme__s1") == ("unlocked", "surrogate_adme")


@pytest.fixture
def lr_metrics() -> pd.DataFrame:
    """Frozen (two seeds) and reduced (one seed) results for ecfp on one endpoint."""
    return pd.DataFrame(
        [
            {"flavor": "ecfp__s1", "endpoint": "herg", "r2": 0.4},
            {"flavor": "ecfp__s2", "endpoint": "herg", "r2": 0.6},
            {"flavor": "lr_reduced__ecfp__s1", "endpoint": "herg", "r2": 0.7},
            {"flavor": "notaflavor__s1", "endpoint": "herg", "r2": 0.9},
        ]
    )


def test_mode_flavor_frame_averages_seeds_and_drops_unknown_flavors(lr_metrics):
    grouped = mode_flavor_frame(lr_metrics, "r2")

    # the two frozen seeds average to 0.5; the bogus flavor is filtered out
    frozen = grouped[(grouped["flavor_base"] == "ecfp") & (grouped["mode"] == "frozen")]
    assert frozen["r2"].item() == pytest.approx(0.5)
    assert "notaflavor" not in set(grouped["flavor_base"])


def test_summarize_reports_delta_and_wins_vs_frozen(lr_metrics):
    pivot = build_mode_matrix(mode_flavor_frame(lr_metrics, "r2"), "r2")
    summary = summarize(pivot, "r2")

    # reduced (0.7) beats frozen (0.5) on the one endpoint by +0.2
    assert summary.loc["reduced", "mean_delta_vs_frozen"] == pytest.approx(0.2)
    assert summary.loc["reduced", "wins_vs_frozen"] == 1


def test_summarize_win_direction_flips_for_lower_is_better():
    # for rmse (lower better), a higher value than frozen is a loss, not a win
    pivot = pd.DataFrame(
        {"frozen": [0.5], "reduced": [0.7]}, index=["ecfp · herg"]
    )

    summary = summarize(pivot, "rmse")

    assert summary.loc["reduced", "wins_vs_frozen"] == 0
