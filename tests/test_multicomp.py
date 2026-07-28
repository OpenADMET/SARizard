"""Tests for the many-to-one multiple-comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ttest_ind

from sarizard.analysis.multicomp import dunnett_pvalues

RNG = np.random.default_rng(20260727)


def _control() -> np.ndarray:
    return np.array([0.300, 0.291, 0.310, 0.298, 0.295])


def test_separates_a_clear_effect_from_an_overlapping_one():
    samples = {
        "far": np.array([0.420, 0.415, 0.430, 0.418, 0.425]),
        "near": np.array([0.299, 0.305, 0.292, 0.301, 0.297]),
    }
    pvalues = dunnett_pvalues(samples, _control())
    assert pvalues["far"] < 0.05
    assert pvalues["near"] > 0.05


def test_correction_grows_with_family_size():
    """The same comparison must get a larger p-value when more flavors share the control."""
    treatment = np.array([0.330, 0.336, 0.325, 0.332, 0.329])
    control = _control()
    small = dunnett_pvalues({"a": treatment}, control)["a"]
    large = dunnett_pvalues(
        {"a": treatment} | {f"noise{i}": control + RNG.normal(0, 0.008, 5) for i in range(14)},
        control,
    )["a"]
    assert large > small


def test_family_wise_p_exceeds_the_uncorrected_pairwise_p_under_equal_variances():
    """With comparable spreads, correcting costs power relative to an isolated pairwise test.

    This is not a universal invariant. Dunnett pools variance across the whole family, so a group
    whose own spread is much wider than the pool can come out with a *smaller* p-value than Welch
    gives it, the pooled degrees of freedom outweighing the multiplicity penalty. That is a
    symptom of heteroscedasticity, not a bug, which is why this test fixes the spreads.
    """
    control = _control()
    samples = {f"f{i}": control + 0.02 + RNG.normal(0, 0.01, 5) for i in range(10)}
    corrected = dunnett_pvalues(samples, control)
    for label, sample in samples.items():
        pairwise = ttest_ind(sample, control, equal_var=False).pvalue
        assert corrected[label] > pairwise


def test_pooling_can_beat_welch_when_one_group_is_much_noisier():
    """Guard the documented exception: a noisy group borrows precision from the pooled variance."""
    control = _control()
    noisy = np.array([0.36, 0.31, 0.40, 0.30, 0.38])  # same mean shift, far wider spread
    quiet_family = {f"q{i}": control + RNG.normal(0, 0.002, 5) for i in range(4)}
    corrected = dunnett_pvalues({"noisy": noisy} | quiet_family, control)
    assert corrected["noisy"] < ttest_ind(noisy, control, equal_var=False).pvalue


@pytest.mark.parametrize("size", [0, 1])
def test_undersized_group_is_undefined_not_guessed(size):
    samples = {"tiny": np.array([0.4] * size), "ok": np.array([0.42, 0.41, 0.43, 0.42, 0.40])}
    pvalues = dunnett_pvalues(samples, _control())
    assert np.isnan(pvalues["tiny"])
    assert not np.isnan(pvalues["ok"])


def test_undersized_control_makes_the_whole_family_undefined():
    samples = {"a": np.array([0.42, 0.41, 0.43, 0.42, 0.40])}
    pvalues = dunnett_pvalues(samples, np.array([0.30]))
    assert np.isnan(pvalues["a"])


def test_zero_variance_everywhere_is_undefined_rather_than_certain():
    """A constant control and constant treatments leave no pooled variance to test against."""
    samples = {"a": np.full(5, 0.40)}
    pvalues = dunnett_pvalues(samples, np.full(5, 0.30))
    assert np.isnan(pvalues["a"])


def test_every_requested_label_comes_back():
    samples = {"a": np.array([0.42, 0.41, 0.43, 0.42, 0.40]), "b": np.array([0.30])}
    assert set(dunnett_pvalues(samples, _control())) == {"a", "b"}
