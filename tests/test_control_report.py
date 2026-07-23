"""Tests for the osmordred_surrogate control comparison aggregation."""

import numpy as np
import pandas as pd

from sarizard.analysis.control_report import per_seed_endpoint_means


def _frame() -> pd.DataFrame:
    # two seeds of one base, two endpoints each; a second base to prove selection isolates one
    return pd.DataFrame(
        {
            "flavor": [
                "osmordred_surrogate__s1",
                "osmordred_surrogate__s1",
                "osmordred_surrogate__s2",
                "osmordred_surrogate__s2",
                "chemeleon_stock__s1",
            ],
            "r2": [0.2, 0.4, 0.6, 0.8, 0.99],
            "endpoint": ["a", "b", "a", "b", "a"],
        }
    )


def test_per_seed_means_average_endpoints_within_each_seed():
    means = per_seed_endpoint_means(_frame(), "osmordred_surrogate")

    # seed 1 -> mean(0.2, 0.4) = 0.3; seed 2 -> mean(0.6, 0.8) = 0.7
    np.testing.assert_allclose(np.sort(means), [0.3, 0.7])


def test_per_seed_means_isolates_the_requested_base():
    means = per_seed_endpoint_means(_frame(), "chemeleon_stock")

    np.testing.assert_allclose(means, [0.99])


def test_per_seed_means_empty_for_absent_base():
    means = per_seed_endpoint_means(_frame(), "not_present")

    assert means.size == 0
