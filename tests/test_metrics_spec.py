"""Tests for recipe-to-dataset mapping (longest-prefix match)."""

import pytest

from sarizard.analysis.metrics_spec import dataset_of


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("cyp_mt", "cyp"),
        ("cyp1a2_st", "cyp1a2"),
        ("asap_potency_mt", "asap_potency"),
        ("asap_ksol_st_rand", "asap"),
        ("herg_st", "herg"),
        ("biogen_clint_mt", "biogen"),
        ("expansionrx_caco2_efflux_st", "expansionrx"),
    ],
    ids=["cyp", "cyp1a2-before-cyp", "asap_potency-before-asap", "asap", "herg",
         "biogen", "expansionrx"],
)
def test_longest_prefix_wins(recipe: str, expected: str):
    assert dataset_of(recipe) == expected


def test_unknown_recipe_falls_back_to_first_token():
    assert dataset_of("mystery_endpoint_st") == "mystery"
