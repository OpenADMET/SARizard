"""Tests for recipe-to-dataset mapping (longest-prefix match)."""

import pytest

from sarizard.analysis.metrics_spec import dataset_of


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("cyp_mt", "openadmet_cyp"),
        ("cyp1a2_st", "openadmet_cyp"),
        ("asap_potency_mt", "asap_potency"),
        ("asap_ksol_st_rand", "asap"),
        ("herg_st", "herg"),
        ("biogen_clint_mt", "biogen"),
        ("expansionrx_caco2_efflux_st", "expansionrx"),
    ],
    ids=["cyp_mt-openadmet_cyp", "cyp1a2_st-openadmet_cyp", "asap_potency-before-asap",
         "asap", "herg", "biogen", "expansionrx"],
)
def test_prefix_rule_maps_recipe_to_dataset(recipe: str, expected: str):
    assert dataset_of(recipe) == expected


def test_both_cyp_recipes_share_the_openadmet_cyp_group():
    # the single-task cyp1a2 recipe and the multi-task cyp recipe belong to one dataset group
    assert dataset_of("cyp1a2_st") == dataset_of("cyp_mt") == "openadmet_cyp"


def test_unknown_recipe_falls_back_to_first_token():
    assert dataset_of("mystery_endpoint_st") == "mystery"
