"""Tests for the flavor registry invariants."""

import pytest

from sarizard.pretraining.flavors import flavor_names, get_flavor


def test_flavor_names_are_unique():
    names = flavor_names()

    assert len(names) == len(set(names))


def test_continuous_and_binary_kinds_are_assigned():
    assert get_flavor("osmordred").kind == "continuous"
    assert get_flavor("ecfp").kind == "binary"


def test_surrogate_runs_in_main_env_from_direct_source():
    flavor = get_flavor("surrogate_adme")

    assert flavor.env == "sarizard"
    assert flavor.source == "direct"


def test_every_flavor_has_a_positive_or_unknown_target_dim():
    # target_dim is None for a flavor whose target is not known until it's computed (e.g. a
    # PCA-compressed variant, where the component count depends on the fitted threshold)
    dims = [get_flavor(name).target_dim for name in flavor_names()]
    assert all(dim is None or dim > 0 for dim in dims)


def test_derived_flavor_has_no_target_dim_and_names_its_base():
    flavor = get_flavor("osmordred_pca80")

    assert flavor.target_dim is None
    assert flavor.derived_from == "osmordred"


def test_non_derived_flavor_has_no_derived_from():
    assert get_flavor("osmordred").derived_from is None


def test_get_flavor_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_flavor("not_a_flavor")


def test_standalone_control_is_excluded_from_flavor_names_but_resolvable():
    # osmordred_surrogate is a standalone control: it must resolve by name (so every
    # --flavor-driven stage works) yet stay out of flavor_names(), which drives the sweep,
    # report cards, and meta-model, so it never lands on a comparable report-card column
    assert get_flavor("osmordred_surrogate").standalone is True
    assert "osmordred_surrogate" not in flavor_names()


def test_osmordred_surrogate_reuses_the_osmordred_calculator_on_the_surrogate_corpus():
    flavor = get_flavor("osmordred_surrogate")

    assert flavor.calculator == "osmordred"
    assert flavor.corpus_from == "surrogate_adme"
    assert flavor.target_dim == get_flavor("osmordred").target_dim


def test_registry_flavors_default_to_non_standalone():
    assert get_flavor("osmordred").standalone is False
    assert get_flavor("osmordred").calculator is None
    assert get_flavor("osmordred").corpus_from is None
