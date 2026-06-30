"""Tests for the flavor registry invariants."""

import pytest

from pretraining.flavors import flavor_names, get_flavor


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


def test_every_flavor_has_a_positive_target_dim():
    assert all(get_flavor(name).target_dim > 0 for name in flavor_names())


def test_get_flavor_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_flavor("not_a_flavor")
