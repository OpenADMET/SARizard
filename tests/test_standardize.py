"""Tests for the shared salt/solvent stripping used by the corpus and surrogate builders."""

from sarizard.standardize import standardize_to_canonical


def test_strips_salt_to_largest_fragment():
    canonical, stripped = standardize_to_canonical("CC(=O)[O-].[Na+]")

    assert canonical == "CC(=O)[O-]"
    assert stripped is True


def test_strips_solvent_to_largest_fragment():
    # benzene with a water of crystallization keeps the benzene
    canonical, stripped = standardize_to_canonical("c1ccccc1.O")

    assert canonical == "c1ccccc1"
    assert stripped is True


def test_single_fragment_is_only_canonicalized():
    canonical, stripped = standardize_to_canonical("OCC")

    assert canonical == "CCO"
    assert stripped is False


def test_invalid_smiles_returns_none():
    canonical, stripped = standardize_to_canonical("not_a_molecule")

    assert canonical is None
    assert stripped is False
