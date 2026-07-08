"""Tests for the skfp target builders and, in particular, their env isolation contract."""

from __future__ import annotations

import builtins
import importlib

import pytest

from sarizard.pretraining.features import skfp_targets
from sarizard.pretraining.flavors import flavor_names


def test_is_skfp_flavor_partitions_the_registry() -> None:
    skfp = {n for n in flavor_names() if skfp_targets.is_skfp_flavor(n)}
    assert skfp == {"rdkit2d", "erg", "ecfp", "atompair", "pubchem", "usrcat", "whim", "e3fp"}


def test_is_skfp_flavor_does_not_import_skfp(monkeypatch: pytest.MonkeyPatch) -> None:
    # reproduce the isolated target environments (osmordred, minimol, jazzy), which
    # dispatch through is_skfp_flavor but have no skfp installed; the probe must not import it
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "skfp" or name.startswith("skfp."):
            raise ModuleNotFoundError("No module named 'skfp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    assert skfp_targets.is_skfp_flavor("osmordred") is False
    assert skfp_targets.is_skfp_flavor("ecfp") is True


def test_register_builders_matches_declared_flavor_sets() -> None:
    # _register_builders raises if the built tables drift from the static membership sets,
    # so importing skfp and populating them here is the drift guard's own regression test
    pytest.importorskip("skfp")
    importlib.reload(skfp_targets)
    skfp_targets._register_builders()

    assert skfp_targets._2D_BUILDERS.keys() == skfp_targets._2D_FLAVORS
    assert skfp_targets._3D_BUILDERS.keys() == skfp_targets._3D_FLAVORS
