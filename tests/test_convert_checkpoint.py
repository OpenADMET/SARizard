"""Tests for the foundation export format openadmet-models consumes.

The compatibility invariant is that a foundation file is a ``{hyper_parameters, state_dict}``
dict of plain scalars, loadable under ``torch.load(weights_only=True)`` and rebuildable as a
``BondMessagePassing`` block. These guard that contract against drift.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from sarizard.pretraining.convert_checkpoint import (  # noqa: E402
    extract_foundation,
    save_foundation,
)


def _tiny_mpnn():
    """Build a minimal MPNN with the SARizard compatibility invariants (mean aggregation)."""
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

    mp = BondMessagePassing(d_h=64, depth=2)
    return MPNN(mp, MeanAggregation(), predictor=RegressionFFN(n_tasks=3, input_dim=64))


def test_foundation_is_weights_only_loadable_and_rebuildable(tmp_path):
    from chemprop.nn import BondMessagePassing

    out = save_foundation(_tiny_mpnn(), tmp_path / "foundation.pt")

    loaded = torch.load(out, weights_only=True)
    assert set(loaded) == {"hyper_parameters", "state_dict"}
    # only plain scalars survive, so weights_only loading accepts the file
    assert all(
        isinstance(v, (int, float, str, bool)) for v in loaded["hyper_parameters"].values()
    )
    # openadmet rebuilds the block as BondMessagePassing(**hyper_parameters)
    BondMessagePassing(**loaded["hyper_parameters"])


def test_extract_foundation_rejects_non_bond_message_passing():
    class _NotBondMessagePassing:
        pass

    class _FakeMPNN:
        message_passing = _NotBondMessagePassing()

    # openadmet rebuilds the foundation as BondMessagePassing, so a different block must fail
    with pytest.raises(TypeError):
        extract_foundation(_FakeMPNN())
