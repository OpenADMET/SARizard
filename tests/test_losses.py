"""Tests for the masked-dropout loss mixin: AND with finiteness, training-mode gating."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from sarizard.pretraining.losses import _RandomDropoutMixin  # noqa: E402


class _Recorder:
    """Minimal update target that records the mask the mixin forwards to it."""

    def __init__(self):
        self.last_mask = "unset"

    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        self.last_mask = mask


class _Probe(_RandomDropoutMixin, _Recorder):
    """Mixin over the recorder, with an explicit training flag (real metrics inherit it)."""

    def __init__(self, *, training):
        super().__init__()
        self.training = training


def test_training_dropout_is_anded_with_finiteness():
    torch.manual_seed(0)
    probe = _Probe(training=True)
    finite = torch.ones(1000, 4, dtype=torch.bool)
    finite[:, 0] = False  # an entire column is non-finite upstream

    probe.update(torch.zeros(1000, 4), torch.zeros(1000, 4), mask=finite)

    # the combined mask never re-admits a non-finite target, and random dropout removes some
    # of the finite ones (AND of finiteness and the keep-mask, not OR)
    assert not probe.last_mask[~finite].any()
    assert probe.last_mask.sum() < finite.sum()


def test_validation_keeps_all_finite_targets():
    probe = _Probe(training=False)
    finite = torch.ones(10, 3, dtype=torch.bool)
    finite[0, 0] = False

    probe.update(torch.zeros(10, 3), torch.zeros(10, 3), mask=finite)

    # eval mode applies no random dropout, so the finiteness mask passes through unchanged
    assert torch.equal(probe.last_mask, finite)


def test_validation_without_mask_stays_none():
    probe = _Probe(training=False)

    probe.update(torch.zeros(5, 2), torch.zeros(5, 2))

    assert probe.last_mask is None
