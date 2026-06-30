"""Masked pretraining losses for the CheMeleon descriptor-regression pretext.

A random subset of the descriptor targets contributes to the loss each step, so the
backbone learns to predict targets from the graph rather than memorizing a fixed head.
Continuous targets use masked MSE; binary fingerprint targets use masked BCE on logits.
"""

import torch
from chemprop.nn.metrics import BCELoss, MSE, LossFunctionRegistry, MetricRegistry

from config import DROPOUT_FRACTION


class _RandomDropoutMixin:
    """Mask a random subset of the finite targets out of the loss each step.

    ``DROPOUT_FRACTION`` is the fraction dropped, so a fraction ``1 - DROPOUT_FRACTION``
    of the finite targets is kept and contributes to the loss.
    """

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        lt_mask: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> None:
        # combine the random keep-mask with the incoming finiteness keep-mask via AND: a
        # target enters the loss only if it is finite AND randomly kept. Upstream
        # how-to-train ORs these, which both defeats the dropout on finite targets and
        # re-admits non-finite ones; AND is the correct masked-pretext semantics.
        keep = torch.rand_like(targets) > DROPOUT_FRACTION
        mask = keep if mask is None else torch.logical_and(keep, mask)
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)


@LossFunctionRegistry.register("rdmse")
@MetricRegistry.register("rdmse")
class RandomDropoutMSE(_RandomDropoutMixin, MSE):
    """Masked MSE for continuous descriptor targets."""


@LossFunctionRegistry.register("rdbce")
@MetricRegistry.register("rdbce")
class RandomDropoutBCE(_RandomDropoutMixin, BCELoss):
    """Masked BCE (on logits) for binary fingerprint targets."""
