"""Masked pretraining losses for the CheMeleon descriptor-regression pretext.

A random subset of the descriptor targets contributes to the loss each step, so the
backbone learns to predict targets from the graph rather than memorizing a fixed head.
Continuous targets use masked MSE; binary fingerprint targets use masked BCE on logits.
"""

import torch
from chemprop.nn.metrics import BCELoss, MSE, LossFunctionRegistry, MetricRegistry

# dual import: script-style when run from pretraining/ (sbatch), package-style when imported
# from the repo root (tests)
try:
    from config import DROPOUT_FRACTION
except ImportError:
    from sarizard.pretraining.config import DROPOUT_FRACTION


class _RandomDropoutMixin:
    """Mask a random subset of the finite targets out of the loss each step.

    ``dropout_fraction`` is the fraction dropped, so a fraction ``1 - dropout_fraction``
    of the finite targets is kept and contributes to the loss. Defaults to the shared
    regime constant ``DROPOUT_FRACTION``; pass an explicit value to override it for a
    single flavor (e.g. narrow targets where the regime default keeps under one target
    per step) without touching the constant every other flavor relies on. ``clone()``
    (``torchmetrics.Metric.clone``, used by chemprop to snapshot the criterion as a
    validation metric) is a ``deepcopy``, so this instance attribute survives cloning.
    """

    def __init__(self, *args, dropout_fraction: float = DROPOUT_FRACTION, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout_fraction = dropout_fraction

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        lt_mask: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> None:
        # apply the random dropout only while training. As a torch.nn.Module the metric clone
        # inherits ``self.training``, which Lightning sets to False during validation; gating on
        # it keeps the monitored validation metric deterministic (binary flavors monitor
        # val_loss, which is this masked BCE) while still dropping targets during training.
        # Combine the random keep-mask with the incoming finiteness keep-mask via AND: a target
        # enters the loss only if it is finite AND randomly kept. Upstream how-to-train ORs
        # these, which both defeats the dropout on finite targets and re-admits non-finite ones;
        # AND is the correct masked-pretext semantics.
        if self.training:
            keep = torch.rand_like(targets) > self.dropout_fraction
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
