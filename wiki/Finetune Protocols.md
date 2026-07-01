---
tags: [method, status/blue]
---
# Finetune Protocols

> **Summary:** How each foundation is finetuned onto the ADMET endpoints, and the knobs that
> turn the sweep into controlled experiments. By default the MPNN backbone is frozen and only
> the FFN head trains, so the [[Report Card]] measures representation quality rather than
> initialization luck. Two variations sit on top of the same machinery: pretraining at several
> seeds, and unfreezing the backbone at different learning rates.

## Frozen finetuning (the default sweep)

Every generated recipe sets `mpnn_lr: 0`, so finetuning adapts only the head against the fixed
pretrained representation. This is the clean setting for the fit-to-purpose question: a
flavor's column reflects what its pretraining target learned, not how well a random head
happened to co-adapt. Random-init and CheMeleon-init backbones often finetune to similar
scores, which is exactly why the backbone is frozen here.

## Seeds

Each `(flavor, seed)` is a separate pretraining run exporting `foundations/<flavor>__s<seed>_mp.pt`
with its own recipes and results. Set `FLAVOR_SEEDS` (default one seed) for `run_all.sh` to
sweep several; the [[Report Card]] and [[Meta-Model]] average the seed variants back to one
column or feature per flavor. Only pretraining is seeded (the finetune seed is fixed), so the
spread is foundation-initialization variance, the noise floor any real per-flavor difference
must clear. The stages are resumable per seed: run one seed, then re-run with more and only the
new seeds compute.

## Learning-rate experiments

The LR experiments reuse the frozen sweep's foundations and repeat the finetuning with the
backbone allowed to move, to measure what the frozen protocol costs:

- `reduced` — `mpnn_lr = ffn_lr / 10`: partial unfreezing. Does letting the backbone drift a
  little recover endpoints where the frozen backbone underperforms, or just reintroduce the
  initialization-washing problem?
- `unlocked` — `mpnn_lr = ffn_lr`: full finetuning, the upper bound. The gap between frozen and
  unlocked is the price of the clean ablation.

`bash slurm/run_lr_experiments.sh` generates `configs/lr_<mode>__<flavor>__s<seed>/` recipes off
the existing foundations, finetunes them, and `sarizard/analysis/lr_report.py` compares each mode
against the frozen sweep per (flavor, endpoint) in `plots/lr_ranking_r2.csv` (mean R² delta and
win count). A frozen warmup then coadaptation protocol is planned but not scripted: it needs a
two-phase training schedule the anvil config cannot express yet.

The same three protocols also drive the [[Prescaling Ablation]] triage. `generate.py
--mpnn-lr-mode` threads the protocol through ablation mode, so each prescaling recipe can be
finetuned frozen, `reduced`, and `unlocked` from its own foundation
(`configs/ablation_<name>__s42__{reduced,unlocked}/`). This checks that the preprocessing choice
holds under a moving backbone, not just a frozen one.

## Related

- Produces the columns of the [[Report Card]] and the features of the [[Meta-Model]].
- Runs on foundations built off [[Shared Corpus and Regime]].
- The [[Prescaling Ablation]] triage is the analogous controlled experiment on the pretraining
  side (varying the target preprocessing instead of the finetune protocol).
