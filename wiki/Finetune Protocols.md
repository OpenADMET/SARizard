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

**Every finetuning submission is 5-seed and finetune-only** (standing decision): seeds 1-5 run
off the one fixed seed-42 foundation per flavor (`FOUNDATION_SEED=42 FLAVOR_SEEDS="1 2 3 4 5"`),
so each report-card cell carries an error bar. Do not submit single-seed s42 runs; the earlier
single-seed frozen and reduced results stand as historical and are not extended.

The seeds vary head initialization and training, not the representation, so the spread is
**finetune** variance, and that is the noise floor any per-flavor difference must clear. The
mechanism supports the other reading too (each `(flavor, seed)` can be a separate pretraining
run exporting `foundations/<flavor>__s<seed>_mp.pt`), but foundation-initialization variance is
not what the sweep measures: one foundation per flavor, five finetunes each. The
[[Report Card]] collapses the `<flavor>__s<seed>` variants into one averaged column; the
[[Meta-Model]] scores each seed separately and averages the scores.

Stages are resumable per seed via a skip-if-exists guard on each result directory, so re-running
with more seeds computes only the new ones. That guard keys on the directory, not on a completed
`model.pth`, so a crashed task leaves a partial directory that a rerun would silently skip;
delete the partial directory before resubmitting.

## Learning-rate experiments

The LR experiments reuse the frozen sweep's foundations and repeat the finetuning with the
backbone allowed to move, to measure what the frozen protocol costs:

- `reduced` (`mpnn_lr = ffn_lr / 10`): partial unfreezing. Does letting the backbone drift a
  little recover endpoints where the frozen backbone underperforms, or just reintroduce the
  initialization-washing problem?
- `unlocked` (`mpnn_lr = ffn_lr`): full finetuning, the upper bound. The gap between frozen and
  unlocked was expected to be the price of the clean ablation.

`bash slurm/run_lr_experiments.sh` generates `configs/lr_<mode>__<flavor>__s<seed>/` recipes off
the existing foundations, finetunes them, and `sarizard/analysis/lr_report.py` compares each mode
against the frozen sweep per (flavor, endpoint) in `plots/lr_ranking_r2.csv` (mean R² delta and
win count). A frozen warmup then coadaptation protocol is planned but not scripted: it needs a
two-phase training schedule the anvil config cannot express yet.

**Which protocol pays, measured (5 seeds, all 15 flavors, Dunnett-corrected per protocol).**
`reduced` is where descriptor pretraining shows up: four flavors clear the [[Stock CheMeleon]]
baseline significantly there against three under frozen, and the same three ([[surrogate_adme]],
[[minimol]], [[rdkit2d]]) lead both. `unlocked` is where it disappears: the stock baseline rises
to 0.337 mean R², the best column on the whole card, **no flavor clears it significantly**, and
five fall significantly below. Paying for a full-network finetune recovers the frozen result
rather than improving on it, which is what a protocol that overwrites the pretrained
representation should do. So the gap between frozen and unlocked is not the price of the clean
ablation; it is close to zero on average, and the useful protocol is the one in between.

The same three protocols also drive the [[Prescaling Ablation]] triage. `generate.py
--mpnn-lr-mode` threads the protocol through ablation mode, so each prescaling recipe was
finetuned frozen, `reduced`, and `unlocked` from its own foundation
(`configs/ablation_<name>__s42__{reduced,unlocked}/`). This checked whether the preprocessing
choice holds under a moving backbone, not just a frozen one; it does not fully hold (the
winning recipe shifts by protocol, though the margin between the top two is narrow), see
[[Prescaling Ablation]] and `FINDINGS.md`.

## Related

- Produces the columns of the [[Report Card]] and the features of the [[Meta-Model]].
- Runs on foundations built off [[Shared Corpus and Regime]].
- The [[Prescaling Ablation]] triage is the analogous controlled experiment on the pretraining
  side (varying the target preprocessing instead of the finetune protocol).
