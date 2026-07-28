---
tags: [flavor, status/green]
---
# surrogate_adme

> **Summary:** 25 surrogate ADME predictions (permeability, clearance, binding, CYP), regressed
> with MSE. The most on-task target: it pretrains the backbone to predict ADME endpoints
> directly. Unlike every other flavor, the Novartis released dataset (273K molecules, 25
> precomputed labels) serves as the pretraining corpus for this flavor rather than the shared
> 944K PubChem corpus.

- Target: 25 continuous values · Loss: MSE · Source: Novartis released CSV (CC BY 4.0)
- Calculator: `sarizard/pretraining/features/surrogate_target.py` (reads CSV directly, no training step)
- Env: `sarizard` (main environment, no isolated env needed)

## Setup

Download the released dataset (not redistributed here):

```
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-49979-3/MediaObjects/41467_2024_49979_MOESM4_ESM.zip
```

Unzip to get `protacdb2.0_zinc_chembl_dataset.csv` and place it at
`cache/surrogate/protacdb2.0_zinc_chembl_dataset.csv` (the default the SLURM pipeline reads;
`cache/` is gitignored). Then:

```
conda activate sarizard
python -m sarizard.pretraining.features.compute_target --flavor surrogate_adme \
    --csv-path cache/surrogate/protacdb2.0_zinc_chembl_dataset.csv
python -m sarizard.pretraining.features.pack_target --flavor surrogate_adme
```

This writes `cache/targets/surrogate_adme/target.npy` (25-dim targets) and
`cache/targets/surrogate_adme/corpus_smiles.parquet` (the 273K canonical SMILES).
The pretrain step uses `corpus_smiles.parquet` in place of the shared corpus.

## Status

Complete: pretrained, finetuned at 5 seeds under all three protocols, and evaluated.

It was once held out of Milestone 7's fan-out over the target-dropout-fraction question. At 25
dims the fixed `DROPOUT_FRACTION=0.85` masked-pretext dropout keeps under one target element per
step on average, which starves supervision. That is now settled by a hard invariant rather than a
per-flavor call: any target at or under 30 dims pretrains at `dropout_fraction=0.0`, enforced in
`train.py`, which rejects a nonzero override for such a flavor. See
[[Shared Corpus and Regime]].

## Hypothesis

Because the target is itself ADME prediction, this should transfer strongly to the matching
families: [[Clearance]], [[Permeability]], and [[CYP Inhibition]]. It is the clearest test of
whether an on-task pretraining target beats generic representations.

## Result (frozen sweep, 5 seeds)
Frozen mean R² **0.370 ± 0.011** against a 0.294 ± 0.010 stock baseline, the largest margin on
the card, and it holds it under reduced (0.374). Under unlocked it is still nominally top at
0.368 against stock's 0.337, but that margin does not survive correcting the 15 flavors as one
family (p=0.11), so no flavor beats stock under unlocked. Beats stock on 27 of 32
endpoint-columns frozen, and is
the single-flavor winner on 18 of them, so it sets the bar the [[Meta-Model]] has to clear.

Read it with the caveat attached: this is a different-corpus reference arm, not an
apples-to-apples column, and the [[osmordred_surrogate]] control attributes most of its lead to
its on-task ADME target rather than its Novartis chemical space. A target that is itself a set of
ADME predictions is closer to distilling an existing ADME model than to learning a general
representation.

## Related

- Learned-model sibling: [[minimol]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
