---
tags: [flavor, status/planned]
---
# surrogate_adme

> **Summary:** 25 surrogate ADME predictions (permeability, clearance, binding, CYP), regressed
> with MSE. The most on-task target: it pretrains the backbone to predict ADME endpoints
> directly. Unlike every other flavor, the Novartis released dataset (273K molecules, 25
> precomputed labels) serves as the pretraining corpus for this flavor rather than the shared
> 250K PubChem set.

- Target: 25 continuous values · Loss: MSE · Source: Novartis released CSV (CC BY 4.0)
- Calculator: `pretraining/features/surrogate_target.py` (reads CSV directly, no training step)
- Env: `sarizard` (main environment, no isolated env needed)

## Setup

Download the released dataset (not redistributed here):

```
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-49979-3/MediaObjects/41467_2024_49979_MOESM4_ESM.zip
```

Unzip to get `protacdb2.0_zinc_chembl_dataset.csv`, then:

```
conda activate sarizard
python -m pretraining.features.compute_target --flavor surrogate_adme \
    --csv-path /path/to/protacdb2.0_zinc_chembl_dataset.csv
python -m pretraining.features.pack_target --flavor surrogate_adme
```

This writes `cache/targets/surrogate_adme/target.npy` (25-dim targets) and
`cache/targets/surrogate_adme/corpus_smiles.parquet` (the 273K canonical SMILES).
The pretrain step uses `corpus_smiles.parquet` in place of the shared 250K corpus.

## Hypothesis

Because the target is itself ADME prediction, this should transfer strongly to the matching
families: [[Clearance]], [[Permeability]], and [[CYP Inhibition]]. It is the clearest test of
whether an on-task pretraining target beats generic representations.

## Related

- Learned-model siblings: [[minimol]], [[ml_qm]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
