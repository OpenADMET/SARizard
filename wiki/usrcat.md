---
tags: [flavor, status/yellow]
---
# usrcat

> **Summary:** 60 USRCAT descriptors: ultrafast shape recognition moments augmented with
> pharmacophore channels, regressed with MSE. A 3D target, so it depends on a generated
> conformer (RDKit ETKDG + MMFF94, seeded) and is only approximately reproducible.

- Target: 60 continuous values · Loss: MSE · Source: direct compute, conformers (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

3D shape plus pharmacophore should favor binding-shape endpoints: [[Potency]], [[hERG]], and
[[CYP Inhibition]]. Tests whether injecting 3D shape into a 2D backbone helps where geometry
matters.

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.285 ± 0.018 against a 0.294 ± 0.010 stock baseline, a statistical tie, and it
neither clears nor falls significantly below stock under any protocol. Beats stock on 14 of 32
endpoint-columns frozen. The most thoroughly unremarkable column on the card.

The 250K partial sweep had `usrcat` winning [[Potency]] and [[hERG]] specifically, which read as
the study's one 3D-shape specialization signal. That does not reappear on the full-corpus 5-seed
card, where [[jazzy]] takes potency and [[minimol]] takes hERG. Treat the earlier read as a
single-seed, smaller-corpus artifact.

## Related

- 3D siblings: [[whim]], [[e3fp]] · conformer note in `sarizard/pretraining/config.py`
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
