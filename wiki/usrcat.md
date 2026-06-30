---
tags: [flavor, status/planned]
---
# usrcat

> **Summary:** 60 USRCAT descriptors: ultrafast shape recognition moments augmented with
> pharmacophore channels, regressed with MSE. A 3D target, so it depends on a generated
> conformer (RDKit ETKDG + MMFF94, seeded) and is only approximately reproducible.

- Target: 60 continuous values · Loss: MSE · Source: direct compute, conformers (scikit-fingerprints)
- Calculator: `pretraining/features/skfp_targets.py`

## Hypothesis

3D shape plus pharmacophore should favor binding-shape endpoints: [[Potency]], [[hERG]], and
[[CYP Inhibition]]. Tests whether injecting 3D shape into a 2D backbone helps where geometry
matters.

## Related

- 3D siblings: [[whim]], [[e3fp]] · conformer note in `pretraining/config.py`
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
