---
tags: [flavor, status/yellow]
---
# osmordred

> **Summary:** The richest physicochemical target: 3585 osmordred 2D descriptors (a performant
> Mordred reimplementation), regressed with MSE. The closest in-study analog to the Mordred
> block [[Stock CheMeleon]] learns, computed on [[Shared Corpus and Regime]].

- Target: 3585 continuous descriptors · Loss: MSE · Source: direct compute (isolated env)
- Calculator: `sarizard/pretraining/features/osmordred_target.py`

## Hypothesis

Broad physicochemical coverage should transfer to property-driven endpoints: [[Solubility]],
[[Lipophilicity]], [[Permeability]], and [[Clearance]]. The high target dimensionality is the
strongest test of whether a wide descriptor block helps or just adds noise.

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.301 ± 0.013 against a 0.294 ± 0.010 stock baseline, **not** a significant
margin (family-wise p=0.99); reduced 0.341 is nominally ahead but also not significant (p=0.12),
and unlocked 0.298 is significantly below (p=0.024). Beats stock on 19 of 32 endpoint-columns
frozen. Its reduced margin looked significant before the flavors were corrected as one family.

The widest target in the study (3585 dims) does not buy a better foundation than
[[rdkit2d]]'s 200, and the [[osmordred PCA targets]] show it does not need its width either: 70
components transfer just as well.

## Related

- Regime: [[Shared Corpus and Regime]] · Reference: [[Stock CheMeleon]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
- The representative continuous flavor for the [[Prescaling Ablation]] triage
