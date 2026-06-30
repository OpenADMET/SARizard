---
tags: [flavor, status/planned]
---
# jazzy

> **Summary:** 6 Jazzy descriptors: hydrogen-bond donor and acceptor strengths and the apolar,
> polar, and total free energy of hydration (sdc, sdx, sa, dga, dgp, dgtot), regressed with MSE.
> A small, physically grounded solvation target.

- Target: 6 continuous values · Loss: MSE · Source: direct compute (isolated env, pins rdkit==2024.3.1)
- Calculator: `pretraining/features/jazzy_target.py`

## Hypothesis

Solvation and H-bonding govern membrane crossing and dissolution, so this should serve
[[Permeability]] and [[Solubility]]. The tiny target dimensionality also tests whether a few
mechanistically chosen values can rival wide descriptor blocks.

## Related

- Regime: [[Shared Corpus and Regime]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
