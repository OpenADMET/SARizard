---
tags: [flavor, status/yellow]
---
# jazzy

> **Summary:** 6 Jazzy descriptors: hydrogen-bond donor and acceptor strengths and the apolar,
> polar, and total free energy of hydration (sdc, sdx, sa, dga, dgp, dgtot), regressed with MSE.
> A small, physically grounded solvation target.

- Target: 6 continuous values · Loss: MSE · Source: direct compute (isolated env, pins rdkit==2024.3.1)
- Calculator: `sarizard/pretraining/features/jazzy_target.py`

## Hypothesis

Solvation and H-bonding govern membrane crossing and dissolution, so this should serve
[[Permeability]] and [[Solubility]]. The tiny target dimensionality also tests whether a few
mechanistically chosen values can rival wide descriptor blocks.

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.305 ± 0.009 against a 0.294 ± 0.010 stock baseline, not significant
(family-wise p=0.85), and 0.343 under reduced, which is the best any flavor outside the top three
manages but still short of significance once corrected (p=0.07). Beats stock on 17 of 32
endpoint-columns frozen and
takes [[Potency]] outright (0.558 against stock 0.490), the best result for a 6-dim target on any
family.

Its 250K-era result predates the target-dropout invariant and was run under the fixed 0.85
masked-pretext dropout, which at 6 dims meant under one supervised target per step. Under the
full-corpus rerun it pretrains at `dropout_fraction=0.0` per the enforced sub-threshold rule, so
the current number is the one to quote.

## Related

- Regime: [[Shared Corpus and Regime]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
