---
tags: [flavor, status/green]
---
# rdkit2d

> **Summary:** 200 curated RDKit 2D physicochemical descriptors, regressed with MSE. A compact
> physchem target, the low-dimensional counterpart to [[osmordred]].

- Target: 200 continuous descriptors · Loss: MSE · Source: direct compute (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

A focused physchem block should serve [[Solubility]] and [[Lipophilicity]] well, and tests
whether 200 well-chosen descriptors match the much wider [[osmordred]] target on
[[Permeability]] and [[Clearance]].

## Result (frozen sweep, 5 seeds)
Frozen mean R² **0.323 ± 0.011** against a 0.294 ± 0.010 stock baseline, significant, and
0.356 under reduced; unlocked drops it to a statistical tie with stock. Beats stock on 24 of 32
endpoint-columns frozen, its largest single margin being ASAP HLM clearance (+0.36). The
strongest direct-compute descriptor block, confirming that 200 well-chosen descriptors match or
beat [[osmordred]]'s 3585.

One caution carried over from the [[PXR External Test]]: `rdkit2d` led the sweep's internal
Butina-split PXR column, the cleanest specialization signal the study produced, but sits mid-pack
and below stock on both fixed external PXR hold-outs. That specialization did not transfer.

## Related

- Regime: [[Shared Corpus and Regime]] · Wider sibling: [[osmordred]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
