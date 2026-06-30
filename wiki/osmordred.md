---
tags: [flavor, status/planned]
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

## Related

- Regime: [[Shared Corpus and Regime]] · Reference: [[Stock CheMeleon]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
- The representative continuous flavor for the [[Prescaling Ablation]] triage
