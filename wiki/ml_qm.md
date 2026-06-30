---
tags: [flavor, status/planned]
---
# ml_qm

> **Summary:** 24 pooled QM descriptors, regressed with MSE. qmdesc predicts atom-level
> (partial charge, nucleophilic and electrophilic Fukui, NMR shielding) and bond-level (bond
> order, bond length) quantum-chemical descriptors; each is pooled (mean, std, min, max) to a
> fixed 24-dim molecule vector.

- Target: 24 continuous values · Loss: MSE · Source: learned model (isolated env, qmdesc)
- Calculator: `pretraining/features/qmdesc_target.py`

## Hypothesis

Electronic structure and reactivity drive metabolism and covalent liability, so this may favor
[[CYP Inhibition]] and [[Clearance]] (metabolic lability), and possibly [[hERG]]. Pooling
discards per-atom resolution; the per-atom node/edge target is a noted follow-up in `TODO.md`.

## Related

- Learned-model siblings: [[minimol]], [[surrogate_adme]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
