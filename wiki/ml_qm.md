---
tags: [flavor, status/planned]
---
# ml_qm

> **Summary:** 24 pooled QM descriptors, regressed with MSE. qmdesc predicts atom-level
> (partial charge, nucleophilic and electrophilic Fukui, NMR shielding) and bond-level (bond
> order, bond length) quantum-chemical descriptors; each is pooled (mean, std, min, max) to a
> fixed 24-dim molecule vector.

- Target: 24 continuous values · Loss: MSE · Source: learned model (isolated env, qmdesc)
- Calculator: `sarizard/pretraining/features/qmdesc_target.py`

## Status

Held out of Milestone 7's fan-out (unlike its sibling [[minimol]], which is running). The
target-dropout-fraction blocker in `TODO.md` (Future experiments) names this flavor directly:
at 24 dims, the fixed `DROPOUT_FRACTION=0.85` masked-pretext dropout keeps under 1 target
element per step on average, and the item calls for ablating that fraction before fan-out
rather than only "if it underperforms." Pending that ablation, or an explicit decision to
defer it the way Milestone 6 deferred it for [[jazzy]].

## Hypothesis

Electronic structure and reactivity drive metabolism and covalent liability, so this may favor
[[CYP Inhibition]] and [[Clearance]] (metabolic lability), and possibly [[hERG]]. Pooling
discards per-atom resolution; the per-atom node/edge target is a noted follow-up in `TODO.md`.

## Related

- Learned-model siblings: [[minimol]], [[surrogate_adme]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
