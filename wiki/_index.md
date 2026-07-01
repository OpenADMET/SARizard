---
tags: [moc]
---
# Index: fit-to-purpose foundation flavors

> **Summary:** Map of the study into whether the descriptor target a CheMeleon-style D-MPNN
> is pretrained to regress determines which ADMET endpoints it serves best, and whether an
> ensemble of foundations beats the best single one. Every flavor shares one corpus and one
> regime ([[Shared Corpus and Regime]]); only the target block differs. The [[Report Card]]
> ranks flavors per endpoint; the [[Meta-Model]] tests the ensemble question. Open Graph view
> to see it (`_index` and `README` are filtered out).

## Legend

🟠 reference backbone · 🔵 method or endpoint family · 🟣 planned flavor · 🟢 wins · 🟡 competitive · 🔴 underperforms

## Artifacts (the center)

- [[Report Card]] 🔵 — heatmap of endpoints by flavors, one selectable metric (default R²)
- [[Meta-Model]] 🔵 — stacks per-flavor predictions per endpoint, tests the ensemble lift

## Method

- [[Shared Corpus and Regime]] 🔵 — the fixed 250K corpus and pretraining regime every flavor shares
- [[Finetune Protocols]] 🔵 — frozen finetuning, multi-seed foundations, and the LR experiments
- [[Prescaling Ablation]] 🔵 — triage that fixes the descriptor preprocessing before the flavor sweep
- [[Stock CheMeleon]] 🟠 — the published 1M-PubChem foundation, an external reference column

## Foundation flavors (off [[Shared Corpus and Regime]])

Physicochemical descriptors:
- [[osmordred]] 🟣 — 3585 osmordred 2D descriptors (MSE)
- [[rdkit2d]] 🟣 — 200 RDKit 2D descriptors (MSE)
- [[erg]] 🟣 — 315 extended reduced-graph pharmacophore (MSE)
- [[jazzy]] 🟣 — 6 hydration-energy and H-bond strengths (MSE)

Fingerprints (binary, leaky pretext):
- [[ecfp]] 🟣 — 2048 ECFP4 bits (BCE)
- [[atompair]] 🟣 — 2048 atom-pair bits (BCE)
- [[pubchem]] 🟣 — 881 PubChem keys (BCE)

3D geometry (conformer-dependent):
- [[usrcat]] 🟣 — 60 USRCAT shape and pharmacophore moments (MSE)
- [[whim]] 🟣 — 114 WHIM holistic geometry (MSE)
- [[e3fp]] 🟣 — 1024 E3FP 3D bits (BCE)

Learned-model targets:
- [[minimol]] 🟣 — 512 minimol embedding (MSE)
- [[surrogate_adme]] 🟣 — 25 surrogate ADME predictions (MSE)
- [[ml_qm]] 🟣 — 24 pooled qmdesc QM descriptors (MSE)

## Endpoint families

- [[Clearance]] · [[Permeability]] · [[Solubility]] · [[Lipophilicity]] · [[Potency]] · [[CYP Inhibition]] · [[hERG]]
