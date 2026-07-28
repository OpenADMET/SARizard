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
>
> **Status:** the sweep is complete. All 15 flavors are evaluated at 5 finetune seeds under all
> three protocols, and the three standalone studies below have landed. Headline numbers and the
> read on each result are in the repo-root `FINDINGS.md`.

## Legend

🟠 reference backbone · 🔵 method, control, or endpoint family · 🟢 significantly above stock · 🟡 no significant difference · 🔴 significantly below stock

Verdicts are Dunnett's test on each flavor's 5 finetune seeds under frozen and reduced, with
the 15 flavors of a protocol corrected together as one family; see `README.md`. No flavor is
🟣 planned any more.

## Artifacts (the center)

- [[Report Card]] 🔵: endpoints by flavors, an R² card and an MAE %-change card per protocol
- [[Meta-Model]] 🔵: stacks per-flavor predictions per endpoint, tests the ensemble lift

## Method

- [[Shared Corpus and Regime]] 🔵: the fixed 944K-molecule corpus and pretraining regime every flavor shares
- [[Finetune Protocols]] 🔵: frozen finetuning, multi-seed foundations, and the LR experiments
- [[Prescaling Ablation]] 🔵: triage that fixes the descriptor preprocessing before the flavor sweep
- [[Stock CheMeleon]] 🟠: the published 1M-PubChem foundation, an external reference column

## Foundation flavors (off [[Shared Corpus and Regime]])

All 15 are pretrained, finetuned at 5 seeds under all three protocols, and evaluated; the color
is each one's verdict against [[Stock CheMeleon]] (see `README.md` for the test).

Physicochemical descriptors:
- [[osmordred]] 🟡: 3585 osmordred 2D descriptors (MSE)
- [[osmordred PCA targets]] 🟡: 70/147/237 PCA components of the above (MSE), three flavors
- [[rdkit2d]] 🟢: 200 RDKit 2D descriptors (MSE)
- [[erg]] 🔴: 315 extended reduced-graph pharmacophore (MSE)
- [[jazzy]] 🟡: 6 hydration-energy and H-bond strengths (MSE)

Fingerprints (binary, leaky pretext):
- [[ecfp]] 🔴: 2048 ECFP4 bits (BCE)
- [[atompair]] 🟡: 2048 atom-pair bits (BCE)
- [[pubchem]] 🟡: 881 PubChem keys (BCE)

3D geometry (conformer-dependent):
- [[usrcat]] 🟡: 60 USRCAT shape and pharmacophore moments (MSE)
- [[whim]] 🔴: 114 WHIM holistic geometry (MSE)
- [[e3fp]] 🔴: 1024 E3FP 3D bits (BCE)

Learned-model targets (the two strongest columns):
- [[minimol]] 🟢: 512 minimol embedding (MSE)
- [[surrogate_adme]] 🟢: 25 surrogate ADME predictions (MSE), a different-corpus reference arm

## Standalone studies (off the [[Report Card]])

- [[osmordred_surrogate]] 🔵: the [[osmordred]] target on [[surrogate_adme]]'s corpus, isolating that flavor's chemical space from its target
- [[External Foundations]] 🔵: four foreign pretrained backbones against [[Stock CheMeleon]], varying the pretraining corpus instead of the target
- [[PXR External Test]] 🔵: every flavor on two fixed external PXR hold-outs, re-testing the sweep's one specialization signal

## Endpoint families

- [[Clearance]] · [[Permeability]] · [[Solubility]] · [[Lipophilicity]] · [[Potency]] · [[CYP Inhibition]] · [[hERG]]
