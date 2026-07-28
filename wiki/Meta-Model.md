---
tags: [method, status/blue]
---
# Meta-Model

> **Summary:** The second north-star artifact. For each endpoint it stacks the per-flavor
> finetuned predictions into a small model (LGBM, random forest, or MLP) and compares its
> cross-validated score to the best single flavor. It answers: does an ensemble of foundations
> beat the best single foundation.

## How it is built

- `sarizard/analysis/meta_model.py` reads the cached per-flavor test predictions (the `y_pred.npy`
  files written by [[Report Card]]'s evaluate step) and, per endpoint, builds a feature matrix
  of one column per flavor.
- Seeds are handled by **scoring each seed independently and averaging the stacker's R² across
  them**, not by averaging the flavors' predictions first. Averaging first was the original
  behavior and was wrong: the seed-randomized multi-task datasets (`chembl_clint_mt`, `cyp_mt`)
  resample their test split per finetune seed, so the per-seed prediction vectors differ in
  length. Scoring per seed also gives the meta-model column the same seed error bars every
  report-card cell carries (`meta_r2_std`, `delta_r2_std`, `n_seeds` in the CSV).
- The stacker is cross-validated over the test molecules, so its reported score is out-of-fold
  and never trained on the rows it scores. The best-single-flavor baseline is scored the same
  way: the winning flavor is chosen on each fold's training rows and scored on its held-out
  rows, so selecting the baseline never peeks at the labels the stacker is graded on.
- Output: a per-endpoint CSV of meta vs best-single R², a lift bar chart, and a summary of how
  often the ensemble wins.

## Leakage discipline

The flavor models never trained on the test molecules, and the meta-model is the only thing
fit here, so the cross-validation makes the comparison leakage-safe. Because the best-single
baseline is also selected fold-wise (not by peeking at the full test set), both sides of the
lift are out-of-fold; the ensemble pays a CV cost the fixed single-flavor vectors do not, so a
positive lift is conservative.

## Result

**Yes, the ensemble beats the best single foundation.** Frozen protocol, all 15 flavors, 5 seeds:
LGBM wins 21 of 32 endpoint-columns, mean R² 0.500 against 0.417 for the best single flavor per
endpoint (mean delta +0.082), and 18 of the 21 wins exceed one standard deviation of their own
delta. `results/meta_model_lgbm.csv` holds the per-endpoint table.

The bar it clears is set mostly by [[surrogate_adme]], the single-flavor winner on 18 of 32
columns, which is itself the different-corpus reference arm, so the lift is measured against a
column that is not apples-to-apples with the rest.

Gains are largest where no single flavor does well (Caco-2 efflux 0.710 against 0.332; ChEMBL RLM
clearance 0.376 against 0.096) and it loses where one already does (Biogen solubility 0.106
against 0.210). Stacking recovers signal spread thinly across flavors and dilutes signal
concentrated in one.

Not run: the RF and MLP estimators, and the reduced/unlocked meta-models. The per-mode path is
wired (`meta_model.py --lr-mode`) but never launched, so the ensemble question is answered under
frozen finetuning only. Since reduced is the protocol where pretraining pays for single flavors,
the reduced meta-model is the one most worth running.

## Related

- Consumes predictions cached by [[Report Card]].
- Stacks all flavors; see the flavor nodes under [[_index]].
